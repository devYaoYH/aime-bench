"""Run restricted arithmetic Python in an isolated macOS worker process.

Use execute_python from native tool-call loops when generated computations
need deterministic standard-library math, bounded output, and CPU/time limits.
AST validation rejects unsupported language; sandbox-exec denies credentials,
repository access, network, and writes. Unsupported programs return tool errors.
The --worker entry point is an internal subprocess protocol, not a benchmark CLI.
Execution requires macOS with sandbox-exec and fails closed when unavailable.
"""
from __future__ import annotations

import ast
import asyncio
import builtins
import contextlib
import importlib
import json
from pathlib import Path
import resource
import sys
import time

ALLOWED_MODULES = {'math', 'itertools', 'functools', 'collections', 'fractions', 'decimal'}
BUILTIN_NAMES = ('abs','all','any','bool','dict','divmod','enumerate','filter','float',
                 'frozenset','int','isinstance','len','list','map','max','min','next',
                 'pow','print','range','repr','reversed','round','set','slice','sorted',
                 'str','sum','tuple','zip','Exception','ValueError','AssertionError')
CODE_LIMIT = 20000
OUTPUT_LIMIT = 6000
TOOL = {'type':'function','function':{
    'name':'python_math',
    'description':('Execute a short Python program for exact arithmetic, finite enumeration, or dynamic programming. '
                   'Print your result and compact checks. No files, network, subprocesses, or package installation. '
                   'Imports allowed: math, itertools, functools, collections, fractions, decimal. '
                   'Each call is independent; variables are not preserved. CPU time 3 seconds; wall time 5 seconds.'),
    'parameters':{'type':'object','properties':{'code':{'type':'string','description':'Python program to execute.'}},
                  'required':['code'],'additionalProperties':False}}}


def validate_code(code):
    if not isinstance(code,str) or not code.strip() or len(code)>CODE_LIMIT:
        raise ValueError('Expected a nonempty Python program of at most 20000 characters')
    tree=ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            if any(alias.name not in ALLOWED_MODULES for alias in node.names):
                raise ValueError('Only the listed mathematical standard-library modules may be imported')
        if isinstance(node,ast.ImportFrom):
            if node.level or node.module not in ALLOWED_MODULES or any(a.name=='*' or a.name.startswith('_') for a in node.names):
                raise ValueError('Unsupported import')
        if isinstance(node,ast.Attribute) and node.attr.startswith('_'):
            raise ValueError('Private attributes are unavailable')
        if isinstance(node,ast.Name) and node.id.startswith('__') and node.id!='__name__':
            raise ValueError('Private names are unavailable')
        if isinstance(node,ast.Name) and node.id in {'open','eval','exec','compile','globals','locals','vars','getattr','setattr','delattr','breakpoint','input','help'}:
            raise ValueError(f'{node.id} is unavailable')
        if isinstance(node,(ast.ClassDef,ast.AsyncFunctionDef,ast.Await,ast.Global,ast.Nonlocal)):
            raise ValueError('Classes, async code, and global/nonlocal declarations are unavailable')
    return tree


class BoundedOutput:
    def __init__(self): self.parts=[]; self.size=0
    def write(self,value):
        available=OUTPUT_LIMIT-self.size
        if len(value)>available:
            self.parts.append(value[:available]); self.size=OUTPUT_LIMIT
            raise ValueError('Tool output limit exceeded; print compact results')
        self.parts.append(value); self.size+=len(value)
        return len(value)
    def flush(self): pass
    def value(self): return ''.join(self.parts)


def worker(code):
    output=BoundedOutput()
    result={'ok':False,'stdout':'','error':None}
    started=time.perf_counter()
    phase='CPU limit'
    try:
        resource.setrlimit(resource.RLIMIT_CPU,(3,3))
        # macOS address/data-space limits are unsuitable for this interpreter.
        # The parent watches resident memory instead of capping virtual mappings.
        phase='file size limit'
        resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
        phase='execution'
        tree=validate_code(code)
        def safe_import(name,globals=None,locals=None,fromlist=(),level=0):
            if level or name not in ALLOWED_MODULES: raise ImportError('Import unavailable')
            return importlib.import_module(name)
        safe_builtins={name:getattr(builtins,name) for name in BUILTIN_NAMES}
        safe_builtins['__import__']=safe_import
        environment={'__builtins__':safe_builtins,'__name__':'__main__'}
        with contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):
            exec(compile(tree,'<python_math>','exec'),environment,environment)
        result['ok']=True
    except BaseException as exc:
        result['error']=f'{phase}: {type(exc).__name__}: {str(exc)[:500]}'
    result['stdout']=output.value()
    result['execution_s']=time.perf_counter()-started
    return result


def sandbox_profile():
    runner=Path(__file__).resolve()
    prefix=Path(sys.base_prefix).resolve()
    # Native Python needs macOS dyld/IPC services. Permit those while denying
    # network, writes, private user/temp reads, and execution of other programs.
    executable=Path(sys.executable).resolve()
    return ('(version 1)\n(allow default)\n(deny network*)\n(deny file-write*)\n'
            '(deny file-read* (subpath "/Users") (subpath "/Volumes") '
            '(subpath "/private/tmp") (subpath "/private/var/folders"))\n'
            f'(allow file-read* (subpath {json.dumps(str(prefix))}) (literal {json.dumps(str(runner))}))\n'
            '(deny process-fork)\n(deny process-exec)\n'
            f'(allow process-exec (literal {json.dumps(str(executable))}) '
            f'(subpath {json.dumps(str(prefix))}))\n')


async def execute_python(code):
    started=time.perf_counter()
    try: validate_code(code)
    except (ValueError,SyntaxError) as exc:
        return {'ok':False,'stdout':'','error':f'{type(exc).__name__}: {exc}','wall_s':time.perf_counter()-started}
    sandbox=Path('/usr/bin/sandbox-exec')
    if sys.platform!='darwin' or not sandbox.exists():
        return {'ok':False,'stdout':'','error':'OS sandbox unavailable; execution disabled','wall_s':time.perf_counter()-started}
    command=[str(sandbox),'-p',sandbox_profile(),str(Path(sys.executable).resolve()),'-I','-S',str(Path(__file__).resolve()),'--worker']
    process=await asyncio.create_subprocess_exec(*command,stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,
        env={'PATH':'/usr/bin:/bin','LC_ALL':'C','PYTHONDONTWRITEBYTECODE':'1'})
    memory_exceeded=False
    async def memory_watchdog():
        nonlocal memory_exceeded
        while process.returncode is None:
            probe=await asyncio.create_subprocess_exec('/bin/ps','-o','rss=','-p',str(process.pid),
                stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.DEVNULL)
            data,_=await probe.communicate()
            if data.strip().isdigit() and int(data.strip())>256*1024:
                memory_exceeded=True
                if process.returncode is None: process.kill()
                return
            await asyncio.sleep(.1)
    monitor=asyncio.create_task(memory_watchdog())
    try:
        stdout,stderr=await asyncio.wait_for(process.communicate(json.dumps({'code':code}).encode()),timeout=5)
        if process.returncode:
            result={'ok':False,'stdout':'','error':'Resident memory limit exceeded' if memory_exceeded else f'Worker exited {process.returncode}: {stderr.decode(errors="replace")[:500]}'}
        else:
            result=json.loads(stdout)
    except asyncio.TimeoutError:
        process.kill(); await process.communicate()
        result={'ok':False,'stdout':'','error':'Wall time limit exceeded'}
    except (ValueError,KeyError) as exc:
        result={'ok':False,'stdout':'','error':f'Invalid worker response: {type(exc).__name__}'}
    finally:
        monitor.cancel()
        await asyncio.gather(monitor,return_exceptions=True)
        if process.returncode is None:
            process.kill(); await process.communicate()
    result['wall_s']=time.perf_counter()-started
    return result


if __name__=='__main__' and sys.argv[1:]==['--worker']:
    print(json.dumps(worker(json.load(sys.stdin)['code'])))
