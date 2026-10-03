"""Regression checks for native tool-call history, strict arguments, accounting, and macOS worker restrictions.

Run this test module after changes to the corresponding library or runner:
    python -m unittest test.test_python_tool_pilot
Tests make no inference calls. Worker checks require macOS sandbox-exec.
"""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

from src.python_math_tool import execute_python, sandbox_profile, validate_code
from src.python_tool_protocol import assistant_history, parse_tool_call, totals


class ProtocolTests(unittest.TestCase):
    def test_preserve_call_ids_and_reasoning(self):
        message={'role':'assistant','content':None,'tool_calls':[{'id':'call-a'}],
                 'reasoning_details':[{'type':'reasoning.text','text':'working'}],
                 'reasoning':'working','provider_extra':'omit'}
        original=deepcopy(message)
        history=assistant_history(message)
        self.assertEqual(history['tool_calls'][0]['id'],'call-a')
        self.assertEqual(history['reasoning_details'],message['reasoning_details'])
        history['tool_calls'][0]['id']='changed'
        self.assertEqual(message,original)
        self.assertNotIn('provider_extra',history)

    def test_strict_arguments(self):
        call={'type':'function','id':'call-a','function':{'name':'python_math','arguments':json.dumps({'code':'print(4)'})}}
        self.assertEqual(parse_tool_call(call),'print(4)')
        call['function']['arguments']=json.dumps({'code':'print(4)','other':True})
        with self.assertRaises(ValueError): parse_tool_call(call)

    def test_accounting_counts_every_round_even_after_error(self):
        rounds=[{'response':{'usage':{'prompt_tokens':10,'completion_tokens':20,'total_tokens':30,'cost':.1}}},
                {'error':'invalid choice','response':{'usage':{'prompt_tokens':30,'completion_tokens':40,'total_tokens':70,'cost':.2}}}]
        result=totals(rounds)
        self.assertEqual(result['completion_tokens'],60)
        self.assertEqual(result['total_tokens'],100)
        self.assertAlmostEqual(result['reported_cost'],.3)
        result=totals(rounds+[{'error':'network error'}])
        self.assertIsNone(result['completion_tokens'])
        self.assertEqual(result['rounds_with_missing_usage'],1)

    def test_reject_unsafe_language(self):
        for code in ['import os','open(".env")','print((1).__class__)','eval("4")','print(__builtins__)']:
            with self.subTest(code=code),self.assertRaises(ValueError): validate_code(code)


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_os_denies_repository_reads_and_network(self):
        # Trusted probe bypasses the language filter to check the OS boundary.
        # Read a public source file, never credentials, and print no contents.
        code='''import socket
try:
    open(%r).close()
    print("READ_ALLOWED")
except PermissionError:
    print("READ_DENIED")
try:
    s=socket.socket()
    s.settimeout(1)
    s.connect(("127.0.0.1", 9))
    print("NETWORK_ALLOWED")
except PermissionError:
    print("NETWORK_DENIED")
''' % str(Path(__file__).resolve())
        process=await asyncio.create_subprocess_exec('/usr/bin/sandbox-exec','-p',sandbox_profile(),
            str(Path(sys.executable).resolve()),'-I','-S','-c',code,
            stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,
            env={'PATH':'/usr/bin:/bin','LC_ALL':'C'})
        stdout,stderr=await asyncio.wait_for(process.communicate(),5)
        self.assertEqual(process.returncode,0,stderr.decode())
        self.assertEqual(stdout.decode().splitlines(),['READ_DENIED','NETWORK_DENIED'])

    async def test_math_and_error_feedback(self):
        result=await execute_python('from fractions import Fraction\nprint(Fraction(1,3)+Fraction(1,6))')
        self.assertTrue(result['ok'],result)
        self.assertEqual(result['stdout'].strip(),'1/2')
        result=await execute_python('print(1/0)')
        self.assertFalse(result['ok'])
        self.assertIn('ZeroDivisionError',result['error'])

    async def test_ordinary_underscore_loop_variable(self):
        result=await execute_python('rows=[[0]*3 for _ in range(9)]\nprint(len(rows))')
        self.assertTrue(result['ok'],result)
        self.assertEqual(result['stdout'].strip(),'9')

    async def test_cpu_limit(self):
        result=await execute_python('while True:\n    pass')
        self.assertFalse(result['ok'])
        self.assertLess(result['wall_s'],6)

    async def test_output_limit(self):
        result=await execute_python('print("x"*7000)')
        self.assertFalse(result['ok'])
        self.assertLessEqual(len(result['stdout']),6000)


if __name__=='__main__': unittest.main()
