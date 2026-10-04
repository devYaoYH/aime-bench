"""Check streamed-ID counting, candidate event matching and continuation replay."""
import json
from pathlib import Path
import tempfile
import unittest
from scripts.analyze_core_v1_answer_tokens import replay, ts
from runner_final.core_v1._streaming import CandidateDetector


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.folder=self.root/'rollout-01';self.folder.mkdir()

    def prepare(self,pieces,*,prefix='',parent=None,endpoint='/v1/chat/completions',winner_text=None):
        if parent:
            p=self.root/f'rollout-{parent:02d}';p.mkdir(exist_ok=True)
            (p/'tokens.json').write_text(json.dumps({'visible_text':prefix}))
        text=prefix+''.join(piece for piece,ids in pieces)
        (self.folder/'tokens.json').write_text(json.dumps({'visible_text':text,'output_token_ids':[i for piece,ids in pieces for i in ids]}))
        (self.folder/'telemetry.json').write_text(json.dumps({'continuation_of_rollout':parent,'endpoint':endpoint}))
        lines=[]
        for n,(piece,ids) in enumerate(pieces):
            choice={'token_ids':ids,'delta':{'content':piece},'index':0}
            if endpoint=='/v1/completions':choice['text']=piece
            lines.append(json.dumps({'elapsed_s':float(n),'timestamp_utc':f'2026-10-04T00:00:0{n}.000+00:00',
                                     'data':json.dumps({'choices':[choice]})}))
        stream=self.root/'stream.jsonl';stream.write_text('\n'.join(lines)+'\n')
        d=CandidateDetector();events=d.feed('content',winner_text or text)
        winner={**events[-1],'observed_at_utc':f'2026-10-04T00:00:0{len(pieces)-1}.000+00:00'}
        return stream,winner

    def test_counts_trigger_chunk_and_separates_verdict_time(self):
        stream,winner=self.prepare([('reasoning ',[1,2]),('\\boxed{12}',[3,4,5]),(' checking',[6,7])],winner_text='reasoning \\boxed{12}')
        winner['observed_at_utc']='2026-10-04T00:00:01.000+00:00'
        row=replay(self.folder,stream,winner,cutoff=ts('2026-10-04T00:00:02.000+00:00'))
        self.assertEqual(row['tokens_at_candidate'],5)
        self.assertEqual(row['tokens_at_cutoff'],7)

    def test_continuation_preserves_detector_character_offsets(self):
        self.folder=self.root/'rollout-02';self.folder.mkdir()
        stream,winner=self.prepare([('\\boxed{12}',[8,9])],prefix='previous reasoning ',parent=1,endpoint='/v1/completions')
        self.assertEqual(replay(self.folder,stream,winner)['tokens_at_candidate'],2)

    def test_rejects_saved_id_drift(self):
        stream,winner=self.prepare([('\\boxed{12}',[8,9])])
        p=self.folder/'tokens.json';saved=json.loads(p.read_text());saved['output_token_ids'].append(10);p.write_text(json.dumps(saved))
        with self.assertRaisesRegex(AssertionError,'Output IDs differ'):
            replay(self.folder,stream,winner)
