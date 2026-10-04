import json
from pathlib import Path
import tempfile
import unittest
from runner_final.integrity_v2_3 import verify_core
from src.common import ROOT
from scripts.profile_v2_3_benchmarks import analyze


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.folder=Path(self.tmp.name)
        self.write('config.json',{'runner_id':'runner_final_core_v2_3','core_manifest_sha256':verify_core(ROOT),
            'git_dirty':False,'benchmark':True,'question_indices':[1],'max_context_tokens':65536,
            'max_concurrent_requests':1,'policy':{'slots':1},'target_correct':1,'served_prompt_tokens':{'1':3}})
        self.write('summary.json',{'status':'completed','solved':1,'time_to_target_s':.1,'target_reached':True,
                                 'official_latency_s':.2,'initialization_and_attempt_latency_s':1})
        self.write('questions.json',[{'problem_idx':1,'problem':'Compute.'}])
        self.write('allocation.json',{'max_concurrent_requests':1,'peak_active_requests':1,
            'questions':{'1':{'used':1}},'admissions':[{'kind':'fresh','active_requests':1}]})
        self.line('solved.jsonl',{'problem_idx':1,'first_solved_elapsed_s':.1})
        self.line('trace/01/verification.jsonl',{'result':{'verdict':True}})
        self.line('trace/01/candidate_validation.jsonl',{'outcome':'enqueued','valid':True,'validation_wall_s':.001})
        self.write('trace/01/question.json',{'rollouts':[{'rollout':1}]})
        self.write('trace/01/rollout-01/request.json',{'messages':[],'max_tokens':65533})
        self.write('trace/01/rollout-01/tokens.json',{'prompt_token_ids':[1,2,3],'output_token_ids':[10],'complete':True})
        self.write('trace/01/rollout-01/telemetry.json',{'rollout':1,'status':'completed','generated_token_ids_count':1})

    def tearDown(self):self.tmp.cleanup()

    def write(self,relative,value):
        path=self.folder/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

    def line(self,relative,value):
        self.write(relative,value)
        path=self.folder/relative;path.write_text(path.read_text()+'\n')

    def test_valid_evidence_and_context_budget(self):
        result=analyze(self.folder)
        self.assertEqual(result['initial_prompt_matches'],1)
        self.assertEqual(result['exact_complete'],1)
        self.write('trace/01/rollout-01/request.json',{'messages':[],'max_tokens':65536})
        with self.assertRaisesRegex(ValueError,'total served context'):analyze(self.folder)

    def test_answer_leak_or_target_time_disagreement_rejected(self):
        self.write('questions.json',[{'problem_idx':1,'problem':'Compute.','answer':'PRIVATE'}])
        with self.assertRaisesRegex(ValueError,'gold-free'):analyze(self.folder)
        self.write('questions.json',[{'problem_idx':1,'problem':'Compute.'}])
        self.line('solved.jsonl',{'problem_idx':1,'first_solved_elapsed_s':.15})
        with self.assertRaisesRegex(ValueError,'Target time'):analyze(self.folder)

    def test_child_prefix_must_match_both_parent_and_served_ids(self):
        self.write('trace/01/rollout-02/request.json',{'prompt':[1,2,3,10,90],'max_tokens':65531})
        self.write('trace/01/rollout-02/tokens.json',{'prompt_token_ids':[1,2,3,10,90],'output_token_ids':[11],'complete':True})
        self.write('trace/01/rollout-02/telemetry.json',{'rollout':2,'status':'completed','generated_token_ids_count':1,'continuation_of_rollout':1})
        result=analyze(self.folder)
        self.assertEqual(result['continuation_prefix_matches'],1)
        self.write('trace/01/rollout-02/tokens.json',{'prompt_token_ids':[1,2,3,99,90],'output_token_ids':[11],'complete':True})
        with self.assertRaisesRegex(ValueError,'Served continuation'):analyze(self.folder)
