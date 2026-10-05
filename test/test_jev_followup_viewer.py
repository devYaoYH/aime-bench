"""Verify separate historical/follow-up APIs without changing the original cohort."""
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from src import viewer_server


class JevFollowupViewerTests(unittest.TestCase):
    def test_historical_and_followup_records_have_distinct_safe_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); run=root/'run'
            def save(name,value):
                p=run/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
            save('config.json',{})
            save('summary.json',{'first_inference_request_at_utc':'2026-09-30T22:52:12+00:00','results':[{'problem_idx':1}]})
            save('questions/01.json',{'response':{'choices':[{'message':{'content':'','reasoning':'trace'}}]}})
            historical={'rows':[{'problem_idx':1,'sample_number':2,'probability':.8}]}
            followup={'rows':[{'problem_idx':1,'sample_number':1,'probability':.6,'finish_reason':'length'}]}
            save('jev_calibration/summary.json',historical)
            save('jev_calibration_followup/summary.json',followup)
            save('jev_calibration/01-02.json',{'probability':.8})
            save('jev_calibration_followup/01-01.json',{'probability':.6,'post_cutoff':True})
            with patch.object(viewer_server,'RUNS',root),patch.object(viewer_server.Handler,'log_message'):
                result=viewer_server.build_overview('run')
                self.assertEqual(result['jev_calibration'],historical)
                self.assertEqual(result['jev_calibration_followup'],followup)
                server=viewer_server.ThreadingHTTPServer(('127.0.0.1',0),viewer_server.Handler)
                thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
                try:
                    conn=http.client.HTTPConnection('127.0.0.1',server.server_port)
                    for path,code in [('/api/runs/run/jev-calibration/1/2',200),
                                      ('/api/runs/run/jev-calibration-followup/1/1',200),
                                      ('/api/runs/run/jev-calibration-followup/1/9',404),
                                      ('/api/runs/../jev-calibration-followup/1/1',404)]:
                        conn.request('GET',path);response=conn.getresponse();body=response.read()
                        self.assertEqual(response.status,code,path)
                        if code==200:self.assertIn(b'probability',body)
                    conn.close()
                finally:
                    server.shutdown();server.server_close();thread.join(timeout=5)
