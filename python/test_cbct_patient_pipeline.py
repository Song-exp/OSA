"""모델 추론 없이 환자별 재개·QC·수동 수정·대상 선택을 검증."""
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import cbct_patient_pipeline as pipeline


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.parent=(pipeline.ROOT/'Result_py').resolve()
        self.temp=tempfile.TemporaryDirectory(prefix='osa_cbct_test_',dir=self.parent)
        self.root=Path(self.temp.name)
        self.cb=self.root/'Result_py/CBCT';self.out=self.cb/'patient_pipeline/v2'
        self.patches=[patch.object(pipeline,'ROOT',self.root),patch.object(pipeline,'CB',self.cb),patch.object(pipeline,'OUT',self.out),
                      patch.object(pipeline,'source_signature',return_value={'series_uid':'test','files':2}),
                      patch.object(pipeline,'excel',return_value=pd.DataFrame({'ANB':[5.],'MPFH':[28.],'Nperp_A':[0.],'Nperp_Pog':[-8.]},index=[1]))]
        for p in self.patches:p.start()
        (self.root/'python/osa').mkdir(parents=True)
        (self.root/'python/osa/ceph.py').write_text('# geometry fixture',encoding='utf-8')
        (self.root/'Patient_Info_Data_87명_261019.xlsx').write_bytes(b'excel fingerprint fixture')
        self.points={'N':[0,0,0],'S':[0,55,25],'Ba':[0,90,-30],'RPo':[-55,80,-25],'LPo':[55,80,-25],
          'ROr':[-35,10,-25],'LOr':[35,10,-25],'ANS':[0,-10,-50],'PNS':[0,40,-50],
          'A':[0,2,-60],'B':[0,12,-100],'Pog':[0,8,-115],'Me':[0,12,-122],
          'Gn':[0,10,-121],'RGo':[-45,80,-86],'LGo':[45,80,-86]}
        self.landmark_file=self.cb/'landmarks/P_1_lm_Pred_all.mrk.json'
        self.write_points(self.points)
        self.write_nrrd(self.cb/'nrrd/P_1.nrrd')
        for s in pipeline.STRUCTS:self.write_nrrd(self.cb/f'seg/P_1_seg_{s}.nrrd')
        p={k:np.array(v,float) for k,v in self.points.items()}
        self.ref=self.root/'qc_reference.json'
        pipeline.save(self.ref,{'version':'distance-mad-v1','cutoff':4.,'pairs':[list(x) for x in pipeline.QC_PAIRS],
          'median':[float(np.linalg.norm(p[a]-p[b])) for a,b in pipeline.QC_PAIRS],'scale':[1.]*len(pipeline.QC_PAIRS)})

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        resolved=self.root.resolve()
        assert resolved.parent==self.parent and resolved.name.startswith('osa_cbct_test_')
        self.temp.cleanup()

    def write_points(self,points,system='LPS'):
        pipeline.save(self.landmark_file,{'markups':[{'coordinateSystem':system,'controlPoints':[
          {'label':k,'position':v,'positionStatus':'defined'} for k,v in points.items()]}]})

    def write_nrrd(self,path):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b'NRRD0004\ntype: short\ndimension: 3\nsizes: 3 3 3\nencoding: gzip\n\n'+gzip.compress(np.zeros((3,3,3),dtype='<i2').tobytes()))

    def test_complete_and_resume_without_inference(self):
        with patch.object(pipeline.subprocess,'run') as inference:
            a=pipeline.execute(1,self.ref)
            b=pipeline.execute(1,self.ref)
            inference.assert_not_called()
        self.assertEqual(a,b)
        self.assertEqual(json.loads((self.out/'P_1/status.json').read_text(encoding='utf-8'))['status'],'cached')

    def test_missing_label_not_treated_as_complete(self):
        self.write_points({k:v for k,v in self.points.items() if k!='RPo'})
        with self.assertRaisesRegex(ValueError,'불완전'):pipeline.execute(1,self.ref)
        state=json.loads((self.out/'P_1/status.json').read_text(encoding='utf-8'))
        self.assertEqual(state['status'],'failed')
        self.assertNotIn('segmentation',state['steps'])

    def test_truncated_nrrd_rejected(self):
        p=self.cb/'nrrd/P_1.nrrd';p.write_bytes(p.read_bytes()[:-5])
        with self.assertRaises(ValueError):pipeline.valid_nrrd(p)

    def test_manual_correction_separate_and_recomputes(self):
        bad=dict(self.points);bad['RPo']=[-10,10,-25];self.write_points(bad)
        before=self.landmark_file.read_bytes()
        a=pipeline.execute(1,self.ref)
        self.assertTrue(a['variants']['automatic']['qc_flags'])
        corr=self.root/'P_1.json'
        pipeline.save(corr,{'patient':'P_1','coordinate_system':'LPS','reviewer':'test reviewer',
                           'reason':'synthetic known point','points':{'RPo':self.points['RPo']}})
        b=pipeline.execute(1,self.ref,correction=corr)
        self.assertFalse(b['variants']['reviewed']['qc_flags'])
        self.assertEqual(a['variants']['automatic'],b['variants']['automatic'])
        self.assertEqual(before,self.landmark_file.read_bytes())
        self.assertNotEqual(a['fingerprint'],b['fingerprint'])
        self.assertEqual(len(list((self.out/'P_1/history').glob('*.json'))),1)

    def test_wrong_patient_override_rejected(self):
        corr=self.root/'P_2.json';pipeline.save(corr,{'patient':'P_2','coordinate_system':'LPS','reviewer':'x','reason':'x','points':{}})
        with self.assertRaises(ValueError):pipeline.corrections(self.points,corr,1)

    def test_source_change_does_not_reuse_stale_result(self):
        pipeline.execute(1,self.ref)
        with patch.object(pipeline,'source_signature',return_value={'series_uid':'changed'}):
            with self.assertRaisesRegex(ValueError,'변경 감지'):pipeline.execute(1,self.ref)

    def test_ras_conversion_and_unknown_system(self):
        self.write_points(self.points,'RAS');p,_=pipeline.landmarks(1)
        np.testing.assert_allclose(p['RPo'],np.array(self.points['RPo'])*[-1,-1,1])
        self.write_points(self.points,'unknown')
        with self.assertRaises(ValueError):pipeline.landmarks(1)

    def test_manifest_exclusion_and_duplicate_ids(self):
        f=self.root/'cohort.csv';pd.DataFrame({'Patient':['P_1','P_2'],'Include':['true','false'],'Reason':['confirmed','inquiry']}).to_csv(f,index=False)
        script=Path(pipeline.__file__).resolve()
        r=subprocess.run([sys.executable,'-X','utf8',str(script),'--manifest',str(f)],capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(json.loads(r.stdout)['patients'],['P_1'])
        r=subprocess.run([sys.executable,'-X','utf8',str(script),'--ids','1','P_1'],capture_output=True,text=True,encoding='utf-8')
        self.assertNotEqual(r.returncode,0)

    def test_invalid_reference_not_silently_zipped(self):
        profile=json.loads(self.ref.read_text(encoding='utf-8'));profile['median']=[]
        with self.assertRaises(ValueError):pipeline.qc({k:np.asarray(v) for k,v in self.points.items()},profile)


if __name__=='__main__':unittest.main()
