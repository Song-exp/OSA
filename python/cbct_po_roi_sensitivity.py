"""Shift candidate ROI centers without changing landmarks or choosing by Excel.

Seven locations: center and +/-2 mm along each Po-free anatomical frame axis.
This is a sensitivity grid, not an estimated coordinate uncertainty interval.
"""
import argparse
import json

import numpy as np
import pandas as pd

from cbct_patient_pipeline import CB, OUT, ROOT, digest, landmarks, patient_id, save
from cbct_po_pixel_audit import AUDIT, CASES, METRICS, frame, included, model_clip_bounds, native_stats
from check_landmarks import read_nrrd


def run(pid):
    if pid not in included() or pid not in CASES:
        raise ValueError('An included case with a recorded image candidate is required')
    side=CASES[pid]
    path=OUT/f'P_{pid}/review/{side}_evidence.json'
    evidence=json.loads(path.read_text(encoding='utf-8'))
    scan=CB/f'nrrd/P_{pid}.nrrd'
    if digest(scan)!=evidence['image_sha256']:raise ValueError('Candidate image changed')
    points,files=landmarks(pid)
    if {f.name:digest(f) for f in files}!=evidence['landmarks_sha256']:
        raise ValueError('Candidate source landmarks changed')
    center=np.array(evidence['candidate']['position_lps_mm'])
    basis=frame(points,side)
    volume,directions,origin=read_nrrd(scan)
    clip=model_clip_bounds(volume)
    shifts=[np.zeros(3)]+[np.eye(3)[i]*delta for i in range(3) for delta in [-2.,2.]]
    rows=[]
    for offset in shifts:
        shifted=center+basis@offset
        metrics,_=native_stats(volume,directions,origin,shifted,clip)
        rows.append({'Patient':f'P_{pid}','Side':side,'outward_mm':offset[0],
                     'anterior_mm':offset[1],'superior_mm':offset[2],**metrics})
    folder=AUDIT/f'patients/P_{pid}'
    folder.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(folder/'roi_center_sensitivity.csv',index=False,encoding='utf-8-sig')
    result={'patient':f'P_{pid}','side':side,'evidence_sha256':digest(path),
            'code_sha256':digest(__file__),'pixel_code_sha256':digest(ROOT/'python/cbct_po_pixel_audit.py'),
            'method':'center and +/-2 mm along each Po-free frame axis; 7 positions, not a confidence interval',
            'ranges':{key:[min(r[key] for r in rows),max(r[key] for r in rows)] for key in METRICS}}
    save(folder/'roi_center_sensitivity.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--patient',type=patient_id,required=True)
    run(ap.parse_args().patient)
