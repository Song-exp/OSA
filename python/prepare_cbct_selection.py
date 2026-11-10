"""사용자가 지정한 상세 문의 32명만 제외한 임시 CBCT 처리 명단을 고정한다."""
from datetime import datetime
import json
import re
import pandas as pd
from cbct_patient_pipeline import ROOT, CB, OUT, landmarks, valid_nrrd, STRUCTS, digest, save


def main():
    source=ROOT/'Result_py/patient_matching/환자별_데이터현황_및_문의사항.md'
    text=source.read_text(encoding='utf-8')
    excluded=re.findall(r'^### (P_\d+)\s*$',text,re.M)
    if len(excluded)!=32 or len(set(excluded))!=32:raise ValueError('상세 문의 32명 정의 변경: 목록 검토 후 새 선택 버전 필요')
    patients=pd.read_csv(ROOT/'Result_py/patient_matching/patient_matching_with_raw.csv').환자.tolist()
    if len(patients)!=87 or not set(excluded)<=set(patients):raise ValueError('등록 환자와 문의 목록 불일치')
    output=OUT/'cohort.csv'
    rows=[{'Patient':p,'Include':p not in excluded,
           'Reason':'사용자 지정: 상세 문의 대상이므로 임시 제외' if p in excluded else '상세 문의 32명 제외 후 임시 처리 후보; 임상·변수별 결측·QC 확정 전'} for p in patients]
    cohort=pd.DataFrame(rows)
    if output.exists():
        previous=pd.read_csv(output).astype(str)
        if not previous.equals(cohort.astype(str)):raise ValueError('기존 cohort.csv 내용이 다릅니다. 사용자 변경 명단을 자동 덮어쓰지 않습니다.')
    else:
        output.parent.mkdir(parents=True,exist_ok=True)
        cohort.to_csv(output,index=False,encoding='utf-8-sig')
    included=cohort.loc[cohort.Include,'Patient'].tolist()
    assert len(included)==55
    checks=[]
    for p in included:
        pid=int(p[2:]);issues=[]
        try:valid_nrrd(CB/f'nrrd/{p}.nrrd')
        except Exception as e:issues.append(f'nrrd: {type(e).__name__}: {e}')
        try:landmarks(pid)
        except Exception as e:issues.append(f'landmarks: {type(e).__name__}: {e}')
        for s in STRUCTS:
            try:valid_nrrd(CB/f'seg/{p}_seg_{s}.nrrd')
            except Exception as e:issues.append(f'seg-{s}: {type(e).__name__}: {e}')
        checks.append({'Patient':p,'ArtifactsReady':not issues,'Issues':' | '.join(issues)})
    preflight=pd.DataFrame(checks)
    preflight.to_csv(OUT/'cohort_preflight.csv',index=False,encoding='utf-8-sig')
    state={'version':'exclude-detailed-inquiries-32-v1','created':datetime.now().isoformat(),
           'source':str(source.relative_to(ROOT)),'source_sha256':digest(source),'manifest_sha256':digest(output),
           'registered':87,'excluded_count':32,'included_count':55,'excluded':excluded,'included':included,
           'selection_rule':'Only exact ### P_n detailed inquiry headings. M01–M10 cohort/variable lists do not expand this exclusion.',
           'note':'Temporary processing eligibility, not identity judgment or final analytic cohort. Original/review data retained.',
           'preflight_ready':int(preflight.ArtifactsReady.sum()),
           'preflight_pending':preflight.loc[~preflight.ArtifactsReady,'Patient'].tolist()}
    save(OUT/'cohort_selection.json',state)
    print(json.dumps({k:v for k,v in state.items() if k not in ['included','excluded']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
