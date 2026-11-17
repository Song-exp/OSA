"""Step 2~4 신규 인자의 상세 설명: (변수 뜻, 만든 과정). 뜻 = 해부학적으로 무엇을 재는지·단위·클 때 의미. 과정 = 원자료 → 도구 → 좌표계 → 계산.
계산 근거 코드: cbct_pmca.py, cbct_pav.py, shape_extra_vectors.py, shape_fine_vectors.py, shape_tongue_sinus.py, build_features.py, shape_local_roi.py,
osa_residual_shape.py, osa_joint_axes.py, osa_composite.py, osa_vertical_index.py, osa_experiments.size_axis.
"""
CT = "CBCT 원영상"
MID = "정중 좌표계(원점 PNS, 축 = 좌·앞·위, 앞 축 = ANS–PNS 방향, 귓구멍 점 미사용)"
ALI = f"{CT} → ALI-CBCT 자동 기준점 → {MID}"
AIR = f"{CT} → 공기 밝기 규칙(−1200~−400 HU, 정중선과 이어진 덩어리)으로 인두 기도 분할 → {MID}"
TS_HG = f"{CT} → TotalSegmentator(head_glands_cavities) 자동 분할 → {MID}"
TS_HM = f"{CT} → TotalSegmentator(head_muscles) 자동 분할(0.5 mL 미만 조각 제거)"
TS_CF = f"{CT} → TotalSegmentator(craniofacial_structures) 자동 분할"
TEETH = f"{CT} → 공개 nnU-Net 치아 분할(UNIVERSALLAB, 범용 치아 번호) → 치아별 중심점 → {MID}"
SSM = f"{CT} → AMASSS 자동 분할(하악·상악·두개저) → 표면 2048개 대응점(통계 형상 모델)"
EBH = "Eb(후두개곡 최하점)·설골은 정중 단면에서 밝기 규칙으로 자동 검출"

D = {
 # ---------------- Step 2 기도
 "a_PAV": ("인두 기도 부피(mL). 위로 S–PNS면, 아래로 Eb를 지나는 수평면 사이의 인두 공기 부피. 클수록 인두가 넓음",
           f"{AIR} → 위 경계 S–PNS 평면, 아래 경계 Eb 높이 수평면, 앞 경계 PNS 관상면 → 경계 안 공기 복셀 수 × 복셀 부피. {EBH}"),
 "a_PAV_h": ("인두 평균 단면적(mm²) = 인두 부피 ÷ 기둥 높이. 길이 효과를 뺀 인두의 평균 굵기",
             "a_PAV(mm³ 환산) ÷ H_col(PNS에서 Eb까지 수직 거리)"),
 "H_col": ("인두 기둥 높이(mm): PNS에서 Eb까지의 수직 거리. 클수록 인두가 위아래로 김", f"{ALI} → Eb 높이(밝기 규칙 검출)와 PNS(원점) 높이 차"),
 "AW_min": ("인두 기도 최소 단면적(mm²): 중심선에 수직인 단면 20구간 프로파일의 최솟값", f"{AIR} → 기도 중심선(0.4 mm 간격) 수직 단면 넓이 → Eb(0)~PNS(1) 구간을 20칸으로 나눈 평균 단면적 → 최솟값"),
 "AW_length": ("인두 기도 길이(mm): Eb에서 PNS까지 수직 거리", f"{ALI} → −(Eb의 위아래 좌표)"),
 "AW_min_pos": ("기도가 가장 좁은 위치(0 = Eb 쪽 아래, 1 = PNS 쪽 위). 클수록 좁은 곳이 위(구개 뒤)에 있음", "AW_min을 구한 20구간 프로파일에서 최소 구간의 상대 위치"),
 "HD_mm": ("PMCA 단면의 수력학적 직경(mm) = 4 × 넓이 ÷ 둘레. 같은 넓이라도 납작하면 작아짐(공기 흐름 저항 관련)", f"{AIR} → PMCA(최소 단면) 위치의 중심선 수직 단면 → 넓이 A, Crofton 둘레 P → 4A/P"),
 "nasopharynx_min_area": ("비인두 최소 단면적(mm²): 연구개 위·후비공 뒤 인두 공기의 가장 좁은 가로 단면 넓이", f"{TS_HG} 비인두 라벨 → 1 mm 두께 가로(축상) 단면마다 복셀 수 × 복셀 부피 → 위아래 끝 2 mm 제외 → 최솟값"),
 "nasopharynx_mean_area": ("비인두 평균 단면적(mm²): 비인두 가로 단면 넓이의 평균", f"{TS_HG} 비인두 라벨 → 1 mm 가로 단면 넓이 → 위아래 끝 2 mm 제외 → 평균"),
 "oropharynx_min_area": ("구인두 최소 단면적(mm²): 연구개 아래~후두개 위 인두(혀 뒤) 공기의 가장 좁은 가로 단면 넓이", f"{TS_HG} 구인두 라벨 → 1 mm 가로 단면 넓이 → 위아래 끝 2 mm 제외 → 최솟값"),
 "oropharynx_mean_area": ("구인두 평균 단면적(mm²)", f"{TS_HG} 구인두 라벨 → 1 mm 가로 단면 넓이 → 위아래 끝 2 mm 제외 → 평균"),
 "hypopharynx_min_area": ("하인두 최소 단면적(mm²): 후두개 높이 아래 인두 공기의 가장 좁은 가로 단면 넓이", f"{TS_HG} 하인두 라벨 → 1 mm 가로 단면 넓이 → 위아래 끝 2 mm 제외 → 최솟값"),
 "hypopharynx_mean_area": ("하인두 평균 단면적(mm²)", f"{TS_HG} 하인두 라벨 → 1 mm 가로 단면 넓이 → 위아래 끝 2 mm 제외 → 평균"),
 "oro_naso_min_ratio": ("구인두 최소 단면적 ÷ 비인두 최소 단면적. 작을수록 혀 뒤가 구개 뒤보다 상대적으로 좁음", "oropharynx_min_area ÷ nasopharynx_min_area"),
 "m_NAV": ("비강 기도 부피(mL, 자동). Roh의 NAV 정의(Na–Ba·ANS–PNS·Se–PNS·Rh–ANS 경계) 안의 비강 공기", f"{CT} → ALI 기준점으로 경계 평면 설정 → 공기 밝기 규칙 + TotalSegmentator 비강 라벨 안 공기 → 부피(엑셀 방식에 맞춘 환산 포함)"),
 "soft_palate_length": ("연구개 길이(mm): 정중 단면에서 연구개의 장축 방향 길이. 길수록 연구개가 기도 쪽으로 늘어짐",
                        f"{TS_HG} 연구개 라벨 → 정중면 ±2 mm 점 → 주성분으로 장축 결정 → 장축 방향 최대–최소 거리"),
 "soft_palate_thickness": ("연구개 최대 두께(mm): 장축에 수직인 방향의 최대 폭", f"{TS_HG} 연구개 라벨 → 정중면 ±2 mm 점 → 장축을 2 mm 구간으로 나눠 구간별 수직 폭 → 최댓값"),
 "soft_palate_angle": ("연구개 기울기(°): 연구개 장축이 앞 축(구개 방향)과 이루는 각. 클수록 아래로 처짐", f"{TS_HG} 연구개 라벨 → 정중면 ±2 mm 점 → 장축(뒤쪽을 향하게) → 앞뒤·위아래 성분의 각"),
 "soft_palate_PNS_gap": ("PNS와 연구개 사이 가장 가까운 거리(mm)", f"{TS_HG} 연구개 라벨 → 원점(PNS)에서 라벨 점까지 최소 거리"),
 "AWshape_PC1": ("기도 단면 프로파일 모양의 첫째 주성분: 20구간 단면적이 함께 커지고 작아지는 정도(전체 굵기)", "AW_p00~AW_p19(Eb→PNS 20구간 평균 단면적) → 85명에서 구간별 표준화 → PCA 첫째 축 점수"),
 "AWshape_PC2": ("기도 단면 프로파일 모양의 둘째 주성분: 위쪽과 아래쪽 굵기의 대비", "같은 프로파일 PCA 둘째 축 점수"),
 "AWshape_PC3": ("기도 단면 프로파일 모양의 셋째 주성분", "같은 프로파일 PCA 셋째 축 점수"),
 # ---------------- Step 3 골격
 "Pog_Ba_ant": ("턱끝(Pog)과 대후두공 앞점(Ba)의 앞뒤 거리(mm). 작을수록 하악이 두개저에 비해 뒤에 있음(하악 후퇴)", f"{ALI} → Pog 앞뒤 좌표 − Ba 앞뒤 좌표"),
 "SNA": ("SNA(°): N에서 본 S 방향과 A점 방향 사이 각. 클수록 상악이 앞에 있음", f"{ALI} → N–S, N–A 벡터를 정중 시상면에 투영 → 사이 각"),
 "SNB": ("SNB(°): N에서 본 S 방향과 B점 방향 사이 각. 작을수록 하악이 뒤에 있음", f"{ALI} → N–S, N–B 벡터를 정중 시상면에 투영 → 사이 각"),
 "Pog_PNS_ant": ("PNS 기준 턱끝(Pog)의 앞뒤 위치(mm). 작을수록 하악이 상악 뒤쪽 기준으로 후퇴", f"{ALI} → Pog의 앞 좌표(원점이 PNS)"),
 "PNS_B_ant": ("PNS 기준 B점의 앞뒤 위치(mm). 작을수록 아래 치조부가 뒤에 있음", f"{ALI} → B점의 앞 좌표"),
 "Overjet_mm": ("수평 피개(mm): 위 중절치 절단 끝이 아래 중절치 절단 끝보다 앞에 있는 거리. 클수록 하악 후퇴형 교합",
                f"{TEETH} → 위 중절치(8·9) 최저점, 아래 중절치(24·25) 최고점 → 두 끝의 앞뒤 좌표 차(위가 앞이면 +)"),
 "Co_Gn": ("유효 하악 길이(mm): 과두 최상점(좌우 평균)–Gn(Pog·Me 중점) 거리. 작을수록 하악이 짧음", f"{ALI} + AMASSS 하악 분할에서 과두 최상점 → 3D 거리"),
 "Go_Me": ("하악체 길이(mm): 좌우 하악각점(Go) 중점–Me 거리", f"{ALI} → 3D 거리"),
 "body_len": ("하악체 길이(mm): 좌우 Go 중점–Pog 거리", f"{ALI} → 3D 거리"),
 "body_len_res": ("크기·성별과 무관한 하악체 길이(mm): 머리 크기와 성별로 예상되는 길이보다 얼마나 긴지", "body_len ~ 구조 크기 축 + 성별 회귀(85명)의 잔차"),
 "McN_diff": ("McNamara 차(mm) = 유효 하악 길이 − 유효 중안면 길이(Co–A). 작을수록 하악이 상악보다 상대적으로 짧음", "Co_Gn − Co_A"),
 "F1_PC1": ("공동 축 첫째: 하악이 상악·두개저에 비해 뒤·위로 놓이는 배치(하악 후퇴 축)", f"{SSM} → 하악·상악·두개저 대응점을 정중 좌표계 그대로 한 벡터로 이음 → PCA(85명) 첫째 축 점수"),
 "m_NperpPog": ("Nperp-Pog(mm, 자동): N을 지나 FH에 수직인 선에서 Pog까지의 앞뒤 거리", f"{CT} → ALI 기준점 → 4점 FH 평면 → N 수직선 기준 Pog 앞뒤 거리(엑셀 방식 환산 포함)"),
 "m_NperpA": ("Nperp-A(mm, 자동): N 수직선에서 A점까지의 앞뒤 거리", f"{CT} → ALI 기준점 → 4점 FH 평면 → N 수직선 기준 A 앞뒤 거리(엑셀 방식 환산 포함)"),
 "PP_MP": ("구개평면–하악평면 각(°): ANS–PNS 선과 Go 중점–Me 선 사이 각. 클수록 아래 얼굴이 벌어진 고각형", f"{ALI} → 두 선을 정중면에 투영 → 사이 각"),
 "SN_MP": ("S–N선–하악평면 각(°). 클수록 하악평면이 가파름(고각형)", f"{ALI} → S–N, Go 중점–Me 선을 정중면에 투영 → 사이 각"),
 "LAFH_ratio": ("아래 앞얼굴 비율 = ANS–Me ÷ N–Me. 클수록 얼굴 아래 1/2이 김", f"{ALI} → ANS–Me 거리 ÷ N–Me 거리"),
 "VERT_Z": ("수직 배치 지표: 하악이 상악보다 아래로 놓이고 아래 얼굴이 긴 정도(세 방법의 공통 성분)",
            "z(하악 잔여 형상 넷째 성분 MAND_rPC4) + z(아래 앞얼굴 비율) + z(공동 축 넷째 F1_PC4)의 평균. 85명 기준 표준화, 계산 방식 고정"),
 "gonial": ("하악각(°): 과두–Go–Me가 이루는 각. 클수록 하악이 벌어진 형태", f"{ALI} + AMASSS 과두 → 3D 각"),
 "Jarabak": ("Jarabak 비 = S–Go ÷ N–Me(뒤/앞 얼굴 높이 비). 작을수록 시계방향 성장(고각형)", f"{ALI} → S–Go 거리 ÷ N–Me 거리"),
 "ANS_Me": ("아래 앞얼굴 높이(mm): ANS–Me 거리", f"{ALI} → 3D 거리"),
 "N_Me": ("앞얼굴 전체 높이(mm): N–Me 거리", f"{ALI} → 3D 거리"),
 "S_Go": ("뒤얼굴 높이(mm): S–Go 중점 정중면 거리", f"{ALI} → 정중면 거리"),
 "Co_Go": ("하악지 높이(mm): 과두–Go 중점 거리", f"{ALI} + AMASSS 과두 → 3D 거리"),
 "ramus_body": ("하악지/하악체 비 = Co–Go ÷ Go–Me", "Co_Go ÷ Go_Me"),
 "Yaxis": ("Y축 각(°): S–N과 S–Gn 사이 각. 클수록 하악이 아래·뒤로 자란 방향", f"{ALI} → 정중면 투영 각"),
 "FacialAxis_proxy": ("얼굴 축 근사(°): Ba–N선과 PNS–Gn선 사이 각. 작을수록 하악이 뒤로 회전", f"{ALI} → 정중면 투영 각"),
 "F1_PC4": ("공동 축 넷째: 하악이 상악보다 아래로 놓이는 수직 배치", f"{SSM} → 세 구조 대응점 연결 PCA(85명) 넷째 축 점수"),
 "overbite_mm": ("수직 피개(mm): 위 중절치 끝이 아래 중절치 끝을 위아래로 덮는 정도. 음수면 개방교합", f"{TEETH} → 위 중절치 최저점과 아래 중절치 최고점의 높이 차(겹치면 +)"),
 "upper_width_M1": ("위 제1대구치 사이 폭(mm): 좌우 위 첫째 큰어금니 중심 사이 거리. 상악 치열궁의 가로 폭", f"{TEETH} → 위 좌우 제1대구치 중심점 3D 거리"),
 "lower_width_M1": ("아래 제1대구치 사이 폭(mm)", f"{TEETH} → 아래 좌우 제1대구치 중심점 3D 거리"),
 "InterGonial": ("하악 폭(mm): 좌우 하악각점 사이 거리", f"{ALI} → RGo–LGo 3D 거리"),
 "Upper_Taper": ("위 치열궁 비율 = 송곳니 폭 ÷ 제1대구치 폭. 작을수록 앞쪽이 좁은 V자형 치열궁", f"{TEETH} → 위 좌우 송곳니 중심 거리 ÷ 위 좌우 제1대구치 중심 거리"),
 "upper_depth": ("위 치열궁 깊이(mm): 중절치 중심에서 제1대구치 중심까지 앞뒤 거리", f"{TEETH} → 위 중절치 평균 앞 좌표 − 위 제1대구치 평균 앞 좌표"),
 "m_PHI": ("구개 높이 지수(자동): 구개 천장 높이 ÷ 상악 폭. 클수록 구개가 깊고 좁음", f"{CT} → ALI 치아 기준점 → 제1대구치 저항중심 높이 관상면 → Roh 정의(엑셀 방식 환산 포함)"),
 "m_NCWR": ("비강 폭 비(자동) = 비강 폭 ÷ 상악 기저폭. 작을수록 비강이 좁음", f"{CT} → 관상면 비강 최대폭과 상악 기저폭 → 비(엑셀 방식 환산 포함)"),
 "SIZE_PC1": ("구조 크기 축: 하악·상악·두개저 12부위 크기의 공통 성분. 클수록 머리·턱 구조가 큼", f"{SSM} → 부위 12곳의 중심 크기(roi_*_size) → 85명 표준화 → PCA 첫째 축(분산 69%), 계산 방식 고정"),
 "SIZE_MAND": ("하악 4부위 크기의 표준화 평균", "roi_MAND_*_size 4개 z 점수 평균"), "SIZE_MAX": ("상악 4부위 크기의 표준화 평균", "roi_MAX_*_size 4개 z 점수 평균"),
 "SIZE_CB": ("두개저 4부위 크기의 표준화 평균", "roi_CB_*_size 4개 z 점수 평균"),
 "S_N": ("앞 두개저 길이(mm): S–N 거리", f"{ALI} → 3D 거리"), "S_Ba": ("뒤 두개저 길이(mm): S–Ba 거리", f"{ALI} → 3D 거리"),
 "NSBa": ("두개저 각(°): N–S–Ba. 클수록 두개저가 펴져 하악이 뒤에 위치하기 쉬움", f"{ALI} → 정중면 투영 각"),
 "ANS_PNS": ("구개 길이(mm): ANS–PNS 거리", f"{ALI} → 3D 거리"),
 "PNS_Ba_depth": ("PNS 기준 Ba의 뒤쪽 깊이(mm): 골성 비인두의 앞뒤 깊이", f"{ALI} → −(Ba의 앞 좌표)"),
 "Co_A": ("유효 중안면 길이(mm): 과두–A점 거리", f"{ALI} + AMASSS 과두 → 3D 거리"),
 "palate_cb": ("구개 길이 ÷ 뒤 두개저 길이", "ANS_PNS ÷ S_Ba"), "mand_cb": ("유효 하악 길이 ÷ 앞 두개저 길이", "Co_Gn ÷ S_N"),
 "logCS_all": ("하악·상악·두개저 대응점 전체의 log 중심 크기(머리 전체 크기)", f"{SSM} → 세 구조 대응점 연결 → 중심 크기 → log"),
 "AlloShape": ("상대성장 모양 점수: 크기가 커질 때 함께 변하는 얼굴 모양 방향으로 얼마나 가 있는지", f"{SSM} → 세 구조 연결 GPA → 좌표 ~ log 중심 크기 + 성별 회귀 → 크기 계수 방향으로의 투영"),
 "Atyp_D2": ("전형성 거리: 공동 축 점수로 본 평균 얼굴에서의 거리(마할라노비스 D²). 클수록 드문 얼굴 구성", "공동 축(F1) 점수 → 마할라노비스 거리 제곱"),
 "Atyp_Procrustes": ("평균 형상과의 Procrustes 거리. 클수록 평균에서 벗어난 모양", f"{SSM} → 공동 GPA → 평균 형상까지 거리"),
 "PLS_t1": ("골격–기도 연동 점수: 골격 좌표 중 설골·Eb·기도 프로파일과 가장 함께 변하는 성분", f"{ALI} 좌표 ↔ 설골·Eb·기도 프로파일 2-Block PLS(85명) 골격 쪽 첫째 점수"),
 "MaxPost_Clivus_ant": ("상악 후방 부위 중심과 두개저 사대 부위 중심의 앞뒤 차(mm)", f"{SSM} → 부위별 대응점 중심(FH 머리 좌표계) → 앞뒤 좌표 차"),
 # ---------------- Step 4 연조직·연동
 "tongue_mL": ("혀 부피(mL)", f"{TS_HM} 혀 라벨 → 복셀 수 × 복셀 부피"),
 "tongue_c_ant": ("혀 무게중심의 앞뒤 위치(mm, PNS 원점). 작을수록 혀가 뒤에 있음", f"{TS_HM} 혀 라벨 → {MID} → 라벨 점 앞 좌표 평균"),
 "tongue_c_sup": ("혀 무게중심의 위아래 위치(mm)", f"{TS_HM} 혀 라벨 → {MID} → 라벨 점 위 좌표 평균"),
 "tongue_low_sup": ("혀 최저점 높이(mm). 작을수록 혀가 아래로 내려감", f"{TS_HM} 혀 라벨 → {MID} → 위 좌표 최솟값"),
 "tongue_len": ("혀 앞뒤 길이(mm)", f"{TS_HM} 혀 라벨 → {MID} → 앞 좌표 최대–최소"), "tongue_height": ("혀 높이(mm)", f"{TS_HM} 혀 라벨 → {MID} → 위 좌표 최대–최소"),
 "tongue_back_ant": ("혀뿌리(혀의 가장 뒤 경계) 앞뒤 위치(mm). 작을수록 혀뿌리가 뒤(기도 쪽)에 있음", f"{TS_HM} 혀 라벨 → {MID} → 앞 좌표 최솟값"),
 "tongue_mandible_ratio": ("혀 부피 ÷ 하악 뼈 부피. 클수록 하악에 비해 혀가 큼", f"{TS_HM} 혀 부피 ÷ AMASSS 하악 분할 부피"),
 "tongue_MMV": ("혀 부피 ÷ 하악·상악 안 공간 부피(MMV). 클수록 구강 공간에 비해 혀가 큼", "tongue_mL ÷ MMV_mL"),
 "tongue_MMVhull": ("혀 부피 ÷ 하악·상악 볼록 껍질 부피", "tongue_mL ÷ MMV_hull_mL"),
 "SoftLoad_MMV": ("연조직 부하 = (혀 + 연구개 부피) ÷ 구강 공간 부피", "(tongue_mL + 연구개 라벨 부피) ÷ MMV_mL"),
 "MMV_mL": ("하악·상악 안쪽 공간 부피(mL): 구개평면 아래 두 뼈의 볼록 껍질 안에서 뼈를 뺀 공간(혀가 들어가는 공간)",
            f"{CT} → AMASSS 하악·상악 분할 → {MID} → 구개평면(위 좌표 0) 아래 점들의 볼록 껍질 → 1 mm 격자 중 껍질 안·뼈 밖 칸 수"),
 "MMV_hull_mL": ("하악·상악 볼록 껍질 전체 부피(mL, 뼈 포함)", "MMV_mL과 같은 볼록 껍질 → 껍질 안 1 mm 격자 칸 수"),
 "masseter_mL": ("교근 좌우 합 부피(mL)", f"{TS_HM} 교근 좌우 라벨 → 부피 합"), "lat_pterygoid_mL": ("외측익돌근 좌우 합 부피(mL)", f"{TS_HM} 외측익돌근 라벨 → 부피 합"),
 "med_pterygoid_mL": ("내측익돌근 좌우 합 부피(mL)", f"{TS_HM} 내측익돌근 라벨 → 부피 합"), "chew_mL": ("저작근(교근·외측·내측익돌근) 합 부피(mL). 측두근은 영상에서 잘려 제외", f"{TS_HM} 세 근육 좌우 합"),
 "sinus_max_mL": ("상악동 좌우 합 부피(mL)", f"{TS_CF} 상악동 라벨 → 좌우 부피 합"),
 "sinus_max_air_frac": ("상악동 안 공기 비율. 작을수록 점막 비후·액체로 차 있음", f"{TS_CF} 상악동 라벨 → 그 안에서 CT −400 HU 미만 복셀 비율"),
 "sinus_frontal_mL": ("전두동 부피(mL). 영상 위가 잘린 3명은 결측", f"{TS_CF} 전두동 라벨 → 부피"),
 "SDp_mm": ("비중격 최대 편위(mm): 비중격이 정중면에서 좌우로 가장 많이 휜 거리", f"{CT} → TotalSegmentator 좌우 비강 라벨 사이 중격 중심선 → 정중면으로부터 최대 좌우 거리"),
 "MP_H": ("설골–하악평면 거리(mm): 설골이 하악평면(Go–Me)에서 아래로 떨어진 거리. 클수록 설골이 낮음", f"{ALI} + 설골(밝기 규칙 검출) → Go 중점–Me 직선까지 수직 거리"),
 "Hy_Me": ("설골–Me 거리(mm)", f"{ALI} + 설골 → 거리"), "Hy_Ba_dist": ("설골–Ba 정중면 거리(mm)", f"{ALI} + 설골 → 정중면 거리"),
 "Hy_sup": ("설골 위아래 위치(mm, PNS 원점). 작을수록 설골이 아래", f"{CT} → 정중 단면에서 하악 아래 뼈 밝기 덩어리로 설골 검출 → {MID} 위 좌표"),
 "Hy_ant": ("설골 앞뒤 위치(mm). 작을수록 설골이 뒤", f"{CT} → 설골 검출 → {MID} 앞 좌표"),
 "L1_MP_deg": ("아래 앞니–하악평면 각(IMPA 방식, °). 클수록 아래 앞니가 앞으로 기욺", f"{TEETH} → 아래 중절치 장축 → 하악평면(Me→Go)과의 정중면 각"),
 "Condyle_Axial_mean": ("과두 축상각 좌우 평균(°): 위에서 본 과두 장축의 방향", f"{CT} → AMASSS 하악 분할 → 과두 장축 → 정중면 수직선과의 축상면 각"),
 "Condyle_Lat_mean": ("과두 측면각 좌우 평균(°): 옆에서 본 과두 기울기", f"{CT} → AMASSS 하악 분할 → 과두 장축의 측면 각"),
 "lever_Pog_mm": ("턱 벌림 지렛대(mm): 좌우 과두를 잇는 축으로 하악을 5° 벌렸을 때 턱끝(Pog)이 뒤로 밀리는 거리. 클수록 입을 벌릴 때 하악이 기도 쪽으로 많이 감",
                  f"{ALI} + AMASSS 과두 최상점 → 과두 축 기준 5° 강체 회전 → Pog 앞뒤 이동량"),
 "lever_Hy_mm": ("같은 5° 회전에서 설골이 뒤로 밀리는 거리(mm, 강체 가정)", f"{ALI} + 설골 → 과두 축 기준 5° 회전 → 설골 앞뒤 이동량"),
 "lever_arm_Pog_sup": ("과두 축과 턱끝(Pog)의 높이 차(mm): 지렛대 팔 길이", f"{ALI} + AMASSS 과두 → 위 좌표 차"),
 "F2_logCS": ("세 구조 공동 GPA의 log 중심 크기", f"{SSM} → 세 구조 연결 → 중심 크기 log"), "F2MM_logCS": ("하악+상악 GPA의 log 중심 크기", f"{SSM} → 하악·상악 연결 → 중심 크기 log"),
}

STRUCT = {"MAND": "하악", "MAX": "상악", "CB": "두개저"}
REGION = {"symphysis": "턱끝 결합부", "body": "하악체", "gonial": "하악각부", "ramus_condyle": "하악지·과두", "anterior_alveolar": "앞쪽 치조부", "palate": "구개",
          "posterior": "상악 후방(결절)", "orbital_floor": "안와 바닥", "anterior_base": "앞 두개저", "sella": "터키안장", "clivus": "사대", "temporal": "측두부"}
ANCHOR = {"symphysis": "Pog·Me·Gn·B 평균", "body": "Gn–좌우 Go 중점", "gonial": "좌우 Go", "ramus_condyle": "Go 위 35 mm", "anterior_alveolar": "A점", "palate": "ANS–PNS 중점",
          "posterior": "PNS", "orbital_floor": "좌우 안와점", "anterior_base": "N–S 중점", "sella": "S", "clivus": "S–Ba 중점·Ba", "temporal": "좌우 귓구멍 점"}


def detail(v):
    if v in D: return D[v]
    if v.startswith("roi_"):
        _, st, rest = v.split("_", 2); reg, kind = rest.rsplit("_", 1); r, s = REGION.get(reg, reg), STRUCT.get(st, st)
        base = f"{SSM} → 각 대응점을 가장 가까운 기준점 앵커({ANCHOR.get(reg, '')})의 부위로 지정 → {s} {r} 부위 점들"
        if kind == "size": return (f"{s} {r} 부위 크기(mm): 부위 대응점들이 중심에서 퍼진 정도(중심 크기). 클수록 그 부위가 큼", base+" → Procrustes 중심 크기")
        ax = {"cleft": "좌우", "cant": "앞뒤", "csup": "위아래"}[kind]
        return (f"{s} {r} 부위 중심의 {ax} 위치(mm, FH 머리 좌표계)", base+f" → 점 평균 좌표의 {ax} 성분(귓구멍 점 기반 좌표계라 귓구멍 점 오류에 민감)")
    if v.startswith("EFA_"):
        eb = "Eb 포함" if "with_Eb" in v else "Eb 제외"; k = v.rsplit("F", 1)[1]
        return (f"골격 연동 {k}번째 요인 점수({eb}): 여러 골격·설골 점의 위치가 함께 움직이는 패턴 중 {k}번째",
                f"{ALI} → Ba·S·N·ANS·PNS·A·좌우 Go·Me·Pog·B·설골({eb}) 좌표 전부 → 탐색적 요인분석(VARIMAX 회전, 85명) → {k}번째 요인 점수")
    if "_rPC" in v:
        st, k = v.split("_rPC"); s = STRUCT.get(st, st)
        return (f"{s} 잔여 형상 {k}번째 성분: ANB·턱끝–대후두공 거리·구조 크기·성별로 설명되지 않고 남은 {s} 모양 차이",
                f"{SSM} → {s} 대응점 좌표를 ANB + 턱끝–대후두공 거리 + 구조 크기 축 + 성별로 회귀한 잔차 → PCA(85명) {k}번째 점수")
    for p, lab, sts in (("F2MM_PC", "하악+상악 배치·모양(크기 제거 GPA)", "하악·상악"), ("F2_PC", "세 구조 배치·모양(크기 제거 GPA, 구조 간 배치 보존)", "하악·상악·두개저"),
                        ("F1_PC", "세 구조 형태(정중 좌표계 그대로, 크기 포함)", "하악·상악·두개저")):
        if v.startswith(p):
            k = v[len(p):]
            return (f"공동 축 {k}번째: {lab}의 {k}번째 주된 변이", f"{SSM} → {sts} 대응점을 한 벡터로 이음 → {'GPA(크기 제거) → ' if 'GPA' in lab else ''}PCA(85명) {k}번째 점수")
    if v.startswith("res_"):
        b = v[4:]; bd = D.get(b, (b, ""))[0].split("(")[0]
        return (f"{bd}에서 ANB·턱끝–대후두공 거리·구조 크기·성별로 설명되는 부분을 뺀 나머지", f"{b} ~ ANB + 턱끝–대후두공 거리 + 구조 크기 축 + 성별 회귀(전체 표본)의 잔차")
    return None
