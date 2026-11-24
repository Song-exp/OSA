"""피처 전처리 설명 보고서: 회귀 이전에 데이터를 어떻게 만들었는지.
1~5절 = CBCT 측정 기준 보고서의 도구·좌표계·기준점·분할 절을 그대로(그림 포함), 6절 밝기 규칙, 7절 통계 요약 변수, 8절 피처 표
(변수명 | 변수 뜻 | 도메인적 의미 | 계산 과정 | CT 이미지). Claude 아티팩트 규격(문서 뼈대 없음, 테마 토큰, 그림 크게 보기)으로 바로 만든다.
출력: Result_py/analysis/all83/preproc/ (index.html, figs/, tools/) — git 제외(환자 영상 포함)
"""
import re, shutil, warnings; warnings.filterwarnings("ignore")
from html import escape
from pathlib import Path
import osa_newfactor as N, osa_all83 as A, osa_feature_guide as G
from osa_var_detail import detail
from osa_var_domain import domain

R = A.R; OUT = R/"analysis/all83/preproc"; RPT = R/"report_85/CBCT_측정기준_보고서_2026-11-15.html"
GLOSS = R/"CBCT/patient_pipeline/v2/report/glossary_images"; MFIG = R/"report_85/method_figs"; VFIG = R/"analysis/all83/figs"


ALI_USE = {'ALI-CBCT 치아 모델': 'ALI-CBCT의 치아용 모델로, 제1대구치 점을 직접 찍습니다(머리뼈 점은 쓰지 않음).', 'AMASSS': 'ALI-CBCT로 찍은 점을 활용하여 분할 표면을 환자끼리 맞추고 부위 12곳으로 나눕니다.', 'TotalSegmentator': 'ALI-CBCT로 찍은 점을 활용하여 분할 라벨을 정중 좌표계로 옮겨 잽니다.', '치아 분할 (UNIVERSALLAB)': 'ALI-CBCT로 찍은 점을 활용하여 턱 영역만 잘라 넣고, 제1대구치 번호를 확인합니다.'}
SADT = "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools"
TOOLS = {  # 카드 제목 → ((링크, 링크 글), 대표 그림, 그림 설명)
    "ALI-CBCT": ((SADT, "SlicerAutomatedDentalTools (ALI_CBCT 모듈)"), "ali_overview.png", "추출 부위: 머리뼈 기준점 16개와 제1대구치 점 (한 환자, 전체 위치)"),
    "ALI-CBCT 치아 모델": ((SADT, "SlicerAutomatedDentalTools (ALI_CBCT 치아 모델)"), "UR6MB.png", "추출 부위: 위 제1대구치 근심협측 교두점 예시"),
    "AMASSS": ((SADT, "SlicerAutomatedDentalTools (AMASSS 모듈)"), "amasss_overview.png", "추출 부위: 하악·상악·두개저·상기도 분할 (구조별 최대 단면)"),
    "TotalSegmentator": (("https://github.com/wasserth/TotalSegmentator", "TotalSegmentator"), "totalseg_overview.png", "추출 부위: 비인두·구인두·하인두, 연구개·경구개, 혀 (정중 단면)"),
    "치아 분할 (UNIVERSALLAB)": ((SADT + "/releases", "SlicerAutomatedDentalTools 모델 배포 (UNIVERSALLAB_MODEL)"), "teeth_overview.png", "추출 부위: 치아 32개 분할 (위에서 본 투영, 빨강 = 송곳니·제1대구치)"),
}

def tool_sections():
    h = RPT.read_text(encoding="utf-8"); sec = h[h.index('<h2 id="tools">'):h.index('<h2 id="metrics">')]
    css = re.findall(r"<style>(.*?)</style>", h, re.S)[0]
    def cp(m):
        src = m.group(1); name = Path(src).name; f = (GLOSS if "glossary_images" in src else MFIG)/name
        shutil.copy(f, OUT/"tools"/name); return f'"tools/{name}"'
    sec = re.sub(r'"((?:\.\./CBCT/patient_pipeline/v2/report/glossary_images|method_figs)/[^"]+\.png)"', cp, sec)
    shutil.copy(VFIG/"Upper_Taper.png", OUT/"tools"/"teeth_overview.png")
    if (VFIG/"totalseg_overview.png").exists(): shutil.copy(VFIG/"totalseg_overview.png", OUT/"tools"/"totalseg_overview.png")
    parts = sec.split('<div class="tool">'); out = [parts[0]]
    for part in parts[1:]:
        name = re.match(r"<h3>([^<]+)</h3>", part).group(1)
        link, img, cap = TOOLS.get(name, (None, None, None))
        if link:
            au = f'<p><b>ALI-CBCT 점 활용:</b> {ALI_USE[name]}</p>' if name in ALI_USE else ""
            part = part.replace(f"<h3>{name}</h3>", f'<h3>{name}</h3><p class="tlink"><a href="{link[0]}" target="_blank" rel="noopener">{link[1]} ↗</a></p>{au}', 1)
            end = part.index("</div>")
            part = (f'<div class="tgrid"><div>{part[len(f"<h3>{name}</h3>"):end].join(["", ""]) if False else ""}' if False else part)
            fig = f'<figure class="tfig"><a href="tools/{img}"><img src="tools/{img}" alt="{cap}"></a><figcaption>{cap}</figcaption></figure>'
            part = part[:end] + fig + part[end:]
        out.append(part)
    sec = '<div class="tool">'.join(out)
    figs3 = "".join(f'<figure class="card"><a href="tools/{f}"><img src="tools/{f}" alt="{c}"></a><figcaption>{c}</figcaption></figure>' for f, c in (
        ("frame_setup.png", "(가) 좌표계 세우기: FH(노랑)는 RPo·LPo·ROr로 만들고, 원점 N에서 앞·위를 + 방향으로 잡습니다. 정중면(하늘)은 N·Ba를 지나 FH에 수직입니다."),
        ("frame_error.png", "(나) 문제: P_37은 오른쪽 귓구멍 점(RPo)이 머리 안쪽 엉뚱한 곳에 찍혀, 그 점으로 만든 FH(빨강)가 크게 기웁니다."),
        ("frame_fix.png", "(다) 해결: 반대쪽 LPo를 정중면에 거울로 비춘 자리로 RPo를 대신하면 FH(노랑)가 바로 섭니다. 4점 최소제곱 평면(하늘 점선)도 같은 높이를 지납니다.")))
    sec = sec.replace("<h3>② 정중 좌표계", f'<div class="grid frame3">{figs3}</div>\n<h3>② 정중 좌표계', 1)
    if not (OUT/"tools"/"frame_fix.png").exists():
        import osa_frame_figs as FF; FF.setup(); FF.error_fix()                                  # 머리 좌표계 그림(없을 때만)
    sec = sec.replace("밝기 450 이상인", "밝기 400 이상(Dong 2019의 0.40 mm 복셀 최적 기준)인", 1)
    for o, n in (("공기가 1~10 mm 두께의 후두개를", "공기가 0.8 mm(영상 2칸) 이상 두께의 후두개를"),
                 ("앞쪽 갈래(후두개곡)가 끝나는 가장 아랫점을 Eb로 정합니다.", "앞쪽 갈래(후두개곡)가 끝나는 가장 아랫점을 Eb로 정합니다. 탐색 높이는 설골 몸통 아래 끝부터이고, 2줄 이상 잡혀야 인정합니다(6-2·6-5절)."),
                 ("그 뒤쪽 1/3의 꼭대기를 기준점으로 삼습니다.", "그 뒤쪽 1/3의 꼭대기를 Eb 탐색 기준점으로 삼습니다(설골 피처의 설골점은 문헌 정의인 몸통의 가장 앞쪽·위쪽 점 H, 6-1절)."),
                 ("재지 않습니다(4명).", "재지 않습니다(6명)."),
                 ("단단한 뼈(밝기 300 이상)만 남깁니다.", "단단한 뼈(밝기 400 이상, 설골과 같은 문헌 뼈 기준 Dong 2019)만 남깁니다."),
                 ("연골은 밝기가 300보다 낮아", "연골은 밝기가 400보다 낮아")):
        assert sec.count(o) == 1, o; sec = sec.replace(o, n)
    return sec, css


EXTRA = """<h2 id="rules">6. 밝기 규칙 (설골·Eb·인두 기도)</h2>
<p>프로그램이 찍어 주지 않는 설골, 후두개곡 최하점(Eb), 인두 기도는 CBCT 영상의 밝기 값으로 직접 찾았습니다. 세 가지 모두 <b>같은 두 단계(밝기로 칸 표시 → 덩어리 고르기)</b>를 거친 뒤, 각자 다른 기준으로 잘라 냅니다. 서로 이어져 있어 <b>설골 → Eb → 인두 기도</b> 순서로 구합니다(Eb는 설골 근처에서 찾고, 인두 기도는 Eb를 아래 경계로 씁니다). 모든 계산은 환자마다 그 환자의 3D 영상 전체에서 자동으로 했고, 아래 그림은 예시 환자 1명입니다.</p>
<h3>6-0. 공통 과정</h3>
<figure class="card" style="margin:10px 0"><a href="tools/rule_common.png"><img src="tools/rule_common.png" alt="밝기 규칙 공통 과정"></a><figcaption>(가) 원본 정중 단면 → ① 밝기 범위에 드는 칸을 공기(파랑)·뼈(주황)로 표시 → ② 맞닿은 칸끼리 덩어리로 묶고, 목적에 맞는 덩어리(인두 공기, 설골)만 남깁니다.</figcaption></figure>
<div class="tw"><table class="ft"><thead><tr><th>단계</th><th>하는 일</th><th>기준값과 근거</th></tr></thead><tbody>
<tr><td>① 밝기로 칸 표시</td><td>영상의 각 칸(0.4 mm)을 밝기 값으로 공기 또는 뼈로 표시합니다.</td><td><b>공기 −1200~−400</b>: Roh 2025(기준 논문)의 인두 기도 부피(PAV) 기준을 그대로 씀 → Eb·인두 기도에 사용<br><b>뼈 400 이상</b>: 설골 전용 CBCT 기준은 문헌에 없어, 같은 복셀 크기(0.40 mm, 우리 영상 0.39 mm)의 CBCT에서 뼈 부피가 가장 정확했던 최소 기준 400을 그대로 씀(Dong 2019, BMC Oral Health) → 설골에 사용</td></tr>
<tr><td>② 덩어리 고르기</td><td>맞닿은 같은 종류의 칸끼리 한 덩어리로 묶고(연결 성분 분석), 위치·크기 조건에 맞는 덩어리만 남깁니다. 입안·부비동 공기나 다른 뼈처럼 따로 떨어진 덩어리는 여기서 빠집니다.</td><td>Roh 2025가 Mimics에서 쓴 영역 확장(region growing, 한 점에서 같은 밝기로 이어진 칸으로 퍼져 나가기)과 같은 원리를 자동화한 것</td></tr>
</tbody></table></div>
<p><b>ALI-CBCT 점 활용:</b> ALI-CBCT로 찍은 점(N·S·Ba·ANS·PNS)으로 만든 정중 좌표계(원점 PNS)에서 모든 칸을 봅니다.</p>
<p class="note">CBCT 밝기는 일반 CT의 HU처럼 장비 사이에 표준화되어 있지 않아, 같은 기준이라도 장비·촬영 조건에 따라 경계가 조금씩 달라질 수 있습니다(Wagendorf 2023). 그래서 뼈 기준을 포함한 자체 선택 수치는 6-5절의 민감도 분석으로 확인했습니다.</p>
<h3>6-1~6-3. 항목별 자르는 기준</h3>
<div class="mcard"><div><a href="tools/rule_hyoid.png"><img src="tools/rule_hyoid.png" alt="설골"></a><p class="note">주황 = 고른 설골 덩어리, 노랑 ○ = 설골점 H, 노랑 + = Eb 탐색 기준점, 빨강 점선 = Eb 탐색 아래 한계, 초록 = 하악평면</p></div><div><h3>6-1. 설골 (Hy)</h3><dl>
<dt>자르는 기준</dt><dd><ol class="steps"><li>정중선 좌우 3 mm 두께의 뼈 칸을 겹쳐 봅니다(최대 투영).</li><li>PNS 기준 앞뒤 −33~+12 mm 탐색 창 안에서, 창 가장자리에 닿지 않고 넓이 8~150 mm², 앞뒤 길이 18 mm 미만인 덩어리 중 가장 큰 것을 설골로 고릅니다(하악·척추처럼 큰 뼈는 여기서 빠짐).</li><li><b>설골점 H:</b> 덩어리에서 앞쪽·위쪽(45° 방향)으로 가장 멀리 나간 점, 곧 몸통의 앞위 모서리입니다(두부계측 문헌의 설골점 정의).</li><li><b>Eb 탐색 기준점:</b> 덩어리 뒤쪽 1/3의 가장 높은 점은 Eb를 찾는 내부 기준으로만 씁니다. 덩어리의 가장 낮은 높이(몸통 아래 끝)는 Eb 탐색의 아래 한계가 됩니다.</li></ol></dd>
<dt>만든 값</dt><dd>설골 위치(앞뒤·위아래), 하악평면까지 거리(MP-H), 턱끝·두개저까지 거리, Eb 탐색의 기준점</dd>
<dt>근거</dt><dd><b>선행연구:</b> 뼈 기준 400(Dong 2019), 설골점 H의 정의(OSA·두부계측 연구의 "설골 몸통의 가장 앞쪽·위쪽 점"). <b>자체 선택(민감도 확인):</b> 정중 두께 ±3 mm는 ±2~3 mm의 평평한 구간 안에서 가장 좋았고, 탐색 창은 넓혀도 결과가 같았습니다. 크기 조건 중 실제로 작동하는 것은 작은 점 제외뿐입니다(진짜 설골은 최소 13 mm², 걸러진 점은 최대 5.4 mm²). OSA 연구에서는 설골점을 보통 사람이 찍습니다.</dd>
<dt>ALI-CBCT 점 활용</dt><dd>ALI-CBCT로 찍은 점(Go·Me)으로 하악평면을 만들어 설골까지 거리를 잽니다.</dd>
<dt>코드</dt><dd><code>cbct_eb_detect.hyoid</code></dd></dl></div></div>
<div class="mcard"><div><a href="tools/Eb.png"><img src="tools/Eb.png" alt="Eb"></a><p class="note">후두개곡 직접 검출 예시</p></div><div><h3>6-2. 후두개곡 최하점 (Eb)</h3><dl>
<dt>자르는 기준</dt><dd><ol class="steps"><li>정중선 좌우 6 mm 안의 공기 칸(−400 미만)을 겹쳐 보고(후두개곡은 정중 주름 양옆에 한 쌍으로 있음), PNS 45 mm 뒤부터의 영상에서 가장 큰 공기 덩어리(인두)만 씁니다.</li><li><b>설골 몸통 아래 끝</b>부터 설골 기준점 22 mm 위까지, 한 줄씩 앞뒤로 훑습니다. 후두개곡의 바닥(설골–후두개 인대)은 설골 몸통에 붙으므로, 그보다 아래는 보지 않습니다.</li><li>공기가 0.8 mm(영상 2칸) 이상 두께의 조직(후두개)을 사이에 두고 앞뒤 두 갈래로 갈라지는 줄을 찾고, 앞 갈래(후두개곡)의 가장 아랫점을 Eb로 정합니다. 2줄 이상 잡혀야 인정합니다.</li><li>못 찾으면(혀뿌리와 후두개가 맞닿은 경우 등), 직접 찾은 환자들의 '설골 → Eb' 거리 중앙값을 설골점에 더해 추정합니다.</li></ol></dd>
<dt>만든 값</dt><dd>인두 기도의 아래 경계, 기도 길이(PAL)의 끝점</dd>
<dt>근거</dt><dd><b>선행연구:</b> Eb 정의(후두개곡의 가장 아랫점, Yao 2023), 공기 기준(Roh 2025). <b>해부학:</b> 아래 한계 = 설골 몸통 아래 끝. <b>자체 선택(민감도 확인·육안 판정):</b> 정중 두께 ±6 mm(±4~6 mm 평평), 후두개 두께 하한 0.8 mm·최소 2줄·탐색 창 뒤쪽 끝 45 mm(엑셀 PAL과 더 가까워져 바꿈). 후두개 두께 상한과 앞 갈래의 앞쪽 범위는 실제로 거르는 것이 없었습니다. 설골 기준 추정은 자체 방법입니다.</dd>
<dt>약점</dt><dd>설골을 찾은 81명 중 71명은 직접 검출, 나머지는 추정값입니다. 영상 아래가 잘렸거나 Eb가 촬영 범위 밖인 환자는 기도 지표가 결측입니다(6명).</dd>
<dt>코드</dt><dd><code>cbct_eb_detect.vallecula</code></dd></dl></div></div>
<div class="mcard"><div><a href="tools/PAV.png"><img src="tools/PAV.png" alt="PAV"></a><p class="note">인두 기도 부피의 경계</p><a href="tools/PMCA.png"><img src="tools/PMCA.png" alt="PMCA"></a><p class="note">중심선 수직 최소 단면</p><a href="tools/PAL.png"><img src="tools/PAL.png" alt="PAL"></a><p class="note">PNS–Eb 기도 길이</p></div><div><h3>6-3. 인두 기도</h3><dl>
<dt>자르는 기준</dt><dd><ol class="steps"><li>S–PNS 선보다 뒤쪽의 공기 덩어리 중, 정중선 좌우 8 mm 안에 칸이 가장 많은 덩어리를 인두 기도로 고릅니다.</li><li><b>위:</b> S–PNS 면, <b>앞:</b> PNS를 지나는 세로면, <b>아래:</b> Eb를 지나는 수평면으로 잘라 측정 구간을 정합니다.</li><li>기도 한가운데 선(중심선)을 구하고, 0.4 mm마다 그 선에 직각으로 잘라 단면 넓이를 잽니다.</li></ol></dd>
<dt>만든 값</dt><dd>PAV(부피 = 경계 안 칸 수 × 칸 부피), PAL·H_col(길이), PMCA(Eb 위 3 mm~PNS 구간의 최소 단면), 수력학적 직경(HD), Eb → PNS 20구간 단면적 프로파일 → 기도 굵기 주성분(AWshape)</dd>
<dt>근거</dt><dd>공기 기준과 위 경계는 Roh 2025와 같습니다(Roh의 Se를 S로 가정, 연구실 확인 중). <b>아래 경계는 Roh와 다릅니다:</b> Roh는 Eb를 지나 FH에 평행한 면을 썼고, 우리는 귓구멍 점 오류를 피하려고 정중 좌표계의 수평면을 썼습니다. 중심선 직각 단면과 20구간 프로파일은 자체 선택입니다(Roh는 PMCA 단면 방향을 밝히지 않음. 최소 단면적을 공기 흐름에 수직인 면으로 정의한 연구는 있음). <b>민감도 확인:</b> 덩어리 고르기 ±8 mm·중심선 5 mm는 결과에 영향이 없었고, PMCA 아래 여유는 경계 잘림을 피하는 가장 작은 값 3 mm로 정했습니다(6-5절).</dd>
<dt>검증</dt><dd>엑셀(연구실 수작업) 값과의 상관: PAV 0.98(24명), PMCA 0.86(79명). PMCA는 우리 값이 평균 약 35 mm² 큽니다.</dd>
<dt>ALI-CBCT 점 활용</dt><dd>ALI-CBCT로 찍은 점(S·PNS)을 활용하여 위·앞 경계면과 기도 길이의 시작점을 정합니다.</dd>
<dt>코드</dt><dd><code>cbct_pmca.airway · sections</code>, <code>cbct_pav.volume</code>, <code>shape_extra_vectors.profile</code></dd></dl></div></div>
<h3>6-4. 다른 기도 값과의 관계</h3>
<ul><li><b>비인두·구인두·하인두 구간 값</b>(nasopharynx_min_area 등)은 이 규칙이 아니라 TotalSegmentator 인두 라벨(외부 경계)로 잰 별개의 값입니다. 경계가 달라, 구인두 최소 단면적과 PMCA의 상관은 0.21에 그쳤습니다.</li>
<li><b>AMASSS 상기도 분할</b>은 이 계산에 쓰지 않았습니다. 기도 전체를 칠한 것이라 Roh의 경계와 맞지 않아, 점검과 형태 분석에만 썼습니다.</li></ul>
{sens}
<p class="sum">한 줄 요약: 밝기로 공기·뼈 칸을 표시하고(공기 기준 Roh 2025, 뼈 기준 Dong 2019) 덩어리를 고른 뒤, 설골은 크기 조건으로, Eb는 후두개를 사이에 둔 공기 갈라짐으로, 인두 기도는 S–PNS·PNS·Eb 경계면으로 잘라 냅니다.</p>
<h2 id="stats">7. 통계로 요약한 변수 (PCA·요인분석·잔차)</h2>
<p>측정값이나 표면 대응점 좌표를 통계 모델로 요약한 변수입니다. 대부분 85명 기준으로 한 번 적합해 고정했습니다. 아래 그림은 행 = 구성 변수, 열 = 축, 칸 = 적재량입니다.</p>
<div class="grid">
<div class="card tool"><h3>주성분분석 (PCA)</h3><p>서로 상관된 변수들의 공통 변이를 축으로 요약합니다. 구조 크기 축(부위 크기 12개), 기도 프로파일 축(20구간 단면적), 공동 축(하악·상악·두개저 표면 대응점 좌표 전체), 잔여 형상 축(기존 인자로 설명되지 않는 모양)이 있습니다.</p></div>
<div class="card tool"><h3>요인분석 (EFA)</h3><p>골격·설골(·Eb) 기준점 좌표가 함께 움직이는 패턴을 VARIMAX 회전으로 찾습니다.</p></div>
<div class="card tool"><h3>잔차·묶음 점수</h3><p>잔차 변수는 기존 인자(ANB·턱끝–대후두공 거리·크기·성별)로 설명되는 부분을 뺀 나머지입니다. 수직 배치 지표(VERT_Z)는 세 성분의 z점수 평균입니다.</p></div>
</div>
<div class="grid">{pcs}</div>"""


def fig(src, cap):
    return f'<figure class="card"><a href="{src}"><img src="{src}" alt="{escape(cap)}"></a><figcaption>{escape(cap)}</figcaption></figure>'


EDIT_CSS = """#edbar{position:sticky;top:env(safe-area-inset-top,0px);z-index:40;display:flex;flex-wrap:wrap;gap:8px;align-items:center;background:var(--card);border-bottom:1px solid var(--line);padding:8px max(16px,env(safe-area-inset-right,0px)) 8px max(16px,env(safe-area-inset-left,0px))}
body::before{content:"";position:fixed;inset:0 0 auto;height:env(safe-area-inset-top,0px);z-index:41;background:var(--card)}
#edbar button{font:inherit;padding:5px 12px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--fg);cursor:pointer}
#edbar button.on{background:var(--accent);border-color:var(--accent);color:#fff}#edbar button:disabled{opacity:.45;cursor:default}#edst{color:var(--muted);font-size:.85rem}
html.editing [data-eid]{outline:1px dashed var(--accent);outline-offset:2px;cursor:text}html.editing [data-eid]:focus{outline:2px solid var(--accent)}"""

EDIT_JS = """<script id="edits" type="application/json">{}</script>
<script>(function(){
const RESET=':root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0;font:14px system-ui,-apple-system,"Segoe UI",sans-serif}img{max-width:100%}[hidden]{display:none!important}';
const SRC=document.body.innerHTML;            /* 이 시점의 본문 = 게시된 원본(고친 내용은 아래 EDITS에 따로 있음) */
let EDITS={};try{EDITS=JSON.parse(document.getElementById('edits').textContent||'{}');}catch(e){}
for(const id in EDITS){const el=document.querySelector('[data-eid="'+id+'"]');if(el)el.innerHTML=EDITS[id];}
const bt=document.getElementById('edbt'),sv=document.getElementById('edsv'),st=document.getElementById('edst');let dirty=false;
bt.addEventListener('click',()=>{const on=!document.documentElement.classList.contains('editing');document.documentElement.classList.toggle('editing',on);
 bt.classList.toggle('on',on);bt.textContent=on?'수정 끝내기':'글 수정';document.querySelectorAll('[data-eid]').forEach(e=>{e.contentEditable=on?'true':'false';});
 st.textContent=on?'점선 칸을 눌러 고치세요. 다 고치면 "변경 저장"을 누르세요.':(dirty?'저장하지 않은 변경이 있습니다.':'');});
document.addEventListener('input',e=>{const el=e.target.closest&&e.target.closest('[data-eid]');if(!el||!document.documentElement.classList.contains('editing'))return;
 EDITS[el.dataset.eid]=el.innerHTML;dirty=true;sv.disabled=false;st.textContent='저장하지 않은 변경이 있습니다.';});
document.addEventListener('click',e=>{if(document.documentElement.classList.contains('editing')&&e.target.closest('[data-eid] a'))e.stopImmediatePropagation();},true);
sv.addEventListener('click',async()=>{sv.disabled=true;st.textContent='저장하는 중…';
 const art=window.claude&&window.claude.use?await window.claude.use('artifact'):null;
 if(!art){st.textContent='이 화면에서는 저장할 수 없습니다. claude.ai에서 페이지를 열어 저장하세요.';sv.disabled=false;return;}
 const data=JSON.stringify(EDITS).replace(/</g,'\\\\u003c');
 const body=SRC.replace(/(<script id="edits" type="application\\/json">)[\\s\\S]*?(<\\/script>)/,function(m,a,b){return a+data+b;});
 const html='<!doctype html><html><head><meta charset=utf8><meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover"><style>'+RESET+'</style></head><body>'+body+'</body></html>';
 try{await art.publish(html);dirty=false;st.textContent='저장했습니다. 새 버전으로 다시 열립니다.';}
 catch(err){const c=err&&err.code;st.textContent=c==='conflict'?'다른 곳에서 먼저 저장해 이번 변경은 반영되지 않았습니다. 다시 고쳐 저장하세요.':(c==='not_granted'||c==='not_writer')?'편집 권한이 없어 저장할 수 없습니다.':'저장하지 못했습니다. 잠시 뒤 다시 누르세요.';sv.disabled=!dirty;}});
})();</script>"""


def editable(page):
    """본문 글 요소에 data-eid를 붙이고(그림이 든 칸 제외), 편집 막대·저장 스크립트를 넣는다."""
    import re
    i0 = page.index("<main>"); head, body = page[:i0], page[i0:]
    n = [0]
    def tag(m):
        t, rest = m.group(1), m.group(2) or ""
        if t == "td":
            end = body.find("</td>", m.end());
            if "<img" in body[m.end():end]: return m.group(0)
        n[0] += 1; return f'<{t}{rest} data-eid="e{n[0]}">'
    body = re.sub(r"<(p|li|h1|h2|h3|h4|figcaption|td)((?:\s[^>]*)?)>", tag, body)
    head = head.replace("</style>", EDIT_CSS+CMT_CSS+"</style>", 1)
    bar = '<div id="edbar"><button type="button" id="edbt">글 수정</button><button type="button" id="edsv" disabled>변경 저장</button><span id="edst" aria-live="polite"></span><span class="note">글을 드래그하면 댓글 버튼이 뜹니다</span></div>\n'
    return head + bar + body.replace("</main>", "</main>\n"+EDIT_JS+"\n"+CMT_JS, 1)


CMT_CSS = """#cmtb{position:absolute;z-index:50;font:inherit;font-size:.85rem;padding:4px 10px;border-radius:6px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer;box-shadow:0 2px 6px rgba(0,0,0,.2)}"""
# 글을 드래그하면 '댓글' 버튼이 뜨고, 누르면 claude.ai 댓글 창이 그 글을 인용해 열린다(comments composer_only). 글 수정 중에는 숨긴다.
CMT_JS = """<script>(async function(){
const c=await claude.use("comments"); if(!c) return;
const b=document.createElement("button"); b.id="cmtb"; b.type="button"; b.textContent="💬 댓글"; b.hidden=true; document.body.appendChild(b);
let r=null; const H=()=>{b.hidden=true;r=null};
function upd(){const s=getSelection(), m=document.querySelector("main");
  if(!s||s.isCollapsed||!s.rangeCount||document.documentElement.classList.contains("editing")||!s.toString().trim())return H();
  const g=s.getRangeAt(0); if(!m||!m.contains(g.commonAncestorContainer))return H();
  r=g.cloneRange(); const k=g.getBoundingClientRect();
  b.style.left=Math.max(8,Math.min(innerWidth-100,k.left+scrollX))+"px"; b.style.top=(k.bottom+scrollY+6)+"px"; b.hidden=false}
document.addEventListener("mouseup",e=>{if(e.target!==b)setTimeout(upd,0)});
document.addEventListener("touchend",()=>setTimeout(upd,300));
document.addEventListener("keyup",e=>{if(e.shiftKey)upd()});
document.addEventListener("selectionchange",()=>{const s=getSelection();if(!s||s.isCollapsed)H()});
b.addEventListener("mousedown",e=>e.preventDefault());
b.addEventListener("click",async()=>{const t=r;H();if(!t)return;
  try{await c.openComposer({range:t})}catch(e){if(e&&e.code==="unavailable")b.remove()}});
})();</script>"""


def main():
    for sub in ("", "tools", "figs", "figs/pc"): (OUT/sub).mkdir(parents=True, exist_ok=True)
    d = N.data(); cat = A.catalog()
    sec, css = tool_sections()
    for k in ("PAV", "PMCA", "PAL", "Eb"): shutil.copy(GLOSS/f"{k}.png", OUT/"tools"/f"{k}.png")
    imgs = "".join(fig(f"tools/{k}.png", c) for k, c in (("PAV", "인두 기도 부피의 경계"), ("PMCA", "중심선 수직 최소 단면"), ("PAL", "PNS–Eb 기도 길이"), ("Eb", "후두개곡 최하점")))
    pcs = ""
    for f, c in (("SIZE_PC1", "구조 크기 축 적재량"), ("AWshape", "기도 프로파일 PC 적재량"), ("F1_load", "공동 축 F1 (부위 × 방향)"), ("F2_load", "공동 축 F2"), ("F2MM_load", "공동 축 F2MM"),
                 ("MAND_rPC_load", "하악 잔여 형상"), ("MAX_rPC_load", "상악 잔여 형상"), ("CB_rPC_load", "두개저 잔여 형상"), ("EFA_with_Eb", "요인분석 (Eb 포함)"), ("EFA_without_Eb", "요인분석 (Eb 제외)")):
        s = VFIG/"pc"/f"{f}.png"
        if s.exists(): shutil.copy(s, OUT/"figs/pc"/s.name); pcs += fig(f"figs/pc/{s.name}", c)
    # 피처 목록 (가이드와 같은 범위)
    step = {}
    for k in N.R11: step.setdefault(N.X11[k], set()).add("Step 1"); step.setdefault(N.M11[k], set()).add("Step 1")
    for v in N.STEP2: step.setdefault(v, set()).add("Step 2")
    for v in N.STEP3: step.setdefault(v, set()).add("Step 3")
    for v in N.step4(d): step.setdefault(v, set()).add("Step 4")
    step.pop("a_PAV_mm3", None); step.setdefault("a_PAV", set()).add("Step 1")
    import osa_var_figs as VF
    _, figs = VF.build(d, list(step))
    groups = {}
    for v in sorted(step):
        g, code = G.where(v); de = detail(v) or cat.get(v) or ("", ""); groups.setdefault(g, []).append((v, de, code))
    body = []
    for g in "ABCDEFG":
        if g not in groups: continue
        t, why = G.SRC[g]
        rows = []
        for v, (mean, proc), code in groups[g]:
            th = ""
            if v in figs:
                shutil.copy(figs[v], OUT/"figs"/figs[v].name); th = f'<a href="figs/{figs[v].name}"><img class="thumb" src="figs/{figs[v].name}" alt="{escape(v)}"></a>'
            else: th = '<span class="note">계산식 변수</span>'
            nm = v.replace("_M1", "_molar")
            rows.append(f'<tr data-q="{escape((v+" "+mean+" "+domain(v)).lower())}"><td><code>{escape(nm)}</code><br><span class="note">{", ".join(sorted(step[v]))}</span></td><td>{escape(mean)}</td>'
                        f'<td>{escape(domain(v))}</td><td>{escape(proc)}<br><span class="note">코드: <code>{escape(code)}</code></span></td><td>{th}</td></tr>')
        body.append(f'<h3 id="g{g}">{escape(t)} <span class="note">{len(rows)}개</span></h3><p class="note">{escape(why)}</p>'
                    f'<div class="tw"><table class="ft"><thead><tr><th>변수명</th><th>변수 뜻</th><th>도메인적 의미 (선행연구에서의 쓰임)</th><th>계산 과정</th><th>CT 이미지</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    n_all = sum(len(v) for v in groups.values())
    nav = "".join(f'<a href="#g{g}">{G.SRC[g][0].split(".")[0]}. {G.SRC[g][0].split(". ", 1)[1].split(" (")[0]}</a>' for g in "ABCDEFG" if g in groups)
    feats = (f'<h2 id="features">8. 피처 표 ({n_all}개)</h2><p>분석에 넣은 모든 피처를 원천별로 정리했습니다. 이름 아래 회색 글자는 그 피처가 쓰인 분석 단계입니다. '
             f'<b>도메인적 의미</b>는 교정 두부계측·OSA 연구에서 그 지표가 어떻게 쓰였는지이며, 확인한 근거는 기준 논문 Roh 2025입니다. 분석법 이름(Steiner, McNamara 등)은 지표의 출처를 밝힌 것이고, 우리가 새로 만든 변수는 "자체 정의"로 표시했습니다. '
             f'CT 이미지는 예시 환자 1명의 CBCT에 해당 위치를 그린 것이며(누르면 크게), Roh 지표 그림은 위 측정 기준 그림을 함께 씁니다.</p>'
             f'<div class="ctl"><label for="q">변수 찾기</label> <input id="q" type="search" placeholder="이름·뜻·의미로 찾기 (예: 하악, PMCA, 혀)" oninput="filt(this.value)"></div>'
             f'<nav class="sub">{nav}</nav>' + "".join(body))
    tok_dark = "--bg:#1a1a19;--card:#222220;--fg:#f0efec;--muted:#c3c2b7;--line:#3a3a37;--accent:#3987e5;--warn:#e3b65c;--warnbg:#3a2e14;--sum:#1d2633"
    page = f"""<title>OSA 피처 전처리</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>{css}
:root[data-theme="dark"]{{{tok_dark};color-scheme:dark}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{color-scheme:dark}}}}
main{{padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}}
.tw{{overflow-x:auto;max-width:100%}}table.ft{{border-collapse:collapse;width:100%;font-size:.86rem}}table.ft th,table.ft td{{border:1px solid var(--line);padding:6px 8px;vertical-align:top}}
table.ft th{{background:var(--sum);position:sticky;top:0}}table.ft td:nth-child(1){{white-space:nowrap}}table.ft td:nth-child(2),table.ft td:nth-child(3),table.ft td:nth-child(4){{min-width:200px}}
img.thumb{{width:130px;max-width:100%;border-radius:4px}}figure.card{{margin:0}}figure.card img{{width:100%}}figure.card figcaption{{font-size:.82rem;color:var(--muted)}}
.tlink{{margin:2px 0 6px!important;font-size:.88rem}}.tfig{{margin:10px 0 0;max-width:420px}}.tfig img{{width:100%;border:1px solid var(--line);border-radius:4px}}.tfig figcaption{{font-size:.8rem;color:var(--muted)}}.frame3{{grid-template-columns:repeat(auto-fill,minmax(300px,1fr));margin:10px 0}}.frame3 figure:first-child{{grid-column:1/-1}}nav.sub{{display:flex;flex-wrap:wrap;gap:4px 14px;margin:8px 0}}.ctl{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0}}.ctl input{{font:inherit;padding:5px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);min-width:0;width:min(420px,100%)}}
#lb{{position:fixed;inset:0;z-index:100;background:rgba(0,0,0,.82);display:flex;align-items:center;justify-content:center;padding:max(16px,env(safe-area-inset-top,0px)) max(16px,env(safe-area-inset-right,0px)) max(16px,env(safe-area-inset-bottom,0px)) max(16px,env(safe-area-inset-left,0px));cursor:zoom-out}}
#lb img{{max-width:100%;max-height:100%;background:#fff;border-radius:4px}}#lb button{{position:absolute;top:max(12px,env(safe-area-inset-top,0px));right:max(12px,env(safe-area-inset-right,0px));font:inherit;padding:6px 12px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}}
a:focus-visible,button:focus-visible,input:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}</style>
<main>
<h1>CBCT 피처 전처리와 정의</h1>
<p class="sub">회귀분석 이전 단계: 어떤 도구로 데이터 점을 뽑고, 각 피처를 어떻게 계산했는지 · 대상 83명 · 2026-11-18</p>
<nav><a href="#tools">1. 사용한 프로그램</a><a href="#frames">2. 좌표계</a><a href="#ali">3. 기준점</a><a href="#amasss">4. 구조 분할</a><a href="#other">5. 연조직·치아 분할</a><a href="#rules">6. 밝기 규칙</a><a href="#sens">6-5. 민감도 분석</a><a href="#stats">7. 통계 요약 변수</a><a href="#features">8. 피처 표</a></nav>
{sec}
{EXTRA.format(imgs=imgs, pcs=pcs, sens=(Path(__file__).with_name("osa_preproc_sens.html")).read_text(encoding="utf-8"))}
{feats}
</main>
<div id="lb" hidden role="dialog" aria-label="그림 크게 보기"><button type="button">닫기</button><img id="lbi" alt=""></div>
<script>document.addEventListener('click',e=>{{const a=e.target.closest('a[href$=".png"]');const lb=document.getElementById('lb');
if(a){{e.preventDefault();document.getElementById('lbi').src=a.getAttribute('href');lb.hidden=false;return;}}if(e.target.closest('#lb'))lb.hidden=true;}});
document.addEventListener('keydown',e=>{{if(e.key==='Escape')document.getElementById('lb').hidden=true;}});
function filt(q){{q=q.trim().toLowerCase();document.querySelectorAll('table.ft tbody tr').forEach(r=>{{r.hidden=q&&!r.dataset.q.includes(q);}});}}</script>
"""
    page = editable(page)
    (OUT/"index.html").write_text(page, encoding="utf-8")
    files = sorted(str(p.relative_to(OUT)).replace("\\", "/") for p in OUT.rglob("*.png"))
    refs = set(re.findall(r'(?:src|href)="((?:tools|figs)/[^"]+)"', page))
    print("saved", OUT/"index.html", round(len(page)/1e6, 2), "MB | 피처", n_all, "| 그림", len(refs), "| 없는 그림", sorted(r for r in refs if not (OUT/r).exists()))
    (OUT/"files.txt").write_text("\n".join(sorted(refs)), encoding="utf-8")


if __name__ == "__main__":
    main()
