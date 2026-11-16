"""보고서용 표현 규칙: 우리 작업 안에서만 통하는 이름을 쓰지 않는다 (데이터 분석 방법론 용어 PCA·회귀·MAE·ICC·교차검증 등은 허용).

check(html): 화면에 보이는 글자(태그·링크 주소 제외)에 금지 표현이 있으면 실패. 두 보고서 생성 스크립트가 저장 전에 부른다.
"""
import re

# 내부 표현 → 보고서 표현 (보고서 작성 시 참고)
REPLACE = {
    "홀드아웃": "검증군 (규칙을 정한 뒤 새로 측정한 31명)",
    "개발 세트·동결 54명·파일럿": "개발군 (규칙을 만들 때 사용한 54명)",
    "분석 명단": "PSG 자료 확인이 끝난 환자",
    "계측만(PSG 회신 대기)": "PSG 자료 확인 중인 환자",
    "등록번호 일치 기준 포함": "등록번호로 본인 확인한 환자",
    "M0·M0r·M1~M4": "기본 모델 / 자동 측정값 모델 / +형태 블록 이름",
    "yao4 / jung": "4점 수평면 / 3점 수평면",
    "fh_repair": "귓구멍·눈밑점 대칭 보정",
    "TTA·좌우 반전 추론": "영상을 좌우로 뒤집어 한 번 더 측정해 평균",
    "엑셀 정렬": "엑셀 측정 방식에 맞춘 환산",
    "원래 정의 / 개선 설정 / 기준선": "자동 측정값 / 보정한 자동 측정값",
    "needs_review": "다시 확인이 필요한 환자",
}
BANNED = ["홀드아웃", "개발 세트", "동결", "파일럿", "분석 명단", "계측만", "등록번호 일치 기준", "M0r", "yao4", "jung", "fh_repair",
          "TTA", "좌우 반전 추론", "엑셀 정렬", "원래 정의", "개선 설정", "needs_review", "Se=S", "robust z", "Track",
          "restore_ext", "compare_all", "Phase 1", "M0 ", "M1 ", "M2 ", "M3 ", "M4 ", "qc_pass", ".csv", ".json", "전문가 확인"]


def visible(html):
    s = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    return re.sub(r"<[^>]+>", " ", s)


def check(html, name=""):
    t = visible(html); hits = {b: t.count(b) for b in BANNED if b in t}
    if hits:
        ctx = {b: t[max(0, t.find(b)-30):t.find(b)+40].replace("\n", " ") for b in hits}
        raise AssertionError(f"{name}: 내부 표현이 남아 있음 {hits}\n" + "\n".join(f"  {b}: …{c}…" for b, c in ctx.items()))
