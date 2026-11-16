"""Add reversible, accessible definitions to report text, preserving SVGs."""
from html import escape
from html.parser import HTMLParser
from pathlib import Path
import json
import re

TERMS = {
    "N": "Nasion(비근점). 전두골과 코뼈가 만나는 봉합의 가장 앞쪽 정중점으로, 코뿌리 부근의 골격 기준점입니다.",
    "Na": "Nasion의 다른 표기입니다. 이 보고서의 N과 같은 비근점입니다.",
    "A": "A point(상악 A점). 전비극과 위 앞니 치조부 사이 상악 앞면에서 가장 오목한 정중점입니다.",
    "B": "B point(하악 B점). 아래 앞니 치조부와 턱끝 사이 하악 앞면에서 가장 오목한 정중점입니다.",
    "S": "Sella(터키안장 중심점). 두개저의 터키안장 중심에 두는 기준점입니다.",
    "Or": "Orbitale(안와점). 눈확 아래쪽 뼈 테두리의 가장 낮은 점이며 FH 기준면을 만드는 데 사용합니다.",
    "Po": "Porion(이공점). 뼈 외이도 윤곽의 가장 위쪽 기준점이며 FH 기준면을 만드는 데 사용합니다.",
    "Go": "Gonion(하악각점). 아래턱뼈 뒤쪽이 꺾이는 하악각 부위의 기준점입니다. 이 보고서의 하악평면은 좌우 Go의 평균점과 Me를 잇습니다.",
    "Ba": "Basion(대후두공 앞쪽 기준점). 대후두공 앞 경계의 가장 아래 정중점으로 정중시상면과 비강 상한 설정에 사용합니다.",
    "Pog": "Pogonion(골성 턱끝 앞점). 아래턱 정중부 뼈에서 가장 앞쪽으로 돌출된 점입니다.",
    "Me": "Menton(턱끝 아래점). 아래턱뼈 정중부에서 가장 아래쪽 점입니다.",
    "ANS": "Anterior nasal spine(전비극). 상악뼈 앞쪽의 코바닥 돌출점으로 구개평면을 만드는 기준점입니다.",
    "PNS": "Posterior nasal spine(후비극). 뼈 입천장 뒤쪽 끝의 정중점으로 구개평면·기도 경계·PAL 계산에 사용합니다.",
    "Rh": "Rhinion(비골 하단점). 코뼈 아래쪽 끝의 기준점으로 비강 앞쪽 경계에 사용합니다.",
    "Eb": "후두개곡의 가장 아래 정중점. 현재 자동 계측은 이 정의를 사용하며, 일부 환자는 설골 위치로 근사했습니다. PAL과 인두 기도 아래 경계에 사용합니다.",
    "Se": "원 논문의 기도 경계 기준점 표기. 현재는 Sella(S)로 가정했지만 원 측정자의 정의 확인이 필요합니다.",
    "RPo": "Right Porion. 환자의 오른쪽 외이도 위쪽 기준점입니다. R은 환자 기준 오른쪽입니다.",
    "LPo": "Left Porion. 환자의 왼쪽 외이도 위쪽 기준점입니다. L은 환자 기준 왼쪽입니다.",
    "ROr": "Right Orbitale. 환자의 오른쪽 눈확 아래 테두리에서 가장 낮은 점입니다.",
    "LOr": "Left Orbitale. 환자의 왼쪽 눈확 아래 테두리에서 가장 낮은 점입니다.",
    "RGo": "Right Gonion. 환자의 오른쪽 하악각 기준점입니다.",
    "LGo": "Left Gonion. 환자의 왼쪽 하악각 기준점입니다.",
    "Sup": "과두 최상점. 현재 과두 각도 계측에서는 하악 분할의 과두 영역에서 가장 높은 점을 찾습니다.",
    "Mid": "과두 경부 중점. 현재 측면각 계산에서 Sup 아래 경부의 좁은 단면 중점을 뜻합니다.",
    "FH": "Frankfort horizontal plane(프랑크푸르트 수평면). 머리의 기준 수평면이며 현재는 RPo·LPo·ROr 세 점으로 만듭니다.",
    "MSP": "Midsagittal plane(정중시상면). 머리를 좌우로 나누는 기준면입니다. 골격 계측에서는 N·Ba를 지나 FH에 수직인 면을 사용합니다.",
    "MP": "Mandibular plane(하악평면). 이 보고서의 주 계산은 좌우 Go 평균점과 Me를 잇는 선을 정중면에 투영합니다.",
    "ANB": "상악 A점과 하악 B점의 앞뒤 관계를 나타내는 각도입니다. N에서 A와 B로 그은 두 선 사이 각도를 정중시상면에 투영해 계산합니다(°).",
    "Nperp": "N을 지나 FH 기준면에 수직인 기준선입니다. A점·Pog점이 이 선보다 얼마나 앞뒤에 있는지 잽니다.",
    "Nperp-A": "N 수직 기준선에서 상악 A점까지의 앞뒤 거리입니다. 현재 계산은 앞쪽을 양수로 표시합니다(mm).",
    "Nperp-Pog": "N 수직 기준선에서 골성 턱끝 Pog까지의 앞뒤 거리입니다. 현재 계산은 앞쪽을 양수로 표시합니다(mm).",
    "MP-FH": "하악평면(MP)과 머리 수평 기준면(FH) 사이의 각도입니다. 아래턱 평면의 기울기를 나타냅니다(°).",
    "MP–FH": "하악평면(MP)과 머리 수평 기준면(FH) 사이의 각도입니다. 아래턱 평면의 기울기를 나타냅니다(°).",
    "NAV": "비강 기도 부피. 정해진 비강 경계 안에서 비강 라벨에 포함된 공기의 부피입니다(mm³). 부비동 포함 범위와 공기 임계값 정의 확인이 남아 있습니다.",
    "PAV": "인두 기도 부피. 정해진 위·아래 경계 사이 인두 기도 공기의 전체 부피입니다(mm³).",
    "PMCA": "인두 기도의 최소 단면적. 현재는 기도 중심선에 수직인 단면 중 가장 좁은 넓이를 사용합니다(mm²). 원 측정의 단면 방향·범위는 확인이 필요합니다.",
    "PAL": "인두 기도 길이. 현재는 후비극 PNS에서 후두개곡 기준점 Eb까지의 직선 거리입니다(mm).",
    "NCWR": "비강 폭 비율. 비강 최대폭(NCW)을 상악 기저폭(MxBW)으로 나누고 100을 곱한 값입니다(%). 부피가 아닌 폭의 비율입니다.",
    "NCW": "비강 최대 좌우 폭. 현재 NCWR 계산에서는 제1대구치 단면의 비강 라벨 최대폭을 사용하며 비갑개 등 조직도 포함합니다(mm).",
    "MxBW": "상악 기저폭. NCWR의 분모인 위턱뼈 기저부 폭입니다. 원 측정의 높이·기준점이 확인되지 않았습니다(mm).",
    "PHI": "구개 높이 지수. 구개 높이(PH)를 상악 폭(MxOW)으로 나누고 100을 곱합니다. 현재 폭은 좌우 제1대구치 저항중심 사이 거리입니다(%).",
    "PH": "구개 높이. 현재는 구개 천장과 교합 높이 사이의 거리입니다(mm).",
    "MxOW": "PHI의 분모인 상악 교합폭입니다. 현재 선택 규칙에서는 좌우 제1대구치 저항중심 사이 폭을 사용합니다(mm).",
    "TDI": "상·하악의 횡적 폭 차이. 상악 폭(MxAW)에서 하악 폭(MnAW)을 뺍니다. 현재 선택 규칙은 제1대구치 저항중심 사이 폭입니다(mm).",
    "MxAW": "상악의 횡적 폭. TDI 계산에 쓰며, 현재는 좌우 상악 제1대구치 저항중심 사이 폭입니다(mm).",
    "MnAW": "하악의 횡적 폭. TDI 계산에 쓰며, 현재는 좌우 하악 제1대구치 저항중심 사이 폭입니다(mm).",
    "UR6": "환자의 오른쪽 위 제1대구치입니다. U는 위턱, R은 오른쪽, 6은 제1대구치를 뜻합니다.",
    "UL6": "환자의 왼쪽 위 제1대구치입니다. U는 위턱, L은 왼쪽, 6은 제1대구치를 뜻합니다.",
    "LR6": "환자의 오른쪽 아래 제1대구치입니다. L(첫 글자)은 아래턱, R은 오른쪽, 6은 제1대구치를 뜻합니다.",
    "LL6": "환자의 왼쪽 아래 제1대구치입니다. 첫 L은 아래턱, 둘째 L은 왼쪽, 6은 제1대구치를 뜻합니다.",
    "UR6MB": "환자의 오른쪽 위 제1대구치 근심협측 교두 기준점입니다. 현재 교합평면을 만드는 데 사용합니다.",
    "UL6MB": "환자의 왼쪽 위 제1대구치 근심협측 교두 기준점입니다. 현재 교합평면을 만드는 데 사용합니다.",
    "CBCT": "Cone-beam computed tomography(콘빔 CT). 치아·턱·두개안면 구조를 보는 3차원 영상입니다.",
    "TMJ": "Temporomandibular joint(턱관절). TMJ 통증은 임상 소견으로, CBCT 모양만으로 통증 여부를 정하지 않습니다.",
    "QC": "Quality control(품질 점검). 이 보고서의 통과 표시는 정해진 자동 점검 기준을 만족했다는 뜻이며 전문가 승인과는 다릅니다.",
    "FOV": "Field of view(촬영 범위). 영상에 담긴 공간의 범위이며 범위 밖 구조는 현재 영상에서 측정할 수 없습니다.",
    "HU": "영상 밝기·밀도 값의 단위 표기입니다. 여기서는 공기를 고르는 임계값에 사용하며 CBCT 값의 보정·장비 차이에 주의해야 합니다.",
    "MAE": "평균 절대오차. 환자별 |복원값 − 엑셀값|을 평균한 값이며 작을수록 두 계측값이 가깝습니다.",
    "CCC": "일치상관계수. 상관뿐 아니라 치우침과 산포 차이도 반영해 두 값의 일치를 나타내며 1에 가까울수록 일치합니다.",
    "AUC": "연속 측정값이 엑셀의 0/1 두 범주를 얼마나 구분하는지 보는 지표입니다. 0.5는 무작위 수준, 1은 완전 구분이며 같은 판정 비율과는 다릅니다.",
    "κ": "카파 계수. 우연한 일치를 보정한 0/1 판정 일치도입니다. 1은 완전 일치, 0은 우연 수준의 일치를 뜻합니다.",
    "ρ": "스피어만 순위상관계수. 값의 순서가 함께 변하는 정도이며 값 자체가 같다는 뜻은 아닙니다.",
    "r": "피어슨 상관계수. 두 값이 선형적으로 함께 변하는 정도입니다. 상관이 높아도 일정한 과대·과소 계측은 남을 수 있습니다.",
    "Type": "엑셀의 분류 변수 이름입니다. 이 작업에서 CBCT로 직접 재측정하는 값이 아니며 원 분류 규칙 확인이 필요합니다.",
    "Group": "엑셀의 집단 분류 변수 이름입니다. 이 작업에서 CBCT로 직접 재측정하는 값이 아니며 원 분류 규칙 확인이 필요합니다.",
}

OVERVIEW = [
    ("NAV", "비강 기도 부피", "계산되지만 엑셀보다 전반적으로 작아 경계·공기 임계값 확인이 남음"),
    ("NCWR", "비강 폭 / 상악 기저폭 ×100", "비강 폭은 재현되지만 분모·측정 높이 정의 확인이 남음"),
    ("PMCA", "인두 기도의 가장 좁은 단면 넓이", "상관은 높지만 전반적으로 크게 나와 단면 방향·범위 확인이 남음"),
    ("교합평면–FH", "치아 교합평면과 머리 수평 기준면 사이 각도", "각도를 계산했지만 원 측정의 평면·점 정의가 충분히 재현되지 않음"),
    ("비강저 경사 (0/1)", "좌우 코바닥 기울어짐의 이진 판정", "기울기는 계산했지만 엑셀의 판정과 일치가 약함"),
    ("비중격 편위 (0/1)", "코 안 중앙 칸막이가 기준면에서 벗어났는지 판정", "현재 기준에서 과다 판정되며 원 기준면·범위 확인이 필요함"),
    ("턱끝 편위 (0/1)", "턱끝이 정중 기준면에서 벗어났는지 판정", "원래 거리 기준을 몰라 판정 일치가 약함"),
    ("과두 축상각", "위에서 본 턱관절 과두 장축의 방향 각도", "좌우 각각 계산했지만 기준선 차이가 의심됨"),
    ("과두 측면각", "옆에서 본 과두 머리·경부 방향 각도", "원 측정 단면·방향이 재현되지 않아 현재 결과의 일치가 낮음"),
    ("관절 공간", "턱관절 과두와 관절와 사이 전·상·후 간격", "현재 자동 계측은 미착수이며 측정 단면 정의 확인도 필요함"),
    ("비폐색·상악동염·TMJ 통증", "임상 증상·진단 정보", "현재 영상 계측만으로 원 임상 기록을 복원하지 않음"),
    ("Type·Group", "엑셀의 분류·집단 변수", "직접 측정값이 아니며 분류 규칙 확인이 필요함"),
]

CSS = '''<style id="cbct-glossary-style">
abbr.cbct-term{font:inherit;color:inherit;text-decoration:underline dotted var(--muted);text-underline-offset:3px;cursor:help}
abbr.cbct-term:focus-visible{outline:2px solid var(--accent);outline-offset:3px;border-radius:2px}
#cbct-term-tooltip{position:fixed;z-index:1000;width:560px;max-width:calc(100vw - 24px);max-height:calc(100vh - 24px);overflow:auto;padding:12px 15px;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:9px;box-shadow:0 5px 24px #0004;font-size:14px;line-height:1.6;white-space:normal}
#cbct-term-tooltip[hidden]{display:none}
#cbct-term-tooltip .term-heading{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:7px}
#cbct-term-tooltip button{font:inherit;cursor:pointer;color:var(--fg);background:var(--chip);border:1px solid var(--line);border-radius:5px;padding:3px 9px}
#cbct-term-tooltip img{display:block;width:100%;height:auto;max-height:46vh;object-fit:contain;background:white;border:1px solid var(--line);margin:8px 0}
#cbct-term-tooltip .term-caption{font-size:12px;color:var(--muted);line-height:1.5}
#cbct-term-tooltip .term-hint{font-size:12px;color:var(--muted);margin:7px 0 0}
.glossary-example-link{font-size:.85rem}
</style>'''

JS = '''<script id="cbct-glossary-script">
document.addEventListener('DOMContentLoaded', () => {
  const tip = document.getElementById('cbct-term-tooltip');
  document.querySelectorAll('abbr.cbct-term').forEach(term => term.removeAttribute('title'));
  const name=document.getElementById('cbct-term-name'), text=document.getElementById('cbct-term-text');
  const image=document.getElementById('cbct-term-image'), imageLink=document.getElementById('cbct-term-image-link');
  const caption=document.getElementById('cbct-term-caption'), pin=document.getElementById('cbct-term-pin');
  const close=document.getElementById('cbct-term-close');
  let active=null, pinned=false, timer=null, suppressFocus=false;
  const cancel=()=>{clearTimeout(timer);timer=null;};
  const place=()=>{
    if(!active || tip.hidden)return;
    const box=active.getBoundingClientRect(), width=tip.offsetWidth, height=tip.offsetHeight;
    tip.style.left=Math.min(Math.max(12,box.left),Math.max(12,innerWidth-width-12))+'px';
    tip.style.top=(box.bottom+8+height<=innerHeight ? box.bottom+8 : Math.max(12,box.top-height-8))+'px';
  };
  const hide=(restore=false)=>{
    cancel();const previous=active;
    if(active)active.removeAttribute('aria-describedby');
    active=null;pinned=false;tip.hidden=true;tip.setAttribute('role','tooltip');
    if(restore && previous){suppressFocus=true;previous.focus({preventScroll:true});suppressFocus=false;}
  };
  const show=(term,force=false)=>{
    if(document.documentElement.dataset.reportEditing==='true')return;
    if(pinned && !force)return;
    cancel();
    if(active && active!==term) active.removeAttribute('aria-describedby');
    active=term;pinned=false;tip.setAttribute('role','tooltip');pin.textContent='고정';
    name.textContent=term.dataset.term;text.textContent=term.dataset.definition;
    const src=term.dataset.exampleSrc;
    imageLink.hidden=!src;caption.hidden=!src;
    if(src){image.src=src;image.alt=term.dataset.term+' 실제 데이터 예시';imageLink.href=src;caption.textContent=term.dataset.exampleCaption;}
    else{image.removeAttribute('src');imageLink.removeAttribute('href');caption.textContent='';}
    tip.hidden=false;
    term.setAttribute('aria-describedby',tip.id);
    place();
  };
  const fix=()=>{if(!active)return;pinned=true;tip.setAttribute('role','dialog');pin.textContent='고정됨';};
  const later=()=>{if(!pinned){cancel();timer=setTimeout(()=>hide(),220);}};
  image.addEventListener('load',place);
  pin.addEventListener('click',fix);close.addEventListener('click',()=>hide(true));
  tip.addEventListener('mouseenter',cancel);tip.addEventListener('mouseleave',later);
  document.addEventListener('mouseover',e=>{const term=e.target.closest&&e.target.closest('abbr.cbct-term');if(term)show(term);});
  document.addEventListener('mouseout',e=>{
    if(!active)return;
    const term=e.target.closest&&e.target.closest('abbr.cbct-term');
    if((term || tip.contains(e.target)) && !tip.contains(e.relatedTarget) && !active.contains(e.relatedTarget))later();
  });
  document.addEventListener('focusin',e=>{if(suppressFocus)return;const term=e.target.closest&&e.target.closest('abbr.cbct-term');if(term)show(term);else if(!tip.contains(e.target) && !pinned)hide();});
  document.addEventListener('click',e=>{
    if(document.documentElement.dataset.reportEditing==='true')return;
    const term=e.target.closest&&e.target.closest('abbr.cbct-term');
    if(term){show(term,true);fix();}else if(!tip.contains(e.target))hide();
  });
  document.addEventListener('keydown',e=>{
    if(document.documentElement.dataset.reportEditing==='true')return;
    if(e.key==='Escape'){hide(true);return;}
    const term=e.target.closest&&e.target.closest('abbr.cbct-term');
    if(term && (e.key==='Enter' || e.key===' ')){e.preventDefault();show(term,true);fix();close.focus();}
  });
  document.addEventListener('scroll',e=>{if(!pinned && !tip.contains(e.target))hide();},true);
  window.addEventListener('resize',place);
});
</script>'''


class Decorator(HTMLParser):
    VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}
    SKIP = {'head', 'script', 'style', 'svg', 'code', 'pre', 'abbr', 'textarea', 'iframe'}

    def __init__(self, source, examples=None):
        super().__init__(convert_charrefs=False)
        self.source, self.out, self.stack = source, [], []
        self.examples = load_examples() if examples is None else examples
        self.offsets = [0]
        for match in re.finditer('\n', source):
            self.offsets.append(match.end())
        self.pattern = re.compile(r'(?<![A-Za-z0-9_])(?:'+"|".join(re.escape(k) for k in sorted(TERMS, key=len, reverse=True))+r')(?![A-Za-z0-9_]|등급)')

    def raw(self):
        row, col = self.getpos()
        position = self.offsets[row-1]+col
        end = self.source.index('>', position)+1
        return self.source[position:end]

    def handle_starttag(self, tag, attrs):
        self.out.append(self.get_starttag_text())
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.out.append(self.get_starttag_text())

    def handle_endtag(self, tag):
        self.out.append(self.raw())
        if tag in self.stack:
            index = len(self.stack)-1-self.stack[::-1].index(tag)
            del self.stack[index:]

    def handle_data(self, data):
        if any(tag in self.SKIP for tag in self.stack):
            self.out.append(data)
        else:
            def replace(match):
                term = match[0]
                definition = escape(TERMS[term], quote=True)
                example=self.examples.get(term)
                extra=(' data-example-src="'+escape(example['src'],quote=True)+'" data-example-caption="'+escape(example['caption'],quote=True)+'"') if example else ''
                return f'<abbr class="cbct-term" tabindex="0" data-term="{escape(term,quote=True)}" data-definition="{definition}" title="{definition}"{extra}>{term}</abbr>'
            self.out.append(self.pattern.sub(replace, data))

    def handle_entityref(self, name):
        self.out.append('&'+name+';')

    def handle_charref(self, name):
        self.out.append('&#'+name+';')

    def handle_comment(self, data):
        self.out.append('<!--'+data+'-->')

    def handle_decl(self, decl):
        self.out.append(self.raw())


def load_examples():
    path=Path(__file__).resolve().parents[1]/'Result_py/CBCT/patient_pipeline/v2/report/glossary_images/manifest.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def overview():
    rows = ''.join('<tr><td>'+escape(name)+'</td><td>'+escape(meaning)+'</td><td>'+escape(state)+'</td></tr>' for name, meaning, state in OVERVIEW)
    examples=load_examples()
    glossary_rows=[]
    for name in ('N', 'A', 'B', 'Or', 'Go', 'Po', 'S', 'Ba', 'Pog', 'Me', 'ANS', 'PNS', 'Rh', 'Eb', 'FH', 'MSP', 'ANB', 'Nperp-A', 'Nperp-Pog', 'MP-FH', 'TDI', 'PHI', 'PAL', 'PAV', 'NAV', 'NCWR', 'PMCA', 'MAE', 'CCC', 'r', 'ρ', 'κ', 'AUC'):
        example=examples.get(name)
        image_link=('<a class="glossary-example-link" href="'+escape(example['src'],quote=True)+'" target="_blank" rel="noopener">'+escape(example['patient'])+' 실제 예시 열기</a>') if example else '—'
        glossary_rows.append('<tr><td>'+escape(name)+'</td><td>'+escape(TERMS[name])+'</td><td>'+image_link+'</td></tr>')
    glossary_rows=''.join(glossary_rows)
    return '''<!-- CBCT_GLOSSARY_OVERVIEW_START -->
<p class="note"><strong>약자 설명:</strong> 점선 밑줄의 약자에 마우스를 올리거나 Tab으로 선택하면 뜻과 관련된 실제 데이터 예시를 함께 볼 수 있습니다. 클릭·Enter로 설명 창을 고정하고 그림을 누르면 원본 크기로 열립니다. Escape로 닫습니다. 표시한 자동 점·측정선은 설명용 예시이며 전문가가 확정한 정답 좌표가 아닙니다. 정의가 미확정인 용어는 현재 가정을 표시했습니다.</p>
<details id="metric-meaning"><summary>부분 복원·재현 약함·미착수 항목은 무엇인가요?</summary><p>이 표는 분석에 포함된 54명에서 지표별 재현 상태를 정리한 것입니다. 부분 복원은 값이 계산되고 연관성이 있지만 크기 차이가 남은 상태, 재현 약함은 엑셀과의 값 또는 0/1 판정 일치가 낮은 상태입니다. 임상 허용오차 기준이나 환자 제외 판정이 아닙니다.</p><div class="tw"><table><tr><th>항목</th><th>무엇을 나타내나</th><th>현재 의미</th></tr>'''+rows+'''</table></div></details>
<details id="landmark-glossary"><summary>랜드마크·지표·통계 약자 사전과 실제 예시 펼치기</summary><div class="tw"><table><tr><th>약자</th><th>이 보고서에서 사용하는 뜻</th><th>실제 데이터 예시</th></tr>'''+glossary_rows+'''</table></div></details>
<!-- CBCT_GLOSSARY_OVERVIEW_END -->'''


def apply_glossary(html):
    # Strip only our own wrappers before redecoration, so report builders can
    # still find their original headings and markup on subsequent runs.
    has_editor='id="report-editor-toolbar"' in html
    if has_editor:
        from cbct_report_editor import strip_editor
        html=strip_editor(html)
    html = strip_glossary(html)
    anchor = '<div class="cohort-tabs"'
    assert html.count(anchor) == 1
    html = html.replace(anchor, overview()+'\n'+anchor, 1)
    parser = Decorator(html)
    parser.feed(html)
    parser.close()
    html = ''.join(parser.out)
    html = html.replace('</head>', CSS+'\n</head>', 1)
    popover='''<!-- CBCT_TERM_POPOVER_START -->
<div id="cbct-term-tooltip" role="tooltip" aria-labelledby="cbct-term-name" hidden>
<div class="term-heading"><strong id="cbct-term-name"></strong><span><button id="cbct-term-pin" type="button">고정</button> <button id="cbct-term-close" type="button" aria-label="용어 설명 닫기">닫기</button></span></div>
<div id="cbct-term-text"></div>
<a id="cbct-term-image-link" target="_blank" rel="noopener" hidden><img id="cbct-term-image" alt="실제 CBCT 예시"></a>
<p class="term-caption" id="cbct-term-caption" hidden></p>
<p class="term-hint">클릭·Enter로 고정 · 그림 클릭으로 확대 · Escape로 닫기</p></div>
<!-- CBCT_TERM_POPOVER_END -->'''
    html = html.replace('</body>', popover+'\n'+JS+'\n</body>', 1)
    if has_editor:
        from cbct_report_editor import apply_editor
        html=apply_editor(html)
    return html


def strip_glossary(html):
    html = re.sub(r'<abbr\b[^>]*class="cbct-term"[^>]*>(.*?)</abbr>', r'\1', html, flags=re.S)
    html = re.sub(r'\s*<!-- CBCT_GLOSSARY_OVERVIEW_START -->.*?<!-- CBCT_GLOSSARY_OVERVIEW_END -->\s*', '\n', html, flags=re.S)
    html = re.sub(r'<style id="cbct-glossary-style">.*?</style>\s*', '', html, flags=re.S)
    html = re.sub(r'<script id="cbct-glossary-script">.*?</script>\s*', '', html, flags=re.S)
    html = re.sub(r'<div id="cbct-term-tooltip"[^>]*></div>\s*', '', html)
    html = re.sub(r'<!-- CBCT_TERM_POPOVER_START -->.*?<!-- CBCT_TERM_POPOVER_END -->\s*', '', html, flags=re.S)
    return html


def main():
    from cbct_report_cohorts import REPORT
    original = REPORT.read_text(encoding='utf-8')
    output = apply_glossary(original)
    svg = re.compile(r'<svg\b.*?</svg>', re.S)
    assert svg.findall(original) == svg.findall(output)
    assert output == apply_glossary(output), 'Glossary decoration must be idempotent'
    REPORT.write_text(output, encoding='utf-8')
    print('Glossary added:', len(TERMS), 'definitions;', output.count('class="cbct-term"'), 'interactive terms; SVGs unchanged')


if __name__ == '__main__':
    main()
