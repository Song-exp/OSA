"""Step1-3_회귀_보고서.html → Claude 아티팩트용 페이지(index.html: 문서 뼈대 제거, 테마 토큰, 그림 크게 보기, 편집기 제거) + 그림 목록(files.json).
Result_py/analysis/all83/artifact/ (git 제외)에 index.html과 figs/를 만든다. 게시는 Claude Code의 Artifact 도구로 한다.
"""
import re, json, shutil
from pathlib import Path
from report_editor import strip

SRC = Path(__file__).resolve().parents[1]/"Result_py/analysis/all83"
OUT = SRC/"artifact"
OUT.mkdir(exist_ok=True)
h = strip((SRC/"Step1-3_회귀_보고서.html").read_text(encoding="utf-8"))
styles = "\n".join(re.findall(r"<style[^>]*>(.*?)</style>", h, re.S))
body = re.search(r"<body[^>]*>(.*)</body>", h, re.S).group(1)

# 표에 넣은 고정색 → 테마 토큰
MAP = {"#b8c2cc": "--grid", "#e6efff": "--k1", "#f1e9ff": "--k2", "#e3f7ee": "--k3", "#fff1dc": "--k4", "#fde8c8": "--hi", "#eefaf0": "--g2",
       "#fff4ec": "--g3", "#f3f6fa": "--hov", "#e3edff": "--on", "#fff3cd": "--focus", "#fff3c4": "--rs"}
LIGHT = {k: v for v, k in MAP.items()}
DARK = {"--grid": "#4a4f55", "--k1": "#1d2a40", "--k2": "#2a2140", "--k3": "#18302a", "--k4": "#3a2b16", "--hi": "#4d3618", "--g2": "#1b2e22",
        "--g3": "#34261a", "--hov": "#2a2d31", "--on": "#1f2c46", "--focus": "#4a4016", "--rs": "#3d3410"}
for lit, tok in MAP.items(): styles = re.sub(re.escape(lit), f"var({tok})", styles, flags=re.I)
dark_base = re.search(r"@media \(prefers-color-scheme:dark\)\{:root:not\(\[data-theme=\"light\"\]\)\{([^}]*)\}\}", styles).group(1)
tok = (":root{" + ";".join(f"{k}:{v}" for k, v in LIGHT.items()) + "}"
       "@media (prefers-color-scheme:dark){:root:not([data-theme=\"light\"]){" + ";".join(f"{k}:{v}" for k, v in DARK.items()) + ";color-scheme:dark}}"
       ":root[data-theme=\"dark\"]{" + dark_base + ";" + ";".join(f"{k}:{v}" for k, v in DARK.items()) + ";color-scheme:dark}")
extra = """main{padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.tw{overflow-x:auto;max-width:100%}td,th{font-variant-numeric:tabular-nums}
#lb{position:fixed;inset:0;z-index:100;background:rgba(0,0,0,.82);display:flex;align-items:center;justify-content:center;padding:max(16px,env(safe-area-inset-top,0px)) max(16px,env(safe-area-inset-right,0px)) max(16px,env(safe-area-inset-bottom,0px)) max(16px,env(safe-area-inset-left,0px));cursor:zoom-out}
#lb img{max-width:100%;max-height:100%;background:#fff;border-radius:4px}
#lb button{position:absolute;top:max(12px,env(safe-area-inset-top,0px));right:max(12px,env(safe-area-inset-right,0px));font:inherit;padding:6px 12px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
a:focus-visible,button:focus-visible,select:focus-visible{outline:2px solid var(--accent);outline-offset:2px}"""
lightbox = """<div id="lb" hidden role="dialog" aria-label="그림 크게 보기"><button type="button" id="lbx">닫기</button><img id="lbi" alt=""></div>
<script>document.addEventListener('click',e=>{const a=e.target.closest('a[href^="figs/"]');const lb=document.getElementById('lb');
if(a){e.preventDefault();document.getElementById('lbi').src=a.getAttribute('href');lb.hidden=false;return;}if(e.target.closest('#lb'))lb.hidden=true;});
document.addEventListener('keydown',e=>{if(e.key==='Escape')document.getElementById('lb').hidden=true;});</script>"""
page = ('<title>OSA 골격·기도 회귀</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
        f"<style>{styles}\n{tok}\n{extra}</style>\n{body}\n{lightbox}\n")
(OUT/"index.html").write_text(page, encoding="utf-8")
refs = sorted(set(re.findall(r'(?:src|href)="(figs/[^"]+)"', page)))
files = {r: str(SRC/r) for r in refs if (SRC/r).exists()}
missing = [r for r in refs if not (SRC/r).exists()]
(OUT/"files.json").write_text(json.dumps(files, ensure_ascii=False, indent=0), encoding="utf-8")
print("페이지", round(len(page)/1e6, 2), "MB | 그림", len(files), "| 없음", missing, "| 남은 고정색", sorted(set(re.findall(r"#[0-9a-fA-F]{6}\b", re.sub(r":root[^{]*\{[^}]*\}|@media[^{]*\{[^}]*\{[^}]*\}\}", "", styles))))[:20])
