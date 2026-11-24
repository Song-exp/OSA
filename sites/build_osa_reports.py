"""Build the two owner-only OSA Sites from the local report sources."""
import re
import shutil
import sys
from html import escape
from pathlib import Path
from urllib.parse import urlsplit

from lxml import etree, html

ROOT = Path(__file__).resolve().parents[1]
SITES = Path(__file__).resolve().parent

STYLE = """
:root{color-scheme:light;--ink:#18293a;--muted:#475b6d;--line:#cddce4;--accent:#00676a;--paper:#fff}
*{box-sizing:border-box}body{margin:0;background:#f2f7f8;color:var(--ink);font:1rem/1.65 system-ui,sans-serif}
main{max-width:82rem;margin:auto;padding:clamp(1rem,4vw,3rem);background:var(--paper);min-height:100vh}
h1,h2,h3{line-height:1.25;overflow-wrap:anywhere}h1{font-size:clamp(1.9rem,4vw,3rem);margin:.2em 0}
h2{margin-top:2.5em;border-bottom:2px solid var(--line);padding-bottom:.35em}h3{margin-top:1.7em}
.intro,.sub,.note{color:var(--muted)}a{color:var(--accent);text-underline-offset:.2em}
nav{display:flex;flex-wrap:wrap;gap:.5rem;margin:1.5rem 0}nav a{border:1px solid var(--line);border-radius:.4rem;padding:.3rem .65rem;background:#f4f9fa}
.table-scroll{overflow-x:auto;margin:1rem 0}table{width:100%;border-collapse:collapse;min-width:32rem}
th,td{padding:.55rem .7rem;border:1px solid var(--line);text-align:left;vertical-align:top}
th{background:#e5f1f2}tbody tr:nth-child(even){background:#f6fafb}pre{overflow:auto;background:#eff5f7;padding:1rem}
code{overflow-wrap:anywhere}figcaption{color:var(--muted)}figure{margin:1rem 0}
.tool,.mcard,.sum{padding:.8rem 1rem;border:1px solid var(--line);border-radius:.5rem;margin:1rem 0}
label{display:block;font-weight:600}input{font:inherit;padding:.5rem;width:min(100%,28rem);border:1px solid var(--muted);border-radius:.3rem}
[hidden]{display:none!important}a:focus-visible,input:focus-visible{outline:3px solid #007f86;outline-offset:3px}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,23rem),1fr));gap:1rem}
.charts figure{border:1px solid var(--line);padding:1rem;border-radius:.5rem}
.charts img{display:block;width:100%;height:auto;margin-bottom:.6rem}
"""

FILTER = """
const input = document.querySelector('#feature-search');
const rows = [...document.querySelectorAll('table[data-searchable] tbody tr')];
const count = document.querySelector('#match-count');
const filter = () => {
  const terms = input.value.toLocaleLowerCase().trim().split(/\\s+/u);
  let matches = 0;
  for (const row of rows) {
    const text = row.textContent.toLocaleLowerCase();
    row.hidden = !terms.every(term => text.includes(term));
    if (!row.hidden) matches++;
  }
  count.textContent = matches + '개 항목';
};
input.addEventListener('input', filter);
filter();
"""


def wrap_tables(document):
    for table in document.xpath(".//table"):
        wrapper = etree.Element("div", {"class": "table-scroll", "tabindex": "0",
                                        "aria-label": "표", "role": "region"})
        table.addprevious(wrapper)
        wrapper.append(table)


def serialize(element):
    return html.tostring(element, encoding="unicode", method="html")


def make_preprocessing():
    source = ROOT / "Result_py/analysis/all83/preproc/index.html"
    main = html.fromstring(source.read_text(encoding="utf-8")).xpath("//main")[0]
    feature_heading = main.xpath("./h2[@id='features']")[0]
    for element in [feature_heading, *feature_heading.itersiblings()]:
        main.remove(element)
    for element in main.xpath(".//img|.//script|.//style|.//svg|.//iframe|.//button|.//input"):
        element.drop_tree()
    for link in main.xpath(".//a"):
        href = link.get("href", "")
        if href == "#features":
            link.drop_tree()
        elif not (href.startswith("#") or (
            urlsplit(href).scheme == "https" and urlsplit(href).hostname not in {"claude.ai", "chatgpt.com"}
        )):
            link.drop_tag()
    for element in list(main.iterdescendants()):
        if element.tag in {"div", "figure", "a", "p", "span"} and not len(element) and not "".join(element.itertext()).strip():
            element.drop_tree()
    for element in main.iter():
        for name in list(element.attrib):
            if name not in {"id", "class", "href", "rowspan", "colspan", "scope", "lang"}:
                del element.attrib[name]
        for name in ("text", "tail"):
            value = getattr(element, name)
            if value:
                setattr(element, name, re.sub(r"\bP[_ -]?\d+\b", "해당 사례", value))
    main.find("h1").text = "CBCT 전처리 보고서"
    for node in main.xpath(".//p[contains(., '아직 확인하지 않은 것')]"):
        node.text = "설골점 H 영상 검토 결과"
        for child in list(node):
            node.remove(child)
        node.text = ("설골점 H는 85명 영상에서 검토했다. 검출 81명 중 80명은 투영 영상에서 몸통 앞위쪽 윤곽에 놓였고, "
                     "촬영 범위가 제한된 1명은 판정을 보류했다. 4명은 미검출이었다. 수동 점 대비 거리 오차는 평가하지 않았다.")
    notice = etree.Element("p", {"class": "intro"})
    notice.text = "CBCT에서 기준점과 구조를 추출하고 계측값을 만드는 과정. 영상 예시는 로컬 원본에서 확인할 수 있습니다."
    main.insert(1, notice)
    wrap_tables(main)
    return serialize(main)


def make_guide():
    source = ROOT / "Result_py/analysis/all83/preproc/index.html"
    original = html.fromstring(source.read_text(encoding="utf-8")).xpath("//main")[0]
    heading = original.xpath("./h2[@id='features']")[0]
    original_sections = list(heading.itersiblings())
    main = html.fragment_fromstring(
        '<main><h1>CBCT 피처 정의 보고서</h1>'
        '<p class="intro">CBCT 원본 보고서의 피처 표를 발췌했습니다. '
        '피처 이름, 뜻, 도메인적 의미, 계산 과정과 사용 단계를 확인할 수 있습니다.</p>'
        '<p class="note">표의 CT 위치 열에 그림이 있는 항목을 표시했습니다. '
        '환자 CBCT 영상은 로컬 원본 HTML의 8절에서 볼 수 있습니다. '
        '아래 통계 그림은 표본 전체의 적재량과 형상 변화를 보여줍니다.</p></main>'
    )
    nav = etree.Element("nav", {"aria-label": "목차"})
    main.append(nav)
    search = html.fragment_fromstring(
        '<div><label for="feature-search">피처 검색</label>'
        '<input id="feature-search" type="search" placeholder="이름, 뜻, 계산 과정, 사용 단계">'
        ' <span id="match-count" role="status"></span></div>'
    )
    main.append(search)
    for node in original_sections:
        if node.tag == "h3" or (node.tag == "p" and node.get("class") == "note") or node.xpath(".//table"):
            main.append(node)
    for group in main.xpath("./h3"):
        link = etree.SubElement(nav, "a", href="#" + group.get("id"))
        link.text = group.text_content()
    for cell in main.xpath(".//td[.//img]"):
        cell.clear()
        cell.text = "로컬 원본에 CT 위치 그림 있음"
    for table in main.xpath(".//table"):
        table.set("data-searchable", "")
    for element in main.iter():
        for attr in list(element.attrib):
            if attr not in {"id", "class", "href", "rowspan", "colspan", "aria-label", "for",
                            "type", "placeholder", "role", "data-searchable"}:
                del element.attrib[attr]
    assert len(main.xpath(".//table[@data-searchable]//tbody/tr")) == 235
    assert sum(cell.text == "로컬 원본에 CT 위치 그림 있음" for cell in main.xpath(".//td")) == 139

    chart_heading = etree.SubElement(main, "h2", id="stat-figures")
    chart_heading.text = "통계 요약 그림"
    etree.SubElement(nav, "a", href="#stat-figures").text = chart_heading.text
    charts = etree.SubElement(main, "div", {"class": "charts"})
    allowed = {"SIZE_PC1", "AWshape", "F1_load", "F2_load", "F2MM_load", "MAND_rPC_load",
               "MAX_rPC_load", "CB_rPC_load", "EFA_with_Eb", "EFA_without_Eb"}
    figure_dir = SITES / "osa-cbct-feature-guide" / "dist" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stats = original.xpath("./h2[@id='stats']")[0]
    for node in stats.itersiblings():
        if node is heading:
            break
        for image in node.xpath(".//img"):
            src = image.get("src", "")
            filename = Path(src).name
            if src != f"figs/pc/{filename}" or Path(filename).stem not in allowed or not filename.endswith(".png"):
                continue
            shutil.copyfile(source.parent / src, figure_dir / filename)
            fig = etree.SubElement(charts, "figure")
            link = etree.SubElement(fig, "a", href=f"figures/{filename}")
            title = image.get("alt", Path(filename).stem)
            etree.SubElement(link, "img", src=f"figures/{filename}", alt=title, loading="lazy")
            etree.SubElement(fig, "figcaption").text = title
    assert len(charts) == len(allowed)
    wrap_tables(main)
    return serialize(main)


def write_site(slug, title, body, script=""):
    output = SITES / slug / "dist" / "index.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    document = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>{STYLE}</style></head>
<body>{body}{f"<script>{script}</script>" if script else ""}</body></html>
"""
    parsed = html.fromstring(document)
    ids = set(parsed.xpath("//*[@id]/@id"))
    assert all(href[1:] in ids for href in parsed.xpath("//a/@href") if href.startswith("#"))
    assert not parsed.xpath("//iframe|//svg|//*[@style]")
    for src in parsed.xpath("//*[@src]/@src"):
        assert slug == "osa-cbct-feature-guide" and re.fullmatch(r"figures/[A-Za-z0-9_]+\.png", src)
        assert (output.parent / src).is_file()
    assert not re.search(r"\bP[_ -]?\d+\b|file://|https://claude\.ai", document)
    output.write_text(document, encoding="utf-8")
    print(slug, len(parsed.xpath("//table")), "tables", len(parsed.xpath("//table[@data-searchable]//tbody/tr")), "search rows")


if __name__ == "__main__":
    if "--guide-only" not in sys.argv:
        write_site("osa-cbct-preprocessing", "CBCT 전처리 보고서", make_preprocessing())
    write_site("osa-cbct-feature-guide", "CBCT 피처 정의 보고서", make_guide(), FILTER)
