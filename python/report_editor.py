"""보고서 HTML에 붙이는 편집기: 글 수정·글 추가·이미지 추가·삭제·임시 저장·수정본 HTML 저장.

- 추가한 이미지는 HTML 안에 data URL로 포함(긴 변 1600 px로 줄여 JPEG/PNG)해 파일을 옮겨도 보인다.
- 원래 그림·링크는 저장할 때 이 PC의 절대 경로로 바꿔, 수정본을 다른 폴더에 저장해도 연결된다.
- 브라우저 임시 저장은 <main> 전체를 원본 지문과 함께 localStorage에 둔다(용량을 넘으면 알리고 파일 저장을 권함).
apply(html): 보고서 HTML에 편집기를 붙인다(여러 번 붙여도 하나만 남음).
"""
import re

STYLE = """<style id="rpt-editor-style">
#rpt-bar{position:sticky;top:0;z-index:50;background:var(--card,#fff);border-bottom:1px solid var(--line,#ddd);padding:8px 16px;box-shadow:0 2px 8px #0001;font:14px/1.5 "Malgun Gothic",system-ui,sans-serif}
#rpt-bar .acts{display:flex;flex-wrap:wrap;gap:6px;align-items:center;max-width:1180px;margin:0 auto}
#rpt-bar button{font:inherit;padding:5px 10px;border:1px solid var(--line,#ccc);border-radius:5px;background:var(--bg,#f7f7f5);color:var(--fg,#111);cursor:pointer}
#rpt-bar button:disabled{opacity:.45;cursor:default}#rpt-bar button.on{background:var(--fg,#111);color:var(--bg,#fff)}#rpt-bar button.save{background:var(--accent,#2a78d6);color:#fff;border-color:var(--accent,#2a78d6)}
#rpt-status{max-width:1180px;margin:4px auto 0;font-size:12px;color:var(--muted,#666)}
html[data-editing="true"] [data-ed="1"]{outline:1px dashed var(--accent,#2a78d6);outline-offset:2px;cursor:text}
html[data-editing="true"] [data-ed="1"]:focus{outline:2px solid var(--accent,#2a78d6)}
.user-block{border-left:3px solid var(--accent,#2a78d6);padding:4px 10px;margin:10px 0}
figure.user-fig{margin:12px 0}figure.user-fig img{max-width:100%;border:1px solid var(--line,#ddd)}figure.user-fig figcaption{font-size:.88rem;color:var(--muted,#666)}
html[data-editing="true"] .user-sel{box-shadow:0 0 0 3px #eb683488}
@media print{#rpt-bar{display:none}[data-ed]{outline:none!important}}
</style>"""

BAR = """<!-- RPT_EDITOR_BAR_START -->
<aside id="rpt-bar" aria-label="보고서 편집 도구"><div class="acts">
<button id="rpt-toggle" type="button">편집 모드 켜기</button>
<button id="rpt-add-text" type="button" disabled>글 추가</button>
<button id="rpt-add-head" type="button" disabled>소제목 추가</button>
<button id="rpt-add-img" type="button" disabled>이미지 추가</button>
<button id="rpt-del" type="button" disabled>추가한 항목 삭제</button>
<button id="rpt-undo" type="button" disabled>실행 취소</button>
<button id="rpt-redo" type="button" disabled>다시 실행</button>
<button id="rpt-save" class="save" type="button">수정본 HTML 저장</button>
<button id="rpt-reset" type="button">처음 내용으로</button>
<input id="rpt-file" type="file" accept="image/*" hidden></div>
<p id="rpt-status" role="status" aria-live="polite">편집 모드를 켜면 글을 고치고, 커서가 있는 문단 아래에 글·소제목·이미지를 넣을 수 있습니다. 원본 파일은 바뀌지 않습니다.</p></aside>
<!-- RPT_EDITOR_BAR_END -->"""

SCRIPT = r"""<script id="rpt-editor-script">
document.addEventListener('DOMContentLoaded',()=>{
  const main=document.querySelector('main'); if(!main)return;
  const $=id=>document.getElementById(id), status=$('rpt-status'), toggle=$('rpt-toggle'), file=$('rpt-file');
  const tools=['rpt-add-text','rpt-add-head','rpt-add-img','rpt-del','rpt-undo','rpt-redo'].map($);
  const SEL='h1,h2,h3,h4,p,li,td,th,figcaption,summary,dt,dd';
  const say=t=>status.textContent=t;
  const hash=s=>{let h=2166136261;for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619);}return (h>>>0).toString(16);};
  const original=main.innerHTML, key='osa-report-draft:'+hash(location.pathname+'|'+original.length+'|'+hash(original));
  let editing=false, last=null, timer=null, dirty=false, selected=null;
  const blocks=()=>[...main.querySelectorAll(SEL)].filter(el=>!el.querySelector(SEL+',img,svg,table,iframe,input,button')&&!el.closest('nav'));
  function setEd(on){blocks().forEach(el=>{if(on){el.contentEditable='true';el.dataset.ed='1';el.spellcheck=false;}else{el.removeAttribute('contenteditable');delete el.dataset.ed;el.removeAttribute('spellcheck');}});}
  function mode(on){editing=on;document.documentElement.dataset.editing=String(on);toggle.textContent=on?'편집 완료':'편집 모드 켜기';toggle.classList.toggle('on',on);
    tools.forEach(b=>b.disabled=!on);setEd(on);if(!on){select(null);if(dirty)saveDraft();}
    say(on?'점선 안을 눌러 글을 고치세요. 글·소제목·이미지는 커서가 있는 문단 바로 아래에 들어갑니다.':'읽기 모드입니다. 수정 내용은 "수정본 HTML 저장"으로 파일에 남기세요.');}
  toggle.onclick=()=>mode(!editing);
  function saveDraft(){clearTimeout(timer);timer=null;try{localStorage.setItem(key,JSON.stringify({savedAt:new Date().toISOString(),html:main.innerHTML}));say('브라우저에 임시 저장했습니다. 파일로 남기려면 "수정본 HTML 저장"을 누르세요.');}
    catch(e){say('임시 저장 용량을 넘었습니다(이미지가 큼). "수정본 HTML 저장"으로 파일에 남겨 주세요.');}}
  function changed(){dirty=true;clearTimeout(timer);timer=setTimeout(saveDraft,600);}
  function anchor(){let a=last&&main.contains(last)?last:null;if(!a)return null;const blk=a.closest('li,td,th,dt,dd,figcaption,summary');
    if(blk){a=blk.closest('ul,ol,table,dl,figure,details,.tw,.grid')||a;}const up=a.closest('.tw,.grid,.card,.mcard,.tool,figure.user-fig,details');return up&&up!==main?up:a;}
  function insert(node){const a=anchor();if(a&&a.parentNode)a.after(node);else main.append(node);node.scrollIntoView({block:'center'});changed();}
  function editable(el){el.contentEditable='true';el.dataset.ed='1';el.spellcheck=false;}
  $('rpt-add-text').onclick=()=>{const p=document.createElement('p');p.className='user-block';p.dataset.user='1';p.textContent='여기에 글을 입력하세요.';editable(p);insert(p);p.focus();last=p;};
  $('rpt-add-head').onclick=()=>{const h=document.createElement('h3');h.dataset.user='1';h.textContent='소제목';editable(h);insert(h);h.focus();last=h;};
  $('rpt-add-img').onclick=()=>file.click();
  file.onchange=()=>{const f=file.files[0];file.value='';if(!f)return;const rd=new FileReader();rd.onload=()=>{const im=new Image();im.onload=()=>{
      const s=Math.min(1,1600/Math.max(im.width,im.height)),c=document.createElement('canvas');c.width=Math.round(im.width*s);c.height=Math.round(im.height*s);
      c.getContext('2d').drawImage(im,0,0,c.width,c.height);const png=/png|gif/i.test(f.type);const url=c.toDataURL(png?'image/png':'image/jpeg',.88);
      const fig=document.createElement('figure');fig.className='user-fig';fig.dataset.user='1';const img=document.createElement('img');img.src=url;img.alt=f.name;
      const cap=document.createElement('figcaption');cap.textContent='그림 설명을 입력하세요 ('+f.name+')';editable(cap);fig.append(img,cap);insert(fig);last=cap;
      say('이미지를 넣었습니다. 아래 설명 칸을 눌러 캡션을 고치세요.');};im.src=rd.result;};rd.readAsDataURL(f);};
  function select(el){if(selected)selected.classList.remove('user-sel');selected=el;if(el)el.classList.add('user-sel');}
  main.addEventListener('click',e=>{if(!editing)return;const u=e.target.closest('[data-user="1"]');select(u);if(e.target.closest('a')&&e.target.closest('[data-ed="1"]'))e.preventDefault();},true);
  $('rpt-del').onclick=()=>{const u=selected||(last&&last.closest('[data-user="1"]'));if(!u){say('삭제할 항목을 먼저 누르세요. 원래 보고서 내용은 글을 지워서 비울 수 있습니다.');return;}
    u.remove();select(null);last=null;changed();say('추가한 항목을 삭제했습니다.');};
  main.addEventListener('focusin',e=>{const b=e.target.closest('[data-ed="1"]');if(b)last=b;});
  main.addEventListener('input',()=>{if(editing)changed();});
  main.addEventListener('paste',e=>{if(!editing||!e.target.closest('[data-ed="1"]'))return;e.preventDefault();document.execCommand('insertText',false,e.clipboardData.getData('text/plain'));});
  ['rpt-undo','rpt-redo'].forEach(id=>{$(id).onmousedown=e=>e.preventDefault();$(id).onclick=()=>{document.execCommand(id==='rpt-undo'?'undo':'redo');changed();};});
  function exportHTML(){const c=document.documentElement.cloneNode(true);c.removeAttribute('data-editing');
    c.querySelectorAll('[contenteditable]').forEach(el=>el.removeAttribute('contenteditable'));c.querySelectorAll('[data-ed]').forEach(el=>{el.removeAttribute('data-ed');el.removeAttribute('spellcheck');});
    c.querySelectorAll('.user-sel').forEach(el=>el.classList.remove('user-sel'));
    for(const el of c.querySelectorAll('[href],[src]'))for(const at of ['href','src']){const v=el.getAttribute(at);if(!v||v.startsWith('#')||/^(data:|blob:|mailto:)/i.test(v))continue;try{el.setAttribute(at,new URL(v,document.baseURI).href);}catch(_){}}
    const t=c.querySelector('#rpt-toggle');t.textContent='편집 모드 켜기';t.classList.remove('on');['rpt-add-text','rpt-add-head','rpt-add-img','rpt-del','rpt-undo','rpt-redo'].forEach(id=>c.querySelector('#'+id).setAttribute('disabled',''));
    c.querySelector('#rpt-status').textContent='저장한 수정본입니다. 편집 모드를 켜서 계속 고칠 수 있습니다.';return '<!doctype html>\n'+c.outerHTML;}
  $('rpt-save').onclick=()=>{if(timer)saveDraft();const blob=new Blob([exportHTML()],{type:'text/html;charset=utf-8'}),u=URL.createObjectURL(blob),a=document.createElement('a');
    const n=new Date(),p=x=>String(x).padStart(2,'0'),stamp=n.getFullYear()+p(n.getMonth()+1)+p(n.getDate())+'_'+p(n.getHours())+p(n.getMinutes());
    const base=decodeURIComponent(location.pathname.split('/').pop()||'report.html').replace(/\.html?$/i,'').replace(/_수정본_.*$/,'');
    a.href=u;a.download=base+'_수정본_'+stamp+'.html';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(u),10000);dirty=false;
    say('수정본 HTML을 내려받았습니다. 원본 파일은 그대로입니다. 추가한 이미지는 파일 안에 들어 있습니다.');};
  $('rpt-reset').onclick=()=>{if(!confirm('처음 내용으로 되돌릴까요? 고친 글과 추가한 글·이미지가 모두 지워집니다.'))return;
    main.innerHTML=original;try{localStorage.removeItem(key);}catch(_){}dirty=false;last=null;if(editing)setEd(true);say('처음 내용으로 되돌렸습니다.');};
  document.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='s'){e.preventDefault();$('rpt-save').click();}});
  window.addEventListener('beforeunload',()=>{if(timer)saveDraft();});
  try{const raw=localStorage.getItem(key);if(raw){const d=JSON.parse(raw);if(d.html&&d.html!==original){main.innerHTML=d.html;dirty=true;say('이 브라우저에 임시 저장된 수정 내용을 불러왔습니다('+d.savedAt.slice(0,16).replace('T',' ')+').');}}}catch(_){}
});
</script>"""


def strip(html):
    html = re.sub(r'<style id="rpt-editor-style">.*?</style>\s*', "", html, flags=re.S)
    html = re.sub(r'<script id="rpt-editor-script">.*?</script>\s*', "", html, flags=re.S)
    return re.sub(r"<!-- RPT_EDITOR_BAR_START -->.*?<!-- RPT_EDITOR_BAR_END -->\s*", "", html, flags=re.S)


def apply(html):
    html = strip(html)
    assert html.count("<main>") == html.count("</head>") == html.count("</body>") == 1
    return html.replace("</head>", STYLE+"\n</head>", 1).replace("<main>", BAR+"\n<main>", 1).replace("</body>", SCRIPT+"\n</body>", 1)


if __name__ == "__main__":
    h = "<html><head></head><body><main><p>a</p></main></body></html>"
    a = apply(h); assert apply(a) == a and a.count('id="rpt-bar"') == 1 and strip(a) == h; print("ok")
