"""Add an in-browser report editor with local drafts and HTML export.

Editing changes report text only. Measurement sources and figures stay separate.
"""
from pathlib import Path
import re

STYLE = '''<style id="report-editor-style">
#report-editor-toolbar{position:sticky;top:0;z-index:100;background:var(--card);border-bottom:1px solid var(--line);padding:9px max(12px,calc((100vw - 1080px)/2));box-shadow:0 2px 8px #0001;font:14px/1.5 system-ui,"Malgun Gothic",sans-serif}
#report-editor-toolbar .editor-actions{display:flex;flex-wrap:wrap;align-items:center;gap:7px}
#report-editor-toolbar button{font:inherit;padding:6px 11px;border:1px solid var(--line);border-radius:5px;color:var(--fg);background:var(--chip);cursor:pointer}
#report-editor-toolbar button:disabled{opacity:.45;cursor:default}
#report-editor-toolbar button[aria-pressed="true"]{color:var(--bg);background:var(--fg)}
#report-editor-toolbar .editor-save{background:var(--accent);color:white;border-color:var(--accent)}
#report-editor-status{font-size:12px;color:var(--muted);margin:5px 0 0}
.cohort-tabs{top:var(--report-editor-offset,0px)}
.cohort-panel [id]{scroll-margin-top:calc(var(--report-editor-offset,0px) + 95px)}
html[data-report-editing="true"] [data-report-editable="true"]{outline:1px dashed var(--accent);outline-offset:3px;min-height:1.2em;cursor:text}
html[data-report-editing="true"] [data-report-editable="true"]:focus{outline:2px solid var(--accent);background:var(--chip)}
html[data-report-editing="true"] abbr.cbct-term{cursor:text;text-decoration:none}
.report-user-notes{border-top:2px solid var(--line);margin-top:30px;padding-top:10px}
.report-user-note{border-left:3px solid var(--accent);padding:8px 12px;background:var(--chip);white-space:pre-wrap}
@media print{#report-editor-toolbar{display:none}:root{--report-editor-offset:0px!important}[data-report-editable="true"]{outline:none!important}}
</style>'''

TOOLBAR = '''<!-- REPORT_EDITOR_TOOLBAR_START -->
<aside id="report-editor-toolbar" aria-label="보고서 편집 도구">
<div class="editor-actions">
<button id="report-edit-toggle" type="button" aria-pressed="false">편집 모드 켜기</button>
<button id="report-edit-note" type="button" disabled>메모 추가</button>
<button id="report-edit-undo" type="button" disabled>실행 취소</button>
<button id="report-edit-redo" type="button" disabled>다시 실행</button>
<button id="report-edit-save" class="editor-save" type="button">수정본 HTML 저장</button>
<button id="report-edit-reset" type="button">열었을 때 내용으로</button>
</div>
<p id="report-editor-status" role="status" aria-live="polite">편집 모드를 켜면 문구·표를 직접 수정·삭제할 수 있습니다. 계측 원본은 변경되지 않습니다.</p>
</aside>
<!-- REPORT_EDITOR_TOOLBAR_END -->'''

SCRIPT = r'''<script id="report-editor-script">
document.addEventListener('DOMContentLoaded', () => {
  const main=document.querySelector('main'), toolbar=document.getElementById('report-editor-toolbar');
  if(!main || !toolbar)return;
  const toggle=document.getElementById('report-edit-toggle'), noteButton=document.getElementById('report-edit-note');
  const undo=document.getElementById('report-edit-undo'), redo=document.getElementById('report-edit-redo');
  const saveButton=document.getElementById('report-edit-save'), reset=document.getElementById('report-edit-reset');
  const status=document.getElementById('report-editor-status');
  const selectors='h1,h2,h3,h4,p,li,td,th,figcaption,summary';
  const blocks=[...main.querySelectorAll(selectors)].filter(el=>!el.closest('svg,script,style,iframe,[data-report-user-notes]') && !el.querySelector(selectors) && !el.querySelector('svg,img,iframe,input,button'));
  const initial=new Map(), byId=new Map();
  blocks.forEach((el,index)=>{const id='text-'+index;el.dataset.reportEditorId=id;initial.set(id,el.innerHTML);byId.set(id,el);});
  const originalNotes=[...main.querySelectorAll('[data-report-user-notes]')].map(el=>({panel:el.closest('[data-cohort-panel]').dataset.cohortPanel,html:el.innerHTML}));
  function fingerprint(text){let hash=2166136261;for(let i=0;i<text.length;i++){hash^=text.charCodeAt(i);hash=Math.imul(hash,16777619);}return (hash>>>0).toString(16);}
  const source=fingerprint(location.pathname+'|'+main.textContent+'|'+blocks.length);
  const storageKey='osa-cbct-report-draft-v1:'+source;
  let editing=false, timer=null, dirty=false, storageAvailable=true, lastFocused=null, noteSequence=0;
  function message(value){status.textContent=value;}
  function offset(){document.documentElement.style.setProperty('--report-editor-offset',toolbar.getBoundingClientRect().height+'px');}
  offset();window.addEventListener('resize',offset);
  if(typeof ResizeObserver!=='undefined')new ResizeObserver(offset).observe(toolbar);
  function safeMarkup(value){
    const template=document.createElement('template');template.innerHTML=String(value);
    const allowed=new Set(['ABBR','A','B','I','EM','STRONG','BR','SPAN','SUB','SUP','CODE','SMALL','DIV','P']);
    const attrs=new Set(['class','title','href','target','rel','tabindex','data-term','data-definition','data-example-src','data-example-caption']);
    for(const el of [...template.content.querySelectorAll('*')]){
      if(!allowed.has(el.tagName)){el.replaceWith(...el.childNodes);continue;}
      for(const attr of [...el.attributes]){
        if(!attrs.has(attr.name)){el.removeAttribute(attr.name);continue;}
        if(['href','data-example-src'].includes(attr.name)){
          try{if(!['file:','http:','https:','mailto:','tel:'].includes(new URL(attr.value,document.baseURI).protocol))el.removeAttribute(attr.name);}catch(_){el.removeAttribute(attr.name);}
        }
      }
      if(el.tagName==='A' && el.target==='_blank')el.rel='noopener';
    }
    return template.innerHTML;
  }
  function notesSnapshot(){return [...main.querySelectorAll('[data-report-user-notes]')].map(el=>({panel:el.closest('[data-cohort-panel]').dataset.cohortPanel,html:el.innerHTML}));}
  function changes(){return [...byId].filter(([id,el])=>el.innerHTML!==initial.get(id)).map(([id,el])=>({id,html:el.innerHTML}));}
  function draft(){return {version:1,source,savedAt:new Date().toISOString(),edits:changes(),notes:notesSnapshot()};}
  function saveDraft(){
    clearTimeout(timer);timer=null;
    try{localStorage.setItem(storageKey,JSON.stringify(draft()));storageAvailable=true;message('초안이 이 브라우저에 저장됐습니다. 수정 내용을 파일로 남기려면 수정본 HTML 저장을 누르세요.');return true;}
    catch(_){storageAvailable=false;message('브라우저 초안을 저장할 수 없습니다. 수정본 HTML 저장으로 파일을 내려받아 주세요.');return false;}
  }
  function changed(){dirty=true;message('수정 중…');clearTimeout(timer);timer=setTimeout(saveDraft,500);}
  function noteSection(panel){
    let section=panel.querySelector('[data-report-user-notes]');
    if(!section){section=document.createElement('section');section.className='report-user-notes';section.dataset.reportUserNotes='true';const heading=document.createElement('h2');heading.textContent='내 메모';section.append(heading);panel.append(section);}
    return section;
  }
  function configure(el,on){el.contentEditable=String(on);el.dataset.reportEditable=String(on);if(on)el.setAttribute('spellcheck','false');else el.removeAttribute('spellcheck');}
  function restoreNotes(notes){
    main.querySelectorAll('[data-report-user-notes]').forEach(el=>el.remove());
    for(const item of notes || []){
      const panel=[...main.querySelectorAll('[data-cohort-panel]')].find(el=>el.dataset.cohortPanel===item.panel);
      if(!panel)continue;
      const section=noteSection(panel);section.innerHTML='';
      const template=document.createElement('template');template.innerHTML=String(item.html);
      const heading=document.createElement('h2');heading.textContent='내 메모';section.append(heading);
      for(const old of template.content.querySelectorAll('.report-user-note')){
        const paragraph=document.createElement('p');paragraph.className='report-user-note';paragraph.dataset.reportUserNote='true';paragraph.innerHTML=safeMarkup(old.innerHTML);configure(paragraph,editing);section.append(paragraph);
      }
    }
  }
  try{
    const raw=localStorage.getItem(storageKey);
    if(raw){const saved=JSON.parse(raw);if(saved.version===1 && saved.source===source && ((saved.edits || []).length || JSON.stringify(saved.notes || [])!==JSON.stringify(originalNotes))){
      for(const edit of saved.edits || [])if(byId.has(edit.id))byId.get(edit.id).innerHTML=safeMarkup(edit.html);
      restoreNotes(saved.notes);dirty=true;message('이 브라우저에 저장된 수정 초안을 복원했습니다. 수정본 HTML 저장으로 파일에 남길 수 있습니다.');
    }}
  }catch(_){storageAvailable=false;message('브라우저 초안 기능을 사용할 수 없습니다. 편집 후 수정본 HTML 저장을 이용해 주세요.');}
  function mode(on){
    editing=on;document.documentElement.dataset.reportEditing=String(on);toggle.setAttribute('aria-pressed',String(on));toggle.textContent=on?'편집 완료':'편집 모드 켜기';
    [...blocks,...main.querySelectorAll('.report-user-note')].forEach(el=>configure(el,on));
    noteButton.disabled=undo.disabled=redo.disabled=!on;
    const closeTip=document.getElementById('cbct-term-close');if(closeTip)closeTip.click();
    if(!on && dirty)saveDraft();else message(on?'점선 테두리 안을 눌러 문구·표를 수정하세요. 용어 팝업은 편집 중 잠시 꺼집니다.':'읽기 모드입니다. 수정본 HTML 저장으로 편집 내용을 파일에 남길 수 있습니다.');
  }
  toggle.addEventListener('click',()=>mode(!editing));
  main.addEventListener('focusin',e=>{const block=e.target.closest('[data-report-editable="true"]');if(block)lastFocused=block;});
  main.addEventListener('input',e=>{
    const block=e.target.closest('[data-report-editable="true"]');if(!editing || !block)return;
    block.querySelectorAll('abbr.cbct-term').forEach(term=>{if(term.textContent!==term.dataset.term)term.replaceWith(document.createTextNode(term.textContent));});
    changed();
  });
  main.addEventListener('click',e=>{if(editing && e.target.closest('a') && e.target.closest('[data-report-editable="true"]'))e.preventDefault();},true);
  main.addEventListener('paste',e=>{
    if(!editing || !e.target.closest('[data-report-editable="true"]'))return;
    e.preventDefault();document.execCommand('insertText',false,e.clipboardData.getData('text/plain'));changed();
  });
  main.addEventListener('drop',e=>{if(editing && e.target.closest('[data-report-editable="true"]'))e.preventDefault();});
  undo.addEventListener('mousedown',e=>e.preventDefault());redo.addEventListener('mousedown',e=>e.preventDefault());
  undo.addEventListener('click',()=>{document.execCommand('undo');changed();});redo.addEventListener('click',()=>{document.execCommand('redo');changed();});
  noteButton.addEventListener('click',()=>{
    const panel=[...main.querySelectorAll('[data-cohort-panel]')].find(el=>!el.hidden);if(!panel)return;
    const paragraph=document.createElement('p');paragraph.className='report-user-note';paragraph.dataset.reportUserNote='true';paragraph.textContent='메모를 입력하세요.';paragraph.dataset.noteSequence=String(++noteSequence);configure(paragraph,true);noteSection(panel).append(paragraph);paragraph.focus();paragraph.scrollIntoView({block:'center'});changed();
  });
  function exportHTML(){
    const clone=document.documentElement.cloneNode(true);
    clone.removeAttribute('data-report-editing');clone.style.removeProperty('--report-editor-offset');
    clone.querySelectorAll('[contenteditable]').forEach(el=>el.removeAttribute('contenteditable'));
    clone.querySelectorAll('[data-report-editable],[data-report-editor-id],[data-note-sequence]').forEach(el=>{el.removeAttribute('data-report-editable');el.removeAttribute('data-report-editor-id');el.removeAttribute('data-note-sequence');el.removeAttribute('spellcheck');});
    clone.querySelectorAll('abbr.cbct-term').forEach(el=>{el.removeAttribute('aria-describedby');el.title=el.dataset.definition;});
    const liveInputs=[...document.querySelectorAll('input')];
    clone.querySelectorAll('input').forEach((el,index)=>{if(liveInputs[index] && ['checkbox','radio'].includes(el.type)){if(liveInputs[index].checked)el.setAttribute('checked','');else el.removeAttribute('checked');}});
    for(const el of clone.querySelectorAll('[href],[src],[data-example-src]'))for(const attr of ['href','src','data-example-src']){
      const value=el.getAttribute(attr);if(!value || value.startsWith('#') || /^(data:|blob:|mailto:|tel:)/i.test(value))continue;
      try{el.setAttribute(attr,new URL(value,document.baseURI).href);}catch(_){}
    }
    const activePanel=[...main.querySelectorAll('[data-cohort-panel]')].find(el=>!el.hidden);if(activePanel)clone.querySelector('body').dataset.reportInitialCohort=activePanel.dataset.cohortPanel;
    const button=clone.querySelector('#report-edit-toggle');button.textContent='편집 모드 켜기';button.setAttribute('aria-pressed','false');
    for(const id of ['report-edit-note','report-edit-undo','report-edit-redo'])clone.querySelector('#'+id).disabled=true;
    clone.querySelector('#report-editor-status').textContent='저장된 수정본입니다. 편집 모드를 켜서 계속 수정할 수 있습니다.';
    const tip=clone.querySelector('#cbct-term-tooltip');if(tip){tip.hidden=true;tip.setAttribute('role','tooltip');}
    return '<!doctype html>\n'+clone.outerHTML;
  }
  saveButton.addEventListener('click',()=>{
    saveDraft();
    const content=exportHTML(),blob=new Blob([content],{type:'text/html;charset=utf-8'}),url=URL.createObjectURL(blob),anchor=document.createElement('a');
    const now=new Date(),pad=n=>String(n).padStart(2,'0');const stamp=now.getFullYear()+'-'+pad(now.getMonth()+1)+'-'+pad(now.getDate())+'_'+pad(now.getHours())+pad(now.getMinutes());
    const original=decodeURIComponent(new URL(location.href).pathname.split('/').pop() || 'CBCT_report.html').replace(/\.html?$/i,'').replace(/_수정본_.*$/,'');
    anchor.href=url;anchor.download=original+'_수정본_'+stamp+'.html';document.body.append(anchor);anchor.click();anchor.remove();setTimeout(()=>URL.revokeObjectURL(url),10000);dirty=false;
    message('수정본 HTML 다운로드를 요청했습니다. 그림은 현재 PC의 원본 경로에 연결되며, 원본 HTML은 덮어쓰지 않습니다.');
  });
  reset.addEventListener('click',()=>{
    if(!confirm('이 파일을 열었을 때의 내용으로 되돌릴까요? 현재 문구 수정과 새 메모가 지워집니다.'))return;
    clearTimeout(timer);for(const [id,el] of byId)el.innerHTML=initial.get(id);restoreNotes(originalNotes);try{localStorage.removeItem(storageKey);}catch(_){}dirty=false;message('파일을 열었을 때의 내용으로 되돌렸습니다.');
  });
  document.addEventListener('keydown',e=>{if((e.ctrlKey || e.metaKey) && e.key.toLowerCase()==='s'){e.preventDefault();saveButton.click();}});
  window.addEventListener('beforeunload',e=>{if(timer)saveDraft();if(dirty && !storageAvailable){e.preventDefault();e.returnValue='';}});
});
</script>'''


def strip_editor(html):
    html=re.sub(r'<style id="report-editor-style">.*?</style>\s*','',html,flags=re.S)
    html=re.sub(r'<script id="report-editor-script">.*?</script>\s*','',html,flags=re.S)
    html=re.sub(r'<!-- REPORT_EDITOR_TOOLBAR_START -->.*?<!-- REPORT_EDITOR_TOOLBAR_END -->\s*','',html,flags=re.S)
    return html


def apply_editor(html):
    html=strip_editor(html)
    assert html.count('<main>')==html.count('</head>')==html.count('</body>')==1
    html=html.replace('</head>',STYLE+'\n</head>',1)
    html=html.replace('<main>',TOOLBAR+'\n<main>',1)
    html=html.replace('</body>',SCRIPT+'\n</body>',1)
    return html


def main():
    from cbct_report_cohorts import REPORT
    before=REPORT.read_text(encoding='utf-8');after=apply_editor(before)
    svg=re.compile(r'<svg\b.*?</svg>',re.S)
    assert svg.findall(before)==svg.findall(after)
    assert after==apply_editor(after),'Editor injection must be idempotent'
    REPORT.write_text(after,encoding='utf-8')
    print('Editor added: inline text/table editing, local drafts, notes, undo/redo, downloadable HTML with preserved image links')


if __name__=='__main__':main()
