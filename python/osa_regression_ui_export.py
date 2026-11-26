"""회귀 UI를 Python 없이 실행되는 HTML 한 장으로 내보낸다(배포용 zip).

osa_regression_ui.py의 서버 요청 3개를 페이지 안에서 처리한다.
  /api/features  → 미리 만든 피처 목록(메타데이터)
  /api/anatomy   → 미리 그린 대표 CT 이미지(예시 환자 P_2, atlas() 결과)
  /api/regression→ 자바스크립트 OLS(regress()와 같은 계산: 완전 사례, 표준화 후 적합, 원 단위 계수·SE, t 검정 p, 95% CI, R²)
자료: load_data()의 피처 63개(기본 = 병합 엑셀 값)와 결과변수 4개. 환자 번호는 넣지 않는다.
출력: Project/dist/OSA_regression_ui/ (osa_regression_ui.html, OSA_regression_ui.bat) 과 OSA/OSA_regression_ui.zip
검증: python osa_regression_ui_export.py --check  (node로 Python regress()와 결과 비교)
"""
import json
import subprocess
import sys
import zipfile

import numpy as np

import osa_regression_ui as U
from run_ali import ROOT

DIST = ROOT/"dist/OSA_regression_ui"; ZIP = ROOT.parent/"OSA_regression_ui.zip"

SHIM = r"""<script>
const OSA_DATA = __DATA__;
(function(){
  const reply = (status, body) => new Response(JSON.stringify(body), {status, headers: {"Content-Type": "application/json"}});
  function betacf(a, b, x){ let qab=a+b, qap=a+1, qam=a-1, c=1, d=1-qab*x/qap; if(Math.abs(d)<1e-300) d=1e-300; d=1/d; let h=d;
    for(let m=1;m<=300;m++){ const m2=2*m; let aa=m*(b-m)*x/((qam+m2)*(a+m2)); d=1+aa*d; if(Math.abs(d)<1e-300) d=1e-300; c=1+aa/c; if(Math.abs(c)<1e-300) c=1e-300; d=1/d; h*=d*c;
      aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2)); d=1+aa*d; if(Math.abs(d)<1e-300) d=1e-300; c=1+aa/c; if(Math.abs(c)<1e-300) c=1e-300; d=1/d; const del=d*c; h*=del; if(Math.abs(del-1)<1e-15) break; } return h; }
  function lgamma(z){ const g=[676.5203681218851,-1259.1392167224028,771.32342877765313,-176.61502916214059,12.507343278686905,-0.13857109526572012,9.9843695780195716e-6,1.5056327351493116e-7];
    if(z<0.5) return Math.log(Math.PI/Math.sin(Math.PI*z))-lgamma(1-z); z-=1; let x=0.99999999999980993; for(let i=0;i<8;i++) x+=g[i]/(z+i+1); const t=z+7.5; return 0.5*Math.log(2*Math.PI)+(z+0.5)*Math.log(t)-t+Math.log(x); }
  function ibeta(x, a, b){ if(x<=0) return 0; if(x>=1) return 1; const bt=Math.exp(lgamma(a+b)-lgamma(a)-lgamma(b)+a*Math.log(x)+b*Math.log(1-x));
    return x<(a+1)/(a+b+2) ? bt*betacf(a,b,x)/a : 1-bt*betacf(b,a,1-x)/b; }
  const tsf2 = (t, df) => ibeta(df/(df+t*t), df/2, 0.5);                       // 양측 p = 2·sf(|t|)
  function tppf975(df){ let lo=0, hi=1000; for(let i=0;i<200;i++){ const m=(lo+hi)/2; (tsf2(m,df)>0.05 ? lo=m : hi=m); } return (lo+hi)/2; }
  function invert(A){ const n=A.length, M=A.map((r,i)=>[...r, ...r.map((_,j)=>i===j?1:0)]);
    const scale=Math.max(...A.map(r=>Math.max(...r.map(Math.abs))));
    for(let c=0;c<n;c++){ let p=c; for(let r=c+1;r<n;r++) if(Math.abs(M[r][c])>Math.abs(M[p][c])) p=r;
      if(Math.abs(M[p][c])<=1e-10*scale) return null; [M[c],M[p]]=[M[p],M[c]]; const v=M[c][c]; for(let j=0;j<2*n;j++) M[c][j]/=v;
      for(let r=0;r<n;r++) if(r!==c){ const f=M[r][c]; if(f) for(let j=0;j<2*n;j++) M[r][j]-=f*M[c][j]; } }
    return M.map(r=>r.slice(n)); }
  function regress(predictors, outcome){
    const O=OSA_DATA.outcomes, F=OSA_DATA.features;
    if(typeof outcome!=="string" || !(outcome in OSA_DATA.outcomeLabels)) throw new Error("결과변수를 선택해 주세요.");
    if(!Array.isArray(predictors) || !predictors.length) throw new Error("피처를 하나 이상 선택해 주세요.");
    if(predictors.some(k=>typeof k!=="string" || !(k in F))) throw new Error("피처 목록에 없는 변수가 들어 있습니다.");
    if(new Set(predictors).size!==predictors.length) throw new Error("같은 피처가 중복 선택됐습니다.");
    const rows=[]; for(let i=0;i<O[outcome].length;i++){ const xs=predictors.map(k=>F[k][i]), y=O[outcome][i];
      if(y!==null && Number.isFinite(y) && xs.every(v=>v!==null && Number.isFinite(v))) rows.push([xs,y]); }
    const n=rows.length, k=predictors.length;
    if(n<=k+1) throw new Error(`완전한 자료는 ${n}명입니다. 피처 ${k}개의 계수와 표준오차를 계산하려면 최소 ${k+2}명이 필요합니다.`);
    const mean=predictors.map((_,j)=>rows.reduce((s,r)=>s+r[0][j],0)/n);
    const sd=predictors.map((_,j)=>Math.sqrt(rows.reduce((s,r)=>s+(r[0][j]-mean[j])**2,0)/n));
    const constant=predictors.filter((_,j)=>sd[j]===0); if(constant.length) throw new Error("분석 표본에서 값이 일정한 피처가 있습니다. "+constant.join(", "));
    const ys=rows.map(r=>r[1]); if(Math.max(...ys)===Math.min(...ys)) throw new Error("분석 표본에서 결과변수 값이 일정합니다.");
    const Xall=rows.map(r=>[1, ...r[0].map((v,j)=>(v-mean[j])/sd[j])]), gram=c=>c.map(a=>c.map(b=>Xall.reduce((s,r)=>s+r[a]*r[b],0)));
    let keep=[0]; for(let j=1;j<=k;j++){ if(invert(gram([...keep,j]))) keep.push(j); }                 // 선형 종속 피처는 빼고 계산(SPSS 방식)
    const dropped=predictors.filter((_,j)=>!keep.includes(j+1)), warning=dropped.length ? "선택한 피처 사이에 선형 종속이 있어 "+dropped.join(", ")+"을(를) 빼고 계산했습니다." : null;
    predictors=keep.slice(1).map(j=>predictors[j-1]); const mean2=keep.slice(1).map(j=>mean[j-1]), sd2=keep.slice(1).map(j=>sd[j-1]);
    mean.length=0; mean.push(...mean2); sd.length=0; sd.push(...sd2);
    const X=Xall.map(r=>keep.map(j=>r[j])), p=keep.length;
    const XtX=gram(keep).map(r=>r), inv=invert(XtX);
    const Xty=Array.from({length:p},(_,a)=>X.reduce((s,r,i)=>s+r[a]*ys[i],0)), beta=inv.map(r=>r.reduce((s,v,b)=>s+v*Xty[b],0));
    const fit=X.map(r=>r.reduce((s,v,a)=>s+v*beta[a],0)), rss=ys.reduce((s,y,i)=>s+(y-fit[i])**2,0), ym=ys.reduce((s,y)=>s+y,0)/n, tss=ys.reduce((s,y)=>s+(y-ym)**2,0);
    const df=n-p, s2=rss/df, T=Array.from({length:p},(_,a)=>Array.from({length:p},(_,b)=>a===b?(a===0?1:1/sd[a-1]):(a===0&&b>0?-mean[b-1]/sd[b-1]:0)));
    const coef=T.map(r=>r.reduce((s,v,b)=>s+v*beta[b],0));
    const cov=inv.map(r=>r.map(v=>v*s2)), TC=T.map(r=>Array.from({length:p},(_,b)=>r.reduce((s,v,a)=>s+v*cov[a][b],0)));
    const se=T.map((r,i)=>Math.sqrt(Math.max(0, TC[i].reduce((s,v,b)=>s+v*r[b],0)))), crit=tppf975(df);
    const terms=["절편",...predictors].map((name,i)=>({name, b:coef[i], se:se[i], p: se[i]>0 ? tsf2(Math.abs(coef[i]/se[i]),df) : null, low:coef[i]-crit*se[i], high:coef[i]+crit*se[i]}));
    const r2=1-rss/tss; return {outcome:OSA_DATA.outcomeLabels[outcome], n, missing:O[outcome].length-n, terms, r2, adjustedR2:1-(1-r2)*(n-1)/df, warning};
  }
  window.OSA_REGRESS = regress;
  window.fetch = async (url, options) => {
    if(url==="/api/features") return reply(200, OSA_DATA.meta);
    if(url==="/api/anatomy") return OSA_DATA.atlas ? reply(200, OSA_DATA.atlas) : reply(503, {error:"CBCT 부위 이미지를 준비하지 못했습니다."});
    if(url==="/api/regression"){ try{ const q=JSON.parse(options.body); return reply(200, regress(q.predictors, q.outcome)); } catch(e){ return reply(400, {error:e.message}); } }
    return reply(404, {error:"페이지를 찾을 수 없습니다."});
  };
})();
</script>
"""
BAT = '@echo off\r\ncd /d "%~dp0"\r\nstart "" "osa_regression_ui.html"\r\n'


def col(s, digits=15):
    return [None if not np.isfinite(v) else float(f"{v:.{digits}g}") for v in s.to_numpy(dtype=float)]


def build():
    features, outcomes, meta = U.load_data()
    data = {"meta": meta, "atlas": U.atlas(), "outcomeLabels": U.OUTCOMES,
            "features": {k: col(features[k]) for k in features}, "outcomes": {k: col(outcomes[k]) for k in U.OUTCOMES}}
    page = (ROOT/"python/osa_regression_ui.html").read_text(encoding="utf-8")
    head = page.index("<script>")
    html = page[:head] + SHIM.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))) + page[head:]
    DIST.mkdir(parents=True, exist_ok=True)
    (DIST/"osa_regression_ui.html").write_text(html, encoding="utf-8")
    (DIST/"OSA_regression_ui.bat").write_bytes(BAT.encode("ascii"))
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in ("OSA_regression_ui.bat", "osa_regression_ui.html"): z.write(DIST/f, f"OSA_regression_ui/{f}")
    return features, outcomes, data


def check(features, outcomes, data):
    """같은 자료로 Python regress()와 자바스크립트 regress()를 비교한다(값을 15자리로 넣어 반올림 오차가 거의 없음)."""
    cases = [(["[기본] ANB", "[기본] MP-FH", "[기본] TDI", "[기본] PMCA", "[기본] PAL", "Age", "Male", "BMI"], "AHI"),
             (["[기본] Nperp-Pog", "airway_shapePC1", "Age", "Male", "BMI"], "logAHI"), (["tongue_vol"], "logAHI_SUP"),
             (list(features.columns), "AHI")]                                                          # 전체 선택: 선형 종속 피처 제외 확인
    js = "const fs=require('fs');const h=fs.readFileSync(process.argv[1],'utf8');const s=h.slice(h.indexOf('<script>')+8,h.indexOf('</script>'));" \
         "global.window={};global.Response=class{};eval(s.replace('const OSA_DATA','global.OSA_DATA'));" \
         f"console.log(JSON.stringify({json.dumps(cases, ensure_ascii=False)}.map(([p,o])=>window.OSA_REGRESS(p,o))));"
    out = json.loads(subprocess.run(["node", "-e", js, str(DIST/"osa_regression_ui.html")], capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    for (p, o), j in zip(cases, out):
        r = U.regress(features, outcomes, p, o); worst = 0.
        assert r["n"] == j["n"] and r["warning"] == j["warning"] and len(r["terms"]) == len(j["terms"]), (r["n"], j["n"], r["warning"], j["warning"])
        for a, b in zip(r["terms"], j["terms"]):
            for key in ("b", "se", "p", "low", "high"):
                worst = max(worst, abs(a[key]-b[key])/max(1e-8, abs(a[key])))
        worst = max(worst, abs(r["r2"]-j["r2"]))
        print(f"{o} · 피처 {len(p)}개 · n={r['n']} · 최대 상대 차이 {worst:.2e}"); assert worst < 1e-3


def main():
    f, o, d = build()
    print("HTML", round((DIST/"osa_regression_ui.html").stat().st_size/1e6, 2), "MB · zip", round(ZIP.stat().st_size/1e6, 2), "MB ·", ZIP)
    if "--check" in sys.argv: check(f, o, d)


if __name__ == "__main__":
    main()
