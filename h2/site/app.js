const D = window.H2;
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const fmt = (n,d=2)=>`${n>=0?'+':'−'}${Math.abs(n).toFixed(d)}`;
const clamp=(n,a,b)=>Math.max(a,Math.min(b,n));

// progress + scrollspy
const sections=$$('.observe-section');
window.addEventListener('scroll',()=>{
  const h=document.documentElement.scrollHeight-innerHeight;
  $('#progress').style.width=`${h?scrollY/h*100:0}%`;
  let current='top';
  sections.forEach(s=>{if(s.getBoundingClientRect().top<180) current=s.id});
  $$('.rail-link').forEach(a=>a.classList.toggle('active',a.getAttribute('href')===`#${current}`));
});

// plain mode
$('#plain-toggle').addEventListener('click',e=>{
  e.currentTarget.classList.toggle('active');
  const on=e.currentTarget.classList.contains('active');
  $$('.plain-note').forEach(n=>n.hidden=!on);
});

// glossary
const drawer=$('#glossary'), scrim=$('#scrim');
function openGloss(){drawer.classList.add('open');drawer.setAttribute('aria-hidden','false');scrim.hidden=false}
function closeGloss(){drawer.classList.remove('open');drawer.setAttribute('aria-hidden','true');scrim.hidden=true}
$('#glossary-toggle').addEventListener('click',openGloss);$('#glossary-close').addEventListener('click',closeGloss);scrim.addEventListener('click',closeGloss);

// timeline
const lagSlider=$('#lag-slider');
function updateLag(){
  const r=D.timeline[+lagSlider.value];
  $('#lag-number').textContent=r.n;
  $('#conv-card').classList.toggle('off',!r.conv); $('#kv-card').classList.toggle('off',!r.kv);
  $('#trace-num').textContent=r.r.toFixed(3); $('#trace-bar').style.width=`${clamp(r.r/.34*100,0,100)}%`;
  $('#recall-num').textContent=`${Math.round(r.recall*100)}%`; $('#recall-bar').style.width=`${r.recall*100}%`;
  $('#lag-note').textContent=r.note;
  $('#conv-card strong').textContent=r.conv?'DIRECT TRACE':'EVICTED'; $('#kv-card strong').textContent=r.kv?'DIRECT TRACE':'EVICTED';
}
lagSlider.addEventListener('input',updateLag);updateLag();

// transplant
const tb=$('#transplant-buttons');
tb.innerHTML=D.transplant.map((r,i)=>`<button class="${i===0?'active':''}" data-id="${r.id}">${r.short}</button>`).join('');
function updateTransplant(id){
  const r=D.transplant.find(x=>x.id===id); $$('#transplant-buttons button').forEach(b=>b.classList.toggle('active',b.dataset.id===id));
  $('#effect-estimate').textContent=fmt(r.estimate);
  const lo=-30, hi=250; $('#effect-ci').style.left=`${(clamp(r.ci[0],lo,hi)-lo)/(hi-lo)*100}%`; $('#effect-ci').style.width=`${(clamp(r.ci[1],lo,hi)-clamp(r.ci[0],lo,hi))/(hi-lo)*100}%`; $('#effect-point').style.left=`${(clamp(r.estimate,lo,hi)-lo)/(hi-lo)*100}%`; $('#effect-copy').textContent=`${r.copy} 95% CI [${fmt(r.ci[0])}, ${fmt(r.ci[1])}].`;
}
tb.addEventListener('click',e=>{if(e.target.matches('button'))updateTransplant(e.target.dataset.id)});updateTransplant('match');

// specificity bars
const specRows=[['Matching',121.62],['Same-template wrong',83.13],['Cross-template',75.75],['Noise',48.23]]; const sm=Math.max(...specRows.map(x=>x[1]));
$('#spec-bars').innerHTML=specRows.map(([k,v])=>`<div class="spec-row"><label>${k}</label><div class="spec-track"><i style="width:${v/sm*100}%"></i></div><strong>${fmt(v)}</strong></div>`).join('');

// dynamics
const dyn=$('#dyn-slider');
function updateDyn(){
  const r=D.dynamics[+dyn.value]; $('#dyn-n').textContent=r.n; $('#dyn-v0').textContent=fmt(r.v0); $('#dyn-vn').textContent=fmt(r.vn); $('#dyn-cr').textContent=r.cr.toFixed(3); $('#memory-vector').style.transform=`rotate(${r.angle}deg)`;
  $('#dyn-copy').textContent=r.n===0?'At the standardized origin, the recurrent difference aligns with the old value-specific output axis.':r.n===2048?'By N=2048 the old-axis effect is unresolved, Cᵣ≈0.124, yet contemporaneous steerability remains positive. The distinction transformed rather than simply disappearing.':`As future drive accumulates, state alignment falls and the old output direction becomes an increasingly poor ruler.`;
}
dyn.addEventListener('input',updateDyn);updateDyn();

// strict-C
function pos(v,range=1){return `${clamp((v+range)/(2*range)*100,3,97)}%`}
function updateStrict(dir){
  const r=D.strictC[dir]; $$('.seg').forEach(b=>b.classList.toggle('active',b.dataset.dir===dir)); $('#truth-word').textContent=r.truth; $('#dt').textContent=fmt(r.dt,3); $('#do').textContent=fmt(r.d0,3); $('#mpre').textContent=fmt(r.mpre,3); $('#mobs').textContent=fmt(r.mobs,3); $('#pai').textContent=fmt(r.pai,3); $('#dt-dot').style.left=pos(r.dt,.7); $('#do-dot').style.left=pos(r.d0,.7); $('#pai-copy').textContent=dir==='fwd'?'The report remains wrong in absolute terms, but shifts toward the target’s private fact relative to the observer.':'The report shifts toward the private fact and crosses the semantic boundary in this direction.';
}
$$('.seg').forEach(b=>b.addEventListener('click',()=>updateStrict(b.dataset.dir)));updateStrict('fwd');
$('#tier-strip').innerHTML=D.tiers.map(t=>`<article class="tier-cell ${t.tone}"><small>${t.name}</small><strong>${t.n}/16</strong><p>${t.detail}</p></article>`).join('');

// TOST
function axisPct(v){return (v+.125)/.25*100}
function updateTost(which){
  const r=D.tost[which]; $$('.tost-btn').forEach(b=>b.classList.toggle('active',b.dataset.tost===which)); $('#tost-mean').textContent=fmt(r.mean,4); $('#tost-p').textContent=`pTOST = ${r.p<.001?r.p.toExponential(2):r.p.toFixed(4)}`; $('#ci90').style.left=`${axisPct(r.ci90[0])}%`; $('#ci90').style.width=`${axisPct(r.ci90[1])-axisPct(r.ci90[0])}%`; $('#tost-point').style.left=`${axisPct(r.mean)}%`; $('#tost-copy').textContent=`90% CI [${fmt(r.ci90[0],4)}, ${fmt(r.ci90[1],4)}]. Practically equivalent on average at the ±0.10-logit boundary.`;
}
$$('.tost-btn').forEach(b=>b.addEventListener('click',()=>updateTost(b.dataset.tost)));updateTost('evolved');

// ladder
$('#ladder').innerHTML=D.ladder.map(r=>`<article class="ladder-row"><span class="num">${r.n}</span><strong>${r.left}</strong><span class="neq">≠</span><strong>${r.right}</strong><span class="answer">${r.answer}</span><p>${r.detail}</p></article>`).join('');
