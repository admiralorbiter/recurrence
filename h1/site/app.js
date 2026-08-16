(() => {
const glossary = window.GLOSSARY || {};
const $ = (s,r=document)=>r.querySelector(s);
const $$ = (s,r=document)=>Array.from(r.querySelectorAll(s));

function openGlossary(termKey){
  const drawer=$("#glossary-drawer"), scrim=$("#scrim"), search=$("#glossary-search");
  drawer?.classList.add("open"); drawer?.setAttribute("aria-hidden","false");
  if(scrim) scrim.hidden=false;
  if(search){search.value=termKey && glossary[termKey] ? glossary[termKey].term : ""; renderGlossary(search.value);}
}
function closeGlossary(){
  $("#glossary-drawer")?.classList.remove("open");
  $("#glossary-drawer")?.setAttribute("aria-hidden","true");
  const scrim=$("#scrim"); if(scrim) scrim.hidden=true;
}
function renderGlossary(q=""){
  const list=$("#glossary-list"); if(!list) return;
  q=q.toLowerCase().trim();
  list.innerHTML=Object.entries(glossary)
   .filter(([k,v])=>!q || k.includes(q) || v.term.toLowerCase().includes(q) || v.short.toLowerCase().includes(q))
   .map(([k,v])=>`<div class="glossary-entry"><h3>${v.term}</h3><p>${v.short}</p><p>${v.long}</p></div>`).join("");
}
$("#glossary-open")?.addEventListener("click",()=>openGlossary());
$("#floating-help")?.addEventListener("click",()=>openGlossary());
$("#glossary-close")?.addEventListener("click",closeGlossary);
$("#scrim")?.addEventListener("click",closeGlossary);
$("#glossary-search")?.addEventListener("input",e=>renderGlossary(e.target.value));
renderGlossary();

const tooltip=$("#tooltip");
$$(".term").forEach(btn=>{
  const key=btn.dataset.term;
  const show=()=>{
    const v=glossary[key]; if(!v || !tooltip) return;
    tooltip.innerHTML=`<strong>${v.term}</strong><br>${v.short}`;
    tooltip.hidden=false;
    const r=btn.getBoundingClientRect();
    tooltip.style.left=Math.min(window.innerWidth-350, Math.max(8,r.left))+"px";
    tooltip.style.top=Math.min(window.innerHeight-130,r.bottom+8)+"px";
  };
  btn.addEventListener("mouseenter",show); btn.addEventListener("focus",show);
  btn.addEventListener("mouseleave",()=>{if(tooltip) tooltip.hidden=true});
  btn.addEventListener("blur",()=>{if(tooltip) tooltip.hidden=true});
  btn.addEventListener("click",()=>openGlossary(key));
});

$("#plain-toggle")?.addEventListener("click",e=>{
  const recap=$("#plain-recap"); if(!recap) return;
  recap.hidden=!recap.hidden;
  e.currentTarget.classList.toggle("active",!recap.hidden);
});

$$(".museum-card > button").forEach(b=>b.addEventListener("click",()=>b.parentElement.classList.toggle("open")));

$$(".checkpoint button").forEach(b=>b.addEventListener("click",()=>{
  const box=b.closest(".checkpoint"); const feedback=$(".feedback",box);
  if(!feedback) return;
  const correct=b.dataset.correct==="true";
  feedback.hidden=false; feedback.classList.toggle("bad",!correct);
  feedback.textContent=correct ? b.dataset.good || "Yes. That is the intended interpretation." : b.dataset.bad || "Not quite. Re-read the distinction above.";
}));

// S04 memory switcher
const memoryBox=$("#memory-lab");
if(memoryBox && window.H1_DATA){
  const buttons=$$(".seg",memoryBox), out=$("#memory-output",memoryBox);
  function render(name){
    const d=window.H1_DATA.s04.find(x=>x.name===name); if(!d) return;
    out.innerHTML=`<div class="compare-pair"><div><span>Accuracy</span><strong>${d.acc.toFixed(1)}%</strong></div><div><span>Prompt tokens</span><strong>${d.tok}</strong></div></div>
      <div class="bar-wrap"><div class="bar-label"><span>Accuracy</span><span>${d.acc.toFixed(1)}%</span></div><div class="bar memory"><div style="width:${d.acc}%"></div></div></div>`;
  }
  buttons.forEach(b=>b.addEventListener("click",()=>{buttons.forEach(x=>x.classList.remove("active"));b.classList.add("active");render(b.dataset.name)}));
  render("Full transcript");
}

// S05 error inheritance
const errBtn=$("#error-toggle");
if(errBtn){
  let bad=false; const state=$("#error-state"), msg=$("#error-msg");
  errBtn.addEventListener("click",()=>{
    bad=!bad;
    state.innerHTML = bad
      ? `<code>reactor_temp = "stable"</code><br><code>door_status = "OPEN"</code> <strong>← erroneous write</strong><br><code>goal = "inspect"</code>`
      : `<code>reactor_temp = "stable"</code><br><code>door_status = "CLOSED"</code><br><code>goal = "inspect"</code>`;
    msg.textContent = bad ? "The persistence layer now faithfully protects the wrong door status. Continuity preserved the error." : "The state is currently correct.";
    errBtn.textContent = bad ? "Repair the write" : "Inject an erroneous write";
  });
}

// S06 replay demo
const replayBtn=$("#replay-run");
if(replayBtn){
  replayBtn.addEventListener("click",()=>{
    $("#online-state").innerHTML="<code>{alpha: prism, goal: active}</code>";
    $("#replay-state").innerHTML="<code>{alpha: prism, goal: active}</code>";
    $("#replay-verdict").innerHTML="<strong>State hash: identical</strong><br><span class='micro'>Same ordered events + same deterministic transition rule → same terminal explicit state.</span>";
  });
}

// Horizon slider/select
const horizonBox=$("#horizon-lab");
if(horizonBox && window.H1_DATA){
  const buttons=$$(".seg",horizonBox), out=$("#horizon-output",horizonBox);
  function render(t){
    const d=window.H1_DATA.s06.horizons.find(x=>x.t===Number(t)); if(!d) return;
    const save=(1-d.incTok/d.transTok)*100;
    out.innerHTML=`<div class="compare-pair"><div><span>Structured state</span><strong>${d.incTok.toFixed(1)}</strong><div>${d.incremental.toFixed(1)}% accuracy</div></div><div><span>Raw transcript</span><strong>${d.transTok.toFixed(1)}</strong><div>${d.transcript.toFixed(1)}% accuracy</div></div></div>
    <p class="micro">At T=${d.t}, structured state uses ${save.toFixed(1)}% fewer query-prompt tokens in this run.</p>`;
  }
  buttons.forEach(b=>b.addEventListener("click",()=>{buttons.forEach(x=>x.classList.remove("active"));b.classList.add("active");render(b.dataset.t)}));
  render(50);
}
})();