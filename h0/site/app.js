const $=(s,r=document)=>r.querySelector(s), $$=(s,r=document)=>[...r.querySelectorAll(s)];
const G=window.GLOSSARY||{}, D=window.H0_DATA||{};

function openDrawer(){
  $("#glossary-drawer")?.classList.add("open");
  $("#glossary-drawer")?.setAttribute("aria-hidden","false");
  if($("#scrim")) $("#scrim").hidden=false;
}
function closeDrawer(){
  $("#glossary-drawer")?.classList.remove("open");
  $("#glossary-drawer")?.setAttribute("aria-hidden","true");
  if($("#scrim")) $("#scrim").hidden=true;
}
function buildGlossary(){
  const list=$("#glossary-list"); if(!list) return;
  const entries=Object.entries(G).sort((a,b)=>a[1].term.localeCompare(b[1].term));
  function render(filter=""){
    const q=filter.trim().toLowerCase();
    list.innerHTML="";
    entries.filter(([_,v])=>!q||v.term.toLowerCase().includes(q)||v.short.toLowerCase().includes(q)||v.long.toLowerCase().includes(q))
      .forEach(([k,v])=>{
        const el=document.createElement("div"); el.className="glossary-entry"; el.dataset.key=k;
        el.innerHTML=`<h3>${v.term}</h3><p><strong>${v.short}</strong></p><p>${v.long}</p>`;
        list.appendChild(el);
      });
  }
  render();
  $("#glossary-search")?.addEventListener("input",e=>render(e.target.value));
}
function positionTooltip(el, tip){
  const r=el.getBoundingClientRect(), w=Math.min(330,window.innerWidth-24);
  tip.style.width=w+"px";
  let left=Math.min(window.innerWidth-w-12,Math.max(12,r.left));
  let top=r.bottom+8;
  if(top+tip.offsetHeight>window.innerHeight-10) top=r.top-tip.offsetHeight-8;
  tip.style.left=left+"px"; tip.style.top=Math.max(8,top)+"px";
}
function initTerms(){
  const tip=$("#tooltip"); if(!tip) return;
  $$(".term").forEach(el=>{
    const show=()=>{
      const g=G[el.dataset.term]; if(!g) return;
      tip.innerHTML=`<strong>${g.term}</strong><br>${g.short}<div style="margin-top:.35rem;font-size:.78rem;opacity:.75">Click for the full glossary entry.</div>`;
      tip.hidden=false; requestAnimationFrame(()=>positionTooltip(el,tip));
    };
    const hide=()=>{tip.hidden=true};
    el.addEventListener("mouseenter",show); el.addEventListener("mouseleave",hide);
    el.addEventListener("focus",show); el.addEventListener("blur",hide);
    el.addEventListener("click",()=>{
      hide(); openDrawer();
      setTimeout(()=>{
        const entry=$(`.glossary-entry[data-key="${el.dataset.term}"]`);
        entry?.scrollIntoView({behavior:"smooth",block:"start"});
      },80);
    });
  });
}
function initChrome(){
  $("#glossary-open")?.addEventListener("click",openDrawer);
  $("#floating-help")?.addEventListener("click",openDrawer);
  $("#glossary-close")?.addEventListener("click",closeDrawer);
  $("#scrim")?.addEventListener("click",closeDrawer);
  document.addEventListener("keydown",e=>{if(e.key==="Escape") closeDrawer()});
  $("#plain-toggle")?.addEventListener("click",e=>{
    const recap=$("#plain-recap"); if(!recap) return;
    recap.hidden=!recap.hidden; e.currentTarget.classList.toggle("active",!recap.hidden);
  });
}

function initStatsLab(){
  const slider=$("#brier-slider"); if(!slider) return;
  let outcome=1;
  const update=()=>{
    const p=Number(slider.value)/100;
    $("#brier-prob").textContent=Math.round(p*100)+"%";
    $("#brier-result").textContent=((p-outcome)**2).toFixed(3);
  };
  slider.addEventListener("input",update);
  $$(".seg").forEach(b=>b.addEventListener("click",()=>{
    $$(".seg").forEach(x=>x.classList.remove("active")); b.classList.add("active"); outcome=Number(b.dataset.outcome); update();
  }));
  const pairs=[[82,61],[66,78],[91,34],[54,49],[71,70],[43,57]];
  let i=0;
  const draw=()=>{
    const [c,w]=pairs[i++%pairs.length];
    $("#auc-correct").textContent=c+"%"; $("#auc-wrong").textContent=w+"%";
    const good=c>w, f=$("#auc-verdict");
    f.textContent=good?"Correct trial ranked higher → a Type-2 “win”.":"Incorrect trial ranked higher → a Type-2 “loss”.";
    f.classList.toggle("bad",!good);
  };
  $("#auc-new")?.addEventListener("click",draw);
}

function initObservers(){
  const buttons=$("#observer-buttons"); if(!buttons||!D.observers) return;
  const view=$("#observer-view"), explain=$("#observer-explain");
  D.observers.forEach((o,i)=>{
    const b=document.createElement("button"); b.className="chip"+(i===0?" active":""); b.textContent=o.label;
    b.addEventListener("click",()=>render(o,b)); buttons.appendChild(b);
  });
  function render(o,b){
    $$(".chip",buttons).forEach(x=>x.classList.remove("active")); b.classList.add("active");
    view.innerHTML=`<div class="eyebrow">What this evaluator sees</div><h3>${o.label}</h3><ul>${o.visible.map(x=>`<li>${x}</li>`).join("")}</ul><div class="eyebrow">run_005 AUROC2</div><div class="big-metric">${o.score.toFixed(3)}</div><div class="metric-bar"><div style="width:${o.score*100}%"></div></div>`;
    explain.innerHTML=`<div class="eyebrow">Why it exists</div><p><strong>${o.question}</strong></p><p>${o.controls}</p>`;
  }
  render(D.observers[0],buttons.firstChild);
}

function initMuseum(){
  const grid=$("#museum-grid"); if(!grid||!D.museum) return;
  D.museum.forEach(m=>{
    const el=document.createElement("article"); el.className="museum-card";
    el.innerHTML=`<button type="button"><strong>${m.title}</strong><br><span class="tag">${m.status}</span></button><div class="body"><p><strong>Why it looked plausible:</strong> ${m.before}</p><p><strong>What changed:</strong> ${m.after}</p><p><strong>General lesson:</strong> ${m.lesson}</p></div>`;
    $("button",el).addEventListener("click",()=>el.classList.toggle("open")); grid.appendChild(el);
  });
}

function initPai(){
  const box=$("#pai-rows"); if(!box||!D.reference) return;
  const r=D.reference, rows=[["Immediate Self",r.self,false],["Visible Answer",r.visible,true],["Reconstruction",r.reconstruction,false],["Input Only",r.inputOnly,false]];
  rows.forEach(([name,v,max])=>{
    const el=document.createElement("div"); el.className="pai-row";
    el.innerHTML=`<div>${name}${max?' <span class="pai-max">← strongest</span>':''}</div><div class="metric-bar" style="margin:0"><div style="width:${v*100}%;background:${name==="Immediate Self"?"var(--wine)":"var(--teal)"}"></div></div><div>${v.toFixed(3)}</div>`;
    box.appendChild(el);
  });
  $("#pai-equation").textContent=`PAI = Self - max(Visible Answer, Reconstruction, Input Only)
    = 0.517 - max(0.678, 0.573, 0.527)
    = 0.517 - 0.678
    = -0.161

95% bootstrap CI: [-0.428, +0.055]
Meaningful-positive SESOI: +0.10`;
}

function initCheckpoint(){
  $$(".checkpoint").forEach(cp=>{
    $$("button",cp).forEach(b=>b.addEventListener("click",()=>{
      const good=b.dataset.choice===cp.dataset.answer, f=$(".checkpoint-feedback",cp);
      f.className="checkpoint-feedback feedback"+(good?"":" bad");
      f.textContent=good
        ?"Yes. This wording keeps the model, task, comparator set, uncertainty, and claim ceiling explicit."
        :"Not quite. The H0 result is narrower: it does not establish a universal introspection, intelligence, exact-sign, or consciousness conclusion.";
    }));
  });
}

function initStress(){
  const modelButtons=$("#model-buttons"); if(!modelButtons||!D.models) return;
  const stage=$("#model-stage");
  D.models.forEach((m,i)=>{
    const b=document.createElement("button"); b.className="chip"+(i===1?" active":""); b.textContent=m.name;
    b.addEventListener("click",()=>renderModel(m,b)); modelButtons.appendChild(b);
  });
  function renderModel(m,b){
    $$(".chip",modelButtons).forEach(x=>x.classList.remove("active")); b.classList.add("active");
    stage.innerHTML=`<div class="evidence-card"><div class="tag">${m.status}</div><h3>${m.name}</h3><div class="eyebrow">First-order accuracy</div><div class="big-metric">${(m.accuracy*100).toFixed(1)}%</div><div class="metric-bar"><div style="width:${m.accuracy*100}%"></div></div></div><div class="evidence-card"><div class="eyebrow">Immediate Self AUROC2</div>${m.self==null?'<div class="big-metric">N/A</div><p><strong>No errors exist.</strong> Correct-vs-incorrect discrimination cannot be estimated.</p>':`<div class="big-metric">${m.self.toFixed(3)}</div><div class="metric-bar"><div style="width:${m.self*100}%;background:var(--wine)"></div></div>`}<p>${m.detail}</p></div>`;
  }
  renderModel(D.models[1],modelButtons.children[1]);

  const reactButtons=$("#reactivity-buttons"), reactStage=$("#reactivity-stage");
  D.reactivity.forEach((m,i)=>{
    const b=document.createElement("button"); b.className="chip"+(i===1?" active":""); b.textContent=m.name;
    b.addEventListener("click",()=>renderReact(m,b)); reactButtons.appendChild(b);
  });
  function renderReact(m,b){
    $$(".chip",reactButtons).forEach(x=>x.classList.remove("active")); b.classList.add("active");
    const cells=Array.from({length:40},(_,i)=>`<span class="${i<m.changed?"changed":"same"}" title="${i<m.changed?"Answer changed":"Same answer"}"></span>`).join("");
    reactStage.innerHTML=`<div class="observer-demo"><div><div class="eyebrow">Answer only</div><div class="big-metric">${(m.answerOnly*100).toFixed(1)}%</div></div><div><div class="eyebrow">Answer + confidence</div><div class="big-metric">${(m.answerConf*100).toFixed(1)}%</div></div></div><div class="answer-strip">${cells}</div><p><strong>${m.changed}/40 selected answers changed.</strong> Exact-answer concordance ${(m.concordance*100).toFixed(1)}%; paired McNemar p = ${m.p.toFixed(4)}.</p><p class="micro">The test does not resolve a net accuracy change, but the item-level choice policy clearly changes.</p>`;
  }
  renderReact(D.reactivity[1],reactButtons.children[1]);
}

document.addEventListener("DOMContentLoaded",()=>{
  buildGlossary(); initTerms(); initChrome(); initStatsLab(); initObservers(); initMuseum(); initPai(); initCheckpoint(); initStress();
});
