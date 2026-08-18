const fmt = n => `${n >= 0 ? '+' : '−'}${Math.abs(n).toFixed(2)}`;
const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));

if (window.H2_DATA) {
  init(window.H2_DATA);
} else {
  fetch('../data/core.json')
    .then(r => {
      if (!r.ok) throw new Error(`Evidence contract failed to load (${r.status})`);
      return r.json();
    })
    .then(data => init(data))
    .catch(err => {
      console.error(err);
      document.body.insertAdjacentHTML('afterbegin', `<div style="background:#ff6b57;padding:14px;font-weight:800">Could not load H2 evidence contract.</div>`);
    });
}

function init(data){
  renderStores(data.architecture);
  initTimeline(data);
  initCausalityQuestion();
  initTransplant(data.s12);
  initSpecificity(data.s12);
  initMeasurementTrap(data.s12.store_race);
  initMediation(data.s12.mediation);
  renderTaxonomy(data.taxonomy);
}

function renderStores(stores){
  const el = document.querySelector('#stores');
  el.innerHTML = stores.map((s,i) => {
    const widths = [18,68,100];
    return `<article class="store ${s.id}">
      <div class="store-head"><strong>${s.label}</strong><span class="chip ${i===2?'frozen':'open'}">${s.capacity}</span></div>
      <div class="visual"><span style="width:${widths[i]}%"></span></div>
      <div class="capacity">${s.name}</div>
    </article>`;
  }).join('');
}

function initTimeline(data){
  const {lags, constant} = data.s11;
  const slider = document.querySelector('#lag-slider');
  const ticks = document.querySelector('#lag-ticks');
  ticks.innerHTML = lags.map((l,i) => (i===0 || i===3 || i===11 || i===14 || i===16 || i===17) ? `<span>${l}</span>` : `<span>·</span>`).join('');

  function update(){
    const i = Number(slider.value);
    const lag = lags[i];
    const trace = constant.rglru_retention[i];
    const acc = constant.cloze_accuracy[i];
    document.querySelector('#lag-value').textContent = lag;
    document.querySelector('#trace-label').textContent = `${Math.round(trace*100)}%`;
    document.querySelector('#trace-meter').style.width = `${clamp(trace*100,0,100)}%`;
    document.querySelector('#cloze-label').textContent = `${Math.round(acc*100)}%`;
    document.querySelector('#cloze-meter').style.width = `${clamp(acc*100,0,100)}%`;
    document.querySelector('#trace-note').textContent = `RG-LRU normalized retention (constant filler): ${trace.toFixed(3)}.`;
    document.querySelector('#cloze-note').textContent = `Paired cloze margin: ${constant.cloze_margin[i] >= 0 ? '+' : ''}${constant.cloze_margin[i].toFixed(2)}; accuracy ${acc.toFixed(2)}.`;

    const conv = lag < 3;
    const kv = lag < 2047;
    const rglru = true;
    document.querySelector('#residency-grid').innerHTML = [
      ['CONV',conv,conv?'event still directly resident':'direct event residency gone'],
      ['KV CACHE',kv,kv?'event still directly resident':'direct event residency gone'],
      ['RG-LRU',rglru,lag===4096?'branch-specific trace remains':'continuous state evolves']
    ].map(([name,on,note]) => `<article class="residency ${on?'on':'off'}"><small>${name}</small><strong>${on?'ACTIVE TRACE':'EVICTED'}</strong><span>${note}</span></article>`).join('');
  }
  slider.addEventListener('input',update);
  update();
}

function initCausalityQuestion(){
  const feedback = document.querySelector('#causality-feedback');
  document.querySelectorAll('[data-causality]').forEach(btn => btn.addEventListener('click',() => {
    if(btn.dataset.causality === 'right'){
      feedback.className = 'feedback good';
      feedback.innerHTML = '<strong>Correct.</strong> Persistent separation is observational. S12 must intervene on the state to test whether the separating direction actually changes later computation.';
    } else {
      feedback.className = 'feedback bad';
      feedback.innerHTML = '<strong>That inference is too strong.</strong> Two states can differ in directions that later computation ignores. Difference is cheap; causation requires intervention.';
    }
  }));
}

function initTransplant(s12){
  const conditions = s12.conditions.filter(c => ['match','unrelated','permuted','noise','kv','whole'].includes(c.id));
  const controls = document.querySelector('#transplant-controls');
  controls.innerHTML = conditions.map((c,i) => `<button data-condition="${c.id}" class="${i===0?'active':''}">${c.short}</button>`).join('');
  const byId = Object.fromEntries(conditions.map(c => [c.id,c]));

  function update(id){
    controls.querySelectorAll('button').forEach(b=>b.classList.toggle('active',b.dataset.condition===id));
    const c = byId[id];
    document.querySelector('#transplant-estimate').textContent = fmt(c.estimate);
    document.querySelector('#transplant-ci').textContent = `95% CI [${fmt(c.ci[0])}, ${fmt(c.ci[1])}]`;
    const lo=-60, hi=175;
    document.querySelector('#ci-range').style.left = `${(clamp(c.ci[0],lo,hi)-lo)/(hi-lo)*100}%`;
    document.querySelector('#ci-range').style.width = `${(clamp(c.ci[1],lo,hi)-clamp(c.ci[0],lo,hi))/(hi-lo)*100}%`;
    document.querySelector('#ci-point').style.left = `${(clamp(c.estimate,lo,hi)-lo)/(hi-lo)*100}%`;
    const explainers = {
      match:'Matching RG-LRU transplantation strongly moves the recipient output along the donor trajectory.',
      unrelated:'A different structured history also moves the recipient substantially—one reason S12 cannot be summarized as “only the correct memory works.”',
      permuted:'A second cross-history mapping again produces substantial donor-directed steering.',
      noise:'Matched-magnitude noise perturbs the system, but much less than structured historical states.',
      kv:'Swapping the KV store also produces large absolute displacement at 2W.',
      whole:'Moving the whole state is the positive-control ceiling: all physical stores come from the donor trajectory.'
    };
    document.querySelector('#transplant-explainer').textContent = explainers[id];
  }
  controls.addEventListener('click',e => { if(e.target.matches('button')) update(e.target.dataset.condition); });
  update('match');
}

function initSpecificity(s12){
  const order=['match','unrelated','permuted','noise'];
  const rows = order.map(id=>s12.conditions.find(c=>c.id===id));
  const max = Math.max(...rows.map(r=>r.estimate));
  document.querySelector('#specificity-bars').innerHTML = rows.map(r=>`<div class="bar-row ${r.id}"><label>${r.short}</label><div class="bar-track"><span style="width:${r.estimate/max*100}%"></span></div><strong>${fmt(r.estimate)}</strong></div>`).join('');
  const fb=document.querySelector('#specificity-feedback');
  document.querySelector('#specificity-choices').addEventListener('click',e=>{
    if(!e.target.matches('button')) return;
    if(e.target.dataset.answer==='right'){
      fb.className='feedback good';
      fb.innerHTML=`<strong>That is the frozen interpretation.</strong> Matching history adds ${fmt(s12.specificity.matching_minus_unrelated)} over the unrelated donor, 95% CI [${fmt(s12.specificity.matching_minus_unrelated_ci[0])}, ${fmt(s12.specificity.matching_minus_unrelated_ci[1])}], while cross-history steering remains large.`;
    } else {
      fb.className='feedback bad';
      fb.innerHTML='<strong>Too simple.</strong> Structured cross-history donors steer much more than matched noise. The mystery is not whether the correct history matters; it is what the shared structured component represents.';
    }
  });
}

function initMeasurementTrap(race){
  document.querySelector('#alpha-kv').textContent=`${(race.alpha_kv*100).toFixed(1)}%`;
  document.querySelector('#alpha-rglru').textContent=`${(race.alpha_rglru*100).toFixed(1)}%`;
  document.querySelector('#declare-kv').addEventListener('click',()=>{
    const verdict=document.querySelector('#measurement-verdict');
    verdict.className='verdict wrong';
    verdict.innerHTML='WRONG<br>MEASUREMENT';
    document.querySelector('#absolute-race').classList.remove('hidden');
  });
}

function initMediation(rows){
  const slider=document.querySelector('#mediation-slider');
  function update(){
    const r=rows[Number(slider.value)];
    const pct=(r.m_post+1)/2*100;
    document.querySelector('#mediation-point').style.left=`${pct}%`;
    document.querySelector('#mediation-value').textContent=`M = ${r.m_post.toFixed(3).replace('-', '−')}`;
    document.querySelector('#mediation-copy').textContent = r.tokens_after_graft===512
      ? 'After 512 tokens, newly written KV remains strongly recipient-anchored.'
      : 'After 2048 tokens—a full local-window turnover—the mean is less recipient-anchored, but remains on the recipient side.';
  }
  slider.addEventListener('input',update); update();
}

function renderTaxonomy(rows){
  document.querySelector('#taxonomy').innerHTML=rows.map(r=>{
    const cls=r.status==='LIVE'?'live':r.status==='OPEN QUESTION'?'open':'frozen';
    return `<article class="door ${cls}"><div><small>${r.status}</small><strong>${r.property}</strong></div><div class="answer">${r.answer}</div><p>${r.note}</p></article>`;
  }).join('');
}
