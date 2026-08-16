(() => {
 const current = document.body.dataset.page;
 document.querySelectorAll('.side-nav a').forEach(a => {
   if (a.getAttribute('href') === current) a.classList.add('active');
 });
 const recap = document.getElementById('plain-recap');
 document.getElementById('plain-toggle')?.addEventListener('click', () => {
   recap.hidden = !recap.hidden;
 });
 const dlg = document.getElementById('glossary-dialog');
 const body = document.getElementById('glossary-body');
 if (body) body.innerHTML = window.H0_GLOSSARY_HTML || '';
 document.getElementById('glossary-open')?.addEventListener('click', () => dlg?.showModal());
 document.getElementById('glossary-close')?.addEventListener('click', () => dlg?.close());
})();
