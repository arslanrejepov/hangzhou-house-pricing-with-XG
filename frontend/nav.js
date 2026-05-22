// nav.js — shared hamburger drawer + active nav link logic
(function () {
  const menuBtn  = document.getElementById('menuBtn');
  const backdrop = document.getElementById('drawerBackdrop');
  const drawer   = document.getElementById('drawer');
  const closeBtn = document.getElementById('drawerClose');

  function openDrawer() {
    backdrop.classList.add('open');
    drawer.classList.add('open');
    document.body.style.overflow = 'hidden';
  }
  function closeDrawer() {
    backdrop.classList.remove('open');
    drawer.classList.remove('open');
    document.body.style.overflow = '';
  }

  if (menuBtn)  menuBtn.addEventListener('click', openDrawer);
  if (backdrop) backdrop.addEventListener('click', closeDrawer);
  if (closeBtn) closeBtn.addEventListener('click', closeDrawer);

  // Mark active nav center + drawer link
  const page = location.pathname.split('/').pop() || 'forecast.html';
  document.querySelectorAll('[data-page]').forEach(a => {
    if (a.dataset.page === page) a.classList.add('active');
  });
})();