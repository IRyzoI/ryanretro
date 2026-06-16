/*
 * Shared site chrome loader for Ryan Retro.
 *
 * Each page drops placeholders where the header / footer should appear:
 *     <div data-include="header"></div>
 *     <div data-include="footer"></div>
 * and loads this script once (in <head>, with `defer`):
 *     <script defer src="/static/partials/include.js"></script>
 *
 * This fetches the matching partial from /static/partials/<name>.html and
 * swaps it in, then re-renders Lucide icons, wires up the mobile menu, and
 * highlights the active nav link. Edit header.html / footer.html once and
 * every page that loads this script picks up the change.
 */
(function () {
    const BASE = '/static/partials/';

    async function inject(el) {
        const name = el.getAttribute('data-include');
        try {
            const res = await fetch(BASE + name + '.html', { cache: 'no-cache' });
            if (!res.ok) throw new Error(res.status + ' ' + res.statusText);
            el.outerHTML = await res.text();
        } catch (err) {
            console.error('[include] failed to load partial "' + name + '"', err);
        }
    }

    function wireMobileMenu() {
        const btn = document.getElementById('mobile-menu-btn');
        const menu = document.getElementById('mobile-menu');
        if (btn && menu) {
            btn.addEventListener('click', () => menu.classList.toggle('hidden'));
        }
    }

    // Highlight the nav link that matches the current page.
    function highlightActiveNav() {
        const path = (location.pathname.replace(/\/+$/, '') || '/');
        const links = document.querySelectorAll('header nav a[href], #mobile-menu a[href]');
        links.forEach((a) => {
            const href = (a.getAttribute('href') || '').replace(/\/+$/, '') || '/';
            if (href !== '/' && (path === href || path.startsWith(href + '/'))) {
                a.classList.remove('text-brand-light');
                a.classList.add('text-white');
                a.setAttribute('aria-current', 'page');
            }
        });
    }

    async function boot() {
        const placeholders = Array.prototype.slice.call(
            document.querySelectorAll('[data-include]')
        );
        await Promise.all(placeholders.map(inject));

        if (window.lucide && typeof window.lucide.createIcons === 'function') {
            window.lucide.createIcons();
        }
        wireMobileMenu();
        highlightActiveNav();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', boot);
    } else {
        boot();
    }
})();
