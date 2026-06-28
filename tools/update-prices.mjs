#!/usr/bin/env node
/**
 * update-prices.mjs — Ryan Retro store price tracker
 *
 * Reads the `products` array straight out of static/shop.html, fetches the
 * current price for every store link, compares against the previous run, and
 * writes static/prices.json. The shop page (and later, handheld pages) overlay
 * that file at runtime to show live prices + "price drop" badges.
 *
 * Store handling:
 *   - Shopify stores (goretroid / ayntec / trimuistore / mangmi / powkiddy):
 *       hit the product `.js` endpoint — clean JSON, no scraping, no JS render.
 *   - Amazon / AliExpress: best-effort HTML scrape with browser-like headers.
 *       If the price can't be read (bot wall / JS-only), we KEEP the last-known
 *       price and mark it `unverified` rather than inventing or zeroing it.
 *
 * Run:  node tools/update-prices.mjs
 * No dependencies — Node 18+ (built-in fetch).
 */

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const SHOP_HTML = join(ROOT, 'static', 'shop.html');
const PRICES_JSON = join(ROOT, 'static', 'prices.json');

// Load .env (for local runs; in CI the values come from GitHub Secrets) BEFORE
// importing the API modules, which read their keys from process.env at load time.
loadEnv(join(ROOT, '.env'));
function loadEnv(path) {
  if (!existsSync(path)) return;
  for (const line of readFileSync(path, 'utf8').split('\n')) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/);
    if (!m) continue;
    const key = m[1];
    let val = m[2].replace(/^["']|["']$/g, '');
    if (process.env[key] === undefined) process.env[key] = val;
  }
}

const { getAmazonPrices, amazonConfigured } = await import('./lib/amazon.mjs');
const { getAliExpressPrices, aliexpressConfigured } = await import('./lib/aliexpress.mjs');

const SHOPIFY_HOSTS = ['goretroid.com', 'ayntec.com', 'trimuistore.com', 'mangmi.com', 'powkiddy.com'];
const UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36';
const TODAY = new Date().toISOString().slice(0, 10);

// ---------------------------------------------------------------------------
// 1. Extract the products array from shop.html (single source of truth)
// ---------------------------------------------------------------------------
function loadProducts() {
  const src = readFileSync(SHOP_HTML, 'utf8');
  const m = src.match(/const products = \[([\s\S]*?)\n\s*\];/);
  if (!m) throw new Error('Could not locate `const products = [...]` in shop.html');
  // The array body is plain JS object-literal syntax; evaluate it in isolation.
  return Function(`"use strict"; return ([${m[1]}]);`)();
}

const round2 = (n) => Math.round(n * 100) / 100;
const hostOf = (url) => { try { return new URL(url).hostname.replace(/^www\./, ''); } catch { return ''; } };

async function fetchText(url, opts = {}) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 20000);
  try {
    const res = await fetch(url, {
      redirect: 'follow',
      signal: ctrl.signal,
      headers: { 'User-Agent': UA, 'Accept-Language': 'en-US,en;q=0.9', ...(opts.headers || {}) },
    });
    return { ok: res.ok, status: res.status, body: await res.text(), finalUrl: res.url };
  } finally {
    clearTimeout(t);
  }
}

// ---------------------------------------------------------------------------
// 2. Per-store price fetchers — each returns a number (USD) or null
// ---------------------------------------------------------------------------
async function fetchShopifyPrice(url) {
  const u = new URL(url);
  u.search = '';
  u.hash = '';
  const jsUrl = u.toString().replace(/\/$/, '') + '.js';
  const { ok, body } = await fetchText(jsUrl);
  if (!ok) return null;
  let j;
  try { j = JSON.parse(body); } catch { return null; }
  const variants = Array.isArray(j.variants) ? j.variants : [];
  const inStock = variants.filter((v) => v.available);
  const pool = inStock.length ? inStock : variants;
  const cents = pool.length ? Math.min(...pool.map((v) => v.price)) : j.price;
  return typeof cents === 'number' ? round2(cents / 100) : null;
}

function isBotWall(html) {
  return /captcha|api-services-support@amazon|To discuss automated access|Enter the characters you see below|Robot Check|punish|x5sec|slidingPuzzle/i.test(html);
}

async function fetchAmazonPrice(url) {
  const { body } = await fetchText(url);
  if (!body || isBotWall(body)) return null;
  const patterns = [
    /"priceAmount":\s*([\d.]+)/,
    /class="a-offscreen">\s*\$([\d,]+\.?\d*)/,
    /"displayPrice":"\$([\d,]+\.?\d*)"/,
    /id="priceblock_ourprice"[^>]*>\s*\$([\d,]+\.?\d*)/,
  ];
  for (const re of patterns) {
    const m = body.match(re);
    if (m) {
      const n = parseFloat(m[1].replace(/,/g, ''));
      if (n > 0) return round2(n);
    }
  }
  return null;
}

async function fetchAliExpressPrice(url) {
  const { body } = await fetchText(url);
  if (!body || isBotWall(body)) return null;
  const patterns = [
    /"formatedActivityPrice":"US \$([\d.]+)"/,
    /"formatedPrice":"US \$([\d.]+)"/,
    /"minActivityAmount":\{[^}]*"value":([\d.]+)/,
    /"minAmount":\{[^}]*"value":([\d.]+)/,
  ];
  for (const re of patterns) {
    const m = body.match(re);
    if (m) {
      const n = parseFloat(m[1]);
      if (n > 0) return round2(n);
    }
  }
  return null;
}

function isShopify(host) {
  return SHOPIFY_HOSTS.some((h) => host.endsWith(h));
}

// Returns { price, trusted, merchant, merchantTrusted }.
//   trusted        = data SOURCE is authoritative (Shopify .js / official API) -> skip sanity bound
//   merchantTrusted = the SELLER is on our allowlist (Amazon.com / official / Ampown...) -> ok to alert
async function fetchRawPrice(link, api) {
  const host = hostOf(link.url);
  if (isShopify(host)) {
    return { price: await fetchShopifyPrice(link.url), trusted: true, merchant: 'Official store', merchantTrusted: true };
  }
  if (host.includes('amzn') || host.includes('amazon')) {
    const o = api.amazon.get(link.url);
    if (o) return { price: o.price, trusted: true, merchant: o.merchant, merchantTrusted: o.trusted };
    return { price: await fetchAmazonPrice(link.url), trusted: false, merchant: null, merchantTrusted: false };
  }
  if (host.includes('aliexpress')) {
    const o = api.ali.get(link.url);
    if (o) return { price: o.price, trusted: true, merchant: o.merchant, merchantTrusted: o.trusted };
    return { price: await fetchAliExpressPrice(link.url), trusted: false, merchant: null, merchantTrusted: false };
  }
  return { price: null, trusted: false, merchant: null, merchantTrusted: false };
}

// ---------------------------------------------------------------------------
// 3. Main
// ---------------------------------------------------------------------------
const prev = existsSync(PRICES_JSON) ? JSON.parse(readFileSync(PRICES_JSON, 'utf8')) : {};
const products = loadProducts();
const out = {};
const drops = [];
const failures = [];

// --- Pre-pass: pull official-API prices in batch (empty maps if unconfigured) ---
const allLinks = products.flatMap((p) => p.links || []);
const amazonLinks = allLinks.filter((l) => /amzn|amazon/.test(hostOf(l.url)));
const aliLinks = allLinks.filter((l) => hostOf(l.url).includes('aliexpress'));

console.log(`Amazon PA-API: ${amazonConfigured() ? 'configured' : 'NOT configured (scrape fallback)'}`);
console.log(`AliExpress API: ${aliexpressConfigured() ? 'configured' : 'NOT configured (scrape fallback)'}`);

const api = { amazon: new Map(), ali: new Map() };
try { api.amazon = await getAmazonPrices(amazonLinks); }
catch (e) { console.error('Amazon API failed, falling back to scrape:', e.message); }
try { api.ali = await getAliExpressPrices(aliLinks); }
catch (e) { console.error('AliExpress API failed, falling back to scrape:', e.message); }
console.log(`API prices fetched — Amazon: ${api.amazon.size}/${amazonLinks.length}, AliExpress: ${api.ali.size}/${aliLinks.length}\n`);

for (const p of products) {
  if (!p.links?.length) continue;
  out[p.id] = {};
  for (const link of p.links) {
    const prevEntry = prev[p.id]?.[link.store] || {};
    let raw = null, trusted = false, merchant = null, merchantTrusted = false;
    try {
      ({ price: raw, trusted, merchant, merchantTrusted } = await fetchRawPrice(link, api));
    } catch (e) {
      raw = null;
    }

    // Apply coupon discount if one is attached to this link.
    let current = raw;
    if (raw != null && typeof link.couponPct === 'number') {
      current = round2(raw * (1 - link.couponPct));
    }

    // Sanity bound for untrusted scrapes: a best-effort match that lands wildly
    // off the known-good reference is almost certainly a garbage regex hit
    // (e.g. Amazon returning $122,696). Reject it -> falls through to unverified.
    if (current != null && !trusted) {
      const ref = prevEntry.current ?? link.price ?? null;
      if (ref != null && (current > ref * 2.5 || current < ref * 0.4)) {
        current = null;
      }
    }

    const entry = { currency: link.currency || '$', lastChecked: TODAY };
    if (merchant) entry.merchant = merchant;
    entry.merchantTrusted = !!merchantTrusted;

    if (current == null) {
      // Could not read a price — keep last-known, flag as unverified.
      entry.current = prevEntry.current ?? link.price ?? null;
      entry.previous = prevEntry.previous ?? null;
      entry.lowest = prevEntry.lowest ?? entry.current ?? null;
      entry.status = 'unverified';
      failures.push(`${p.name} / ${link.store}`);
    } else {
      // `previous` for display falls back to the hardcoded price on first run,
      // but a DROP is only flagged against a real prior measurement (so the
      // bootstrap run doesn't fire badges just because a manual price was stale).
      const measured = prevEntry.current ?? null; // null on first ever run
      const previous = measured ?? link.price ?? null;
      entry.current = current;
      entry.previous = previous;
      entry.lowest = prevEntry.lowest != null ? Math.min(prevEntry.lowest, current) : current;
      entry.status = 'ok';
      // A real drop: cheaper than last measurement, AND from a trusted seller
      // (so we never alert on a scalper shaving a few dollars off an inflated price).
      entry.drop = measured != null && current < measured - 0.005 && merchantTrusted;
      if (entry.drop) {
        const pct = Math.round(((measured - current) / measured) * 100);
        const atLow = current <= entry.lowest + 0.005;
        drops.push({
          id: p.id, name: p.name, store: link.store, merchant: merchant || link.store,
          from: measured, to: current, pct, atLow,
          url: link.url, image: p.image, date: TODAY,
        });
      }
    }
    out[p.id][link.store] = entry;
  }
}

out._meta = { generated: new Date().toISOString(), products: products.length };
writeFileSync(PRICES_JSON, JSON.stringify(out, null, 2) + '\n');

// ---------------------------------------------------------------------------
// 4. Price history — append a dated point per link whenever the price changes.
// ---------------------------------------------------------------------------
const HISTORY_JSON = join(ROOT, 'static', 'price-history.json');
const history = existsSync(HISTORY_JSON) ? JSON.parse(readFileSync(HISTORY_JSON, 'utf8')) : {};
for (const id of Object.keys(out)) {
  if (id === '_meta') continue;
  for (const store of Object.keys(out[id])) {
    const e = out[id][store];
    if (e.current == null) continue;
    const key = `${id}|${store}`;
    const arr = history[key] || (history[key] = []);
    const last = arr[arr.length - 1];
    if (!last || last[1] !== e.current) arr.push([TODAY, e.current]);
    if (arr.length > 180) history[key] = arr.slice(-180);
  }
}
writeFileSync(HISTORY_JSON, JSON.stringify(history) + '\n');

// ---------------------------------------------------------------------------
// 5. Drops feed (rolling, newest-first, capped) + alert artifacts + Discord.
// ---------------------------------------------------------------------------
const DROPS_JSON = join(ROOT, 'static', 'drops.json');
const feed = existsSync(DROPS_JSON) ? JSON.parse(readFileSync(DROPS_JSON, 'utf8')) : [];
writeFileSync(DROPS_JSON, JSON.stringify([...drops, ...feed].slice(0, 60), null, 2) + '\n');

// Alerts use `alertDrops`. TEST_ALERT=true injects one synthetic drop so the
// full pipeline (Discord + email) can be validated WITHOUT touching drops.json.
const alertDrops = process.env.TEST_ALERT === 'true'
  ? [{ id: 'test', name: '[TEST] Price drop alert', store: 'Official', merchant: 'Official store',
       from: 99.99, to: 79.99, pct: 20, atLow: true, date: TODAY,
       url: 'https://ryanretro.com/deals',
       image: 'https://raw.githubusercontent.com/IRyzoI/ryanretro/main/images/shop/nova.png' }]
  : drops;

// This run's alert drops, for the email step + a quick CI signal.
writeFileSync(join(ROOT, 'tools', 'last-run-drops.json'), JSON.stringify(alertDrops, null, 2) + '\n');
const { postDiscord, buildEmailHtml } = await import('./lib/alerts.mjs');
writeFileSync(join(ROOT, 'tools', 'email-body.html'), buildEmailHtml(alertDrops));

if (alertDrops.length && process.env.DISCORD_WEBHOOK_URL) {
  try { await postDiscord(process.env.DISCORD_WEBHOOK_URL, alertDrops); console.log('Posted drops to Discord.'); }
  catch (e) { console.error('Discord post failed:', e.message); }
}

// ---------------------------------------------------------------------------
// 6. Console summary (shows up in the GitHub Actions log)
// ---------------------------------------------------------------------------
console.log(`\nPrice update complete — ${products.length} products, written to static/prices.json`);
console.log(`\nPRICE DROPS (${drops.length}):`);
drops.length
  ? drops.forEach((d) => console.log(`  ▼ ${d.name} / ${d.merchant}: $${d.from} → $${d.to} (-${d.pct}%)`))
  : console.log('  (none)');
console.log(`\nUNVERIFIED (${failures.length}) — kept last-known price:`);
failures.length ? failures.forEach((f) => console.log('  ? ' + f)) : console.log('  (none)');
