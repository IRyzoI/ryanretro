/**
 * AliExpress Affiliate (Open Platform) API — price lookup.
 *
 * Env vars (set in .env locally, GitHub Secrets in CI):
 *   ALIEXPRESS_APP_KEY        Open Platform app key
 *   ALIEXPRESS_APP_SECRET     Open Platform app secret
 *   ALIEXPRESS_TRACKING_ID    affiliate tracking id (default: default)
 *
 * If keys are absent, aliexpressConfigured() returns false and
 * getAliExpressPrices() resolves to an empty map — caller falls back to scraping.
 *
 * Uses the system API gateway (api-sg.aliexpress.com/sync) with HMAC-SHA256
 * request signing, method aliexpress.affiliate.productdetail.get.
 */
import crypto from 'node:crypto';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

// Trusted-store config (AliExpress section) — focus on Ampown / official stores.
const TRUSTED_PATH = join(dirname(fileURLToPath(import.meta.url)), '..', 'trusted-sellers.json');
const TRUSTED = existsSync(TRUSTED_PATH) ? JSON.parse(readFileSync(TRUSTED_PATH, 'utf8')).aliexpress || {} : {};
const ALLOW_STORE_IDS = new Set((TRUSTED.allowStoreIds || []).map(String));
const ALLOW_NAMES = new Set((TRUSTED.allowNames || []).map((n) => n.toLowerCase()));
const MIN_RATING = TRUSTED.minRating ?? 85;
const MIN_SALES = TRUSTED.minSales ?? 100;

// A store is trusted if it's explicitly allowlisted, has "official" in its name,
// OR earns it on reputation: rating >= minRating% AND sales >= minSales. This
// auto-trusts reputable stores while ignoring generic "Store 13840193" sellers.
function isTrustedStore(storeId, storeName, rating, sales) {
  if (storeId && ALLOW_STORE_IDS.has(String(storeId))) return true;
  const name = (storeName || '').toLowerCase();
  for (const allowed of ALLOW_NAMES) if (allowed && name.includes(allowed)) return true;
  if (TRUSTED.trustOfficialKeyword && /\bofficial\b/.test(name)) return true;
  if (rating != null && sales != null && rating >= MIN_RATING && sales >= MIN_SALES) return true;
  return false;
}

const APP_KEY = process.env.ALIEXPRESS_APP_KEY;
const APP_SECRET = process.env.ALIEXPRESS_APP_SECRET;
const TRACKING_ID = process.env.ALIEXPRESS_TRACKING_ID || 'default';
// Ship-to country for pricing. AliExpress geo-prices, so we fetch US prices to
// match the (mostly US) audience — NOT the price the operator sees in their own
// country. Override with ALIEXPRESS_COUNTRY if your audience is elsewhere.
const COUNTRY = process.env.ALIEXPRESS_COUNTRY || 'US';
const GATEWAY = process.env.ALIEXPRESS_GATEWAY || 'https://api-sg.aliexpress.com/sync';
const UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36';

export function aliexpressConfigured() {
  return Boolean(APP_KEY && APP_SECRET);
}

const ITEM_RE = /\/item\/(?:[\w-]+\/)?(\d{6,})\.html/;

/**
 * Resolve an s.click link to the CANONICAL product id (1005...). AliExpress
 * rewrites legacy item ids to the canonical one only after the full redirect
 * chain, so we follow all redirects and read the final URL (the affiliate API
 * returns 0 records for the legacy id). Manual hop-scan is a fallback.
 */
async function resolveItemId(url) {
  try {
    const res = await fetch(url, { redirect: 'follow', headers: { 'User-Agent': UA } });
    const m = (res.url || '').match(ITEM_RE);
    if (m) return m[1];
  } catch { /* fall through to manual */ }

  let cur = url;
  for (let i = 0; i < 5; i++) {
    let res;
    try {
      res = await fetch(cur, { redirect: 'manual', headers: { 'User-Agent': UA } });
    } catch {
      return null;
    }
    const loc = res.headers.get('location');
    if (!loc) {
      const m = (res.url || '').match(ITEM_RE);
      return m ? m[1] : null;
    }
    cur = new URL(loc, cur).toString();
  }
  return cur.match(ITEM_RE)?.[1] || null;
}

/** TOP/ISV signature: HMAC-SHA256 over the sorted key+value concatenation, hex upper. */
function sign(params) {
  const base = Object.keys(params).sort().map((k) => k + params[k]).join('');
  return crypto.createHmac('sha256', APP_SECRET).update(base, 'utf8').digest('hex').toUpperCase();
}

async function productDetail(ids) {
  const params = {
    app_key: APP_KEY,
    method: 'aliexpress.affiliate.productdetail.get',
    sign_method: 'sha256',
    timestamp: String(Date.now()),
    format: 'json',
    v: '2.0',
    // business params
    product_ids: ids.join(','),
    target_currency: 'USD',
    target_language: 'EN',
    country: COUNTRY,
    tracking_id: TRACKING_ID,
    fields: 'product_id,target_sale_price,target_sale_price_currency,sale_price,product_title,shop_id,shop_name,evaluate_rate,lastest_volume',
  };
  params.sign = sign(params);

  const res = await fetch(GATEWAY, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded', 'User-Agent': UA },
    body: new URLSearchParams(params).toString(),
  });
  const text = await res.text();
  let json;
  try { json = JSON.parse(text); } catch { json = null; }
  if (!json) throw new Error(`AliExpress API non-JSON response: ${text.slice(0, 200)}`);
  const err = json.error_response;
  if (err) throw new Error(`AliExpress API error: ${err.code} ${err.msg || err.sub_msg || ''}`);
  return json;
}

const chunk = (arr, n) => Array.from({ length: Math.ceil(arr.length / n) }, (_, i) => arr.slice(i * n, i * n + n));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Pull the product array out of the (somewhat nested) response shape. */
function extractProducts(json) {
  const resp = json?.aliexpress_affiliate_productdetail_get_response || json;
  const result = resp?.resp_result?.result || resp?.result;
  const products = result?.products?.product;
  return Array.isArray(products) ? products : products ? [products] : [];
}

/**
 * @param {Array<{url:string}>} links  AliExpress store links from products
 * @returns {Promise<Map<string, number>>}  url -> price (USD). Empty if unconfigured.
 */
export async function getAliExpressPrices(links) {
  const out = new Map();
  if (!aliexpressConfigured() || !links.length) return out;

  // 1. Resolve every link to a numeric product id.
  const urlToId = new Map();
  for (const l of links) {
    const id = await resolveItemId(l.url);
    if (id) urlToId.set(l.url, id);
  }
  const ids = [...new Set(urlToId.values())];
  if (!ids.length) return out;

  // 2. Fetch product detail (the endpoint accepts multiple ids; chunk to be safe).
  const idToOffer = new Map();
  for (const group of chunk(ids, 20)) {
    const json = await productDetail(group);
    for (const p of extractProducts(json)) {
      const raw = p.target_sale_price ?? p.sale_price;
      const price = typeof raw === 'string' ? parseFloat(raw) : raw;
      if (price > 0) {
        const rating = p.evaluate_rate != null ? parseFloat(String(p.evaluate_rate)) : null;
        const sales = p.lastest_volume != null ? Number(p.lastest_volume) : null;
        idToOffer.set(String(p.product_id), {
          price: Math.round(price * 100) / 100,
          merchant: p.shop_name || 'AliExpress',
          trusted: isTrustedStore(p.shop_id, p.shop_name, rating, sales),
          rating,
          sales,
        });
      }
    }
    await sleep(500);
  }

  // 3. Map offers back onto the original link URLs.
  for (const [url, id] of urlToId) {
    if (idToOffer.has(id)) out.set(url, idToOffer.get(id));
  }
  return out;
}
