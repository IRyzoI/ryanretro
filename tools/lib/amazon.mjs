/**
 * Amazon Creators API — price lookup (replaces PA-API v5, retired May 2026).
 *
 * Auth: OAuth2 client_credentials (Login with Amazon) -> bearer token.
 * Data: POST {catalog}/getItems with itemIds (ASINs), price at
 *       offersV2.listings[0].price.{amount|displayAmount}.
 *
 * Env vars (set in .env locally, GitHub Secrets in CI):
 *   AMAZON_CLIENT_ID       OAuth2 client id (amzn1.application-oa2-client...)
 *   AMAZON_CLIENT_SECRET   OAuth2 client secret (amzn1.oa2-cs.v1...)
 *   AMAZON_PARTNER_TAG     Associates tag (default: ryanretro-20)
 *   AMAZON_MARKETPLACE     default www.amazon.com
 *   AMAZON_TOKEN_URL       default https://api.amazon.com/auth/o2/token
 *   AMAZON_CATALOG_URL     default https://creatorsapi.amazon/catalog/v1/getItems
 *   AMAZON_SCOPE           default creatorsapi::default  (v3.x credential scope)
 *
 * If keys are absent, amazonConfigured() returns false and getAmazonPrices()
 * resolves to an empty map — the caller then falls back to best-effort scraping.
 */
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

// Trusted-seller config (Amazon section). Controls which merchants we price from.
const TRUSTED_PATH = join(dirname(fileURLToPath(import.meta.url)), '..', 'trusted-sellers.json');
const TRUSTED = existsSync(TRUSTED_PATH) ? JSON.parse(readFileSync(TRUSTED_PATH, 'utf8')).amazon || {} : {};
const ALLOW_IDS = new Set(TRUSTED.allowIds || []);
const ALLOW_NAMES = new Set((TRUSTED.allowNames || []).map((n) => n.toLowerCase()));

function isTrustedMerchant(merchant) {
  if (!merchant) return false;
  if (merchant.id && ALLOW_IDS.has(merchant.id)) return true;
  const name = (merchant.name || '').toLowerCase();
  if (ALLOW_NAMES.has(name)) return true;
  if (TRUSTED.trustOfficialKeyword && /\bofficial\b/.test(name)) return true;
  return false;
}

const CLIENT_ID = process.env.AMAZON_CLIENT_ID;
const CLIENT_SECRET = process.env.AMAZON_CLIENT_SECRET;
const PARTNER_TAG = process.env.AMAZON_PARTNER_TAG || 'ryanretro-20';
const MARKETPLACE = process.env.AMAZON_MARKETPLACE || 'www.amazon.com';
const TOKEN_URL = process.env.AMAZON_TOKEN_URL || 'https://api.amazon.com/auth/o2/token';
const CATALOG_URL = process.env.AMAZON_CATALOG_URL || 'https://creatorsapi.amazon/catalog/v1/getItems';
const SCOPE = process.env.AMAZON_SCOPE || 'creatorsapi::default';
const UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36';

export function amazonConfigured() {
  return Boolean(CLIENT_ID && CLIENT_SECRET);
}

// --- OAuth2 token (cached in-process, refreshed with a 60s safety buffer) ---
let tokenCache = { value: null, expiresAt: 0 };

async function getToken() {
  if (tokenCache.value && Date.now() < tokenCache.expiresAt) return tokenCache.value;
  const res = await fetch(TOKEN_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'client_credentials',
      client_id: CLIENT_ID,
      client_secret: CLIENT_SECRET,
      scope: SCOPE,
    }).toString(),
  });
  const text = await res.text();
  let json;
  try { json = JSON.parse(text); } catch { json = null; }
  if (!res.ok || !json?.access_token) {
    const msg = json ? (json.error_description || json.error || JSON.stringify(json)) : text.slice(0, 200);
    throw new Error(`OAuth token ${res.status} — ${msg}`);
  }
  tokenCache = {
    value: json.access_token,
    expiresAt: Date.now() + (json.expires_in || 3600) * 1000 - 60000,
  };
  return tokenCache.value;
}

const ASIN_RE = /\/(?:dp|gp\/product|gp\/aw\/d)\/([A-Z0-9]{10})/;

/** Follow amzn.to / amazon redirects (manual, lightweight) to extract the ASIN. */
async function resolveAsin(url) {
  let cur = url;
  for (let i = 0; i < 5; i++) {
    const direct = cur.match(ASIN_RE);
    if (direct) return direct[1];
    let res;
    try {
      res = await fetch(cur, { redirect: 'manual', headers: { 'User-Agent': UA } });
    } catch {
      return null;
    }
    const loc = res.headers.get('location');
    if (!loc) {
      const m = (res.url || '').match(ASIN_RE);
      return m ? m[1] : null;
    }
    cur = new URL(loc, cur).toString();
  }
  const m = cur.match(ASIN_RE);
  return m ? m[1] : null;
}

/** Call getItems for up to 10 ASINs; returns the parsed JSON response. */
async function getItems(asins) {
  const token = await getToken();
  const res = await fetch(CATALOG_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
      'x-marketplace': MARKETPLACE,
    },
    body: JSON.stringify({
      itemIds: asins,
      itemIdType: 'ASIN',
      partnerTag: PARTNER_TAG,
      partnerType: 'Associates',
      marketplace: MARKETPLACE,
      resources: [
        'offersV2.listings.price',
        'offersV2.listings.condition',
        'offersV2.listings.merchantInfo',
        'offersV2.listings.availability',
      ],
    }),
  });
  const text = await res.text();
  let json;
  try { json = JSON.parse(text); } catch { json = null; }
  if (!res.ok) {
    const msg = json?.errors?.map((e) => e.code + ': ' + e.message).join('; ') || text.slice(0, 300);
    throw new Error(`Creators getItems ${res.status} — ${msg}`);
  }
  return json;
}

const amountOf = (listing) => {
  const money = listing?.price?.money || listing?.price;
  if (!money) return null;
  if (typeof money.amount === 'number' && money.amount > 0) return Math.round(money.amount * 100) / 100;
  const disp = money.displayAmount || money.value;
  if (typeof disp === 'string') {
    const n = parseFloat(disp.replace(/[^0-9.]/g, ''));
    if (n > 0) return Math.round(n * 100) / 100;
  }
  return null;
};
const isNew = (l) => (l?.condition?.value || 'New') === 'New';
const inStock = (l) => !l?.availability?.type || /^IN_STOCK/.test(l.availability.type);

/**
 * Choose the best offer for an item:
 *   1. cheapest New + in-stock offer from a TRUSTED seller (Amazon / official / allowlisted)
 *   2. else the buy-box (or cheapest New) offer, flagged untrusted so alerts can skip it
 * Returns { price, merchant, trusted } or null.
 */
function offerFromItem(item) {
  const listings = (item?.offersV2?.listings || item?.offers?.listings || []).filter((l) => amountOf(l) != null);
  if (!listings.length) return null;
  const fresh = listings.filter((l) => isNew(l) && inStock(l));
  const pool = fresh.length ? fresh : listings;

  const trusted = pool
    .filter((l) => isTrustedMerchant(l.merchantInfo))
    .sort((a, b) => amountOf(a) - amountOf(b));
  if (trusted.length) {
    return { price: amountOf(trusted[0]), merchant: trusted[0].merchantInfo?.name || 'Amazon', trusted: true };
  }
  const pick = pool.find((l) => l.isBuyBoxWinner) || pool[0];
  return { price: amountOf(pick), merchant: pick.merchantInfo?.name || 'third-party', trusted: false };
}

/** All New/in-stock seller offers for an item, cheapest first (top 6), for the per-store dropdown. */
function offersFromItem(item) {
  const listings = (item?.offersV2?.listings || item?.offers?.listings || [])
    .filter((l) => amountOf(l) != null && isNew(l) && inStock(l));
  return listings
    .map((l) => ({
      price: amountOf(l),
      merchant: l.merchantInfo?.name || 'Amazon',
      trusted: isTrustedMerchant(l.merchantInfo),
    }))
    .sort((a, b) => a.price - b.price)
    .slice(0, 6);
}

const chunk = (arr, n) => Array.from({ length: Math.ceil(arr.length / n) }, (_, i) => arr.slice(i * n, i * n + n));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * @param {Array<{url:string}>} links  Amazon store links from products
 * @returns {Promise<Map<string, {price:number, merchant:string, trusted:boolean}>>}
 *          url -> offer. Empty if unconfigured.
 */
export async function getAmazonPrices(links) {
  const out = new Map();
  if (!amazonConfigured() || !links.length) return out;

  // 1. Resolve every link to an ASIN.
  const urlToAsin = new Map();
  for (const l of links) {
    const asin = await resolveAsin(l.url);
    if (asin) urlToAsin.set(l.url, asin);
  }
  const asins = [...new Set(urlToAsin.values())];
  if (!asins.length) return out;

  // 2. Batch getItems (max 10 ASINs/request).
  const asinToOffer = new Map();
  for (const group of chunk(asins, 10)) {
    const json = await getItems(group);
    const items = json?.itemsResult?.items || json?.items || [];
    for (const it of items) {
      const asin = it.asin || it.ASIN;
      const offer = offerFromItem(it);
      if (asin && offer?.price != null) asinToOffer.set(asin, { ...offer, offers: offersFromItem(it) });
    }
    await sleep(1100);
  }

  // 3. Map offers back onto the original link URLs.
  for (const [url, asin] of urlToAsin) {
    if (asinToOffer.has(asin)) out.set(url, asinToOffer.get(asin));
  }
  return out;
}
