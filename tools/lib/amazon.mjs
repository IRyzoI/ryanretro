/**
 * Amazon Product Advertising API 5.0 — price lookup.
 *
 * Env vars (set in .env locally, GitHub Secrets in CI):
 *   AMAZON_ACCESS_KEY     PA-API access key
 *   AMAZON_SECRET_KEY     PA-API secret key
 *   AMAZON_PARTNER_TAG    Associates tag (default: ryanretro-20)
 *   AMAZON_HOST           default webservices.amazon.com
 *   AMAZON_REGION         default us-east-1
 *   AMAZON_MARKETPLACE    default www.amazon.com
 *
 * If keys are absent, amazonConfigured() returns false and getAmazonPrices()
 * resolves to an empty map — the caller then falls back to best-effort scraping.
 *
 * Eligibility note: Amazon only grants PA-API access to Associates accounts
 * with 3+ qualifying sales in a trailing 180-day window.
 */
import crypto from 'node:crypto';

const ACCESS = process.env.AMAZON_ACCESS_KEY;
const SECRET = process.env.AMAZON_SECRET_KEY;
const PARTNER_TAG = process.env.AMAZON_PARTNER_TAG || 'ryanretro-20';
const HOST = process.env.AMAZON_HOST || 'webservices.amazon.com';
const REGION = process.env.AMAZON_REGION || 'us-east-1';
const MARKETPLACE = process.env.AMAZON_MARKETPLACE || 'www.amazon.com';
const SERVICE = 'ProductAdvertisingAPI';
const PATH = '/paapi5/getitems';
const UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36';

export function amazonConfigured() {
  return Boolean(ACCESS && SECRET);
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

const hmac = (key, msg) => crypto.createHmac('sha256', key).update(msg, 'utf8').digest();
const sha256hex = (s) => crypto.createHash('sha256').update(s, 'utf8').digest('hex');

/** Call GetItems for up to 10 ASINs; returns the parsed JSON response. */
async function getItems(asins) {
  const amzdate = new Date().toISOString().replace(/[:-]/g, '').replace(/\.\d{3}/, ''); // YYYYMMDDTHHMMSSZ
  const datestamp = amzdate.slice(0, 8);
  const target = 'com.amazon.paapi5.v1.ProductAdvertisingAPIv1.GetItems';
  const payload = JSON.stringify({
    ItemIds: asins,
    Resources: ['Offers.Listings.Price'],
    PartnerTag: PARTNER_TAG,
    PartnerType: 'Associates',
    Marketplace: MARKETPLACE,
  });

  const headers = {
    'content-encoding': 'amz-1.0',
    'content-type': 'application/json; charset=utf-8',
    host: HOST,
    'x-amz-date': amzdate,
    'x-amz-target': target,
  };
  const signedHeaders = 'content-encoding;content-type;host;x-amz-date;x-amz-target';
  const canonicalHeaders = signedHeaders.split(';').map((h) => `${h}:${headers[h]}\n`).join('');
  const canonicalRequest = ['POST', PATH, '', canonicalHeaders, signedHeaders, sha256hex(payload)].join('\n');

  const scope = `${datestamp}/${REGION}/${SERVICE}/aws4_request`;
  const stringToSign = ['AWS4-HMAC-SHA256', amzdate, scope, sha256hex(canonicalRequest)].join('\n');

  const kDate = hmac('AWS4' + SECRET, datestamp);
  const kRegion = hmac(kDate, REGION);
  const kService = hmac(kRegion, SERVICE);
  const kSigning = hmac(kService, 'aws4_request');
  const signature = crypto.createHmac('sha256', kSigning).update(stringToSign, 'utf8').digest('hex');

  const authorization =
    `AWS4-HMAC-SHA256 Credential=${ACCESS}/${scope}, SignedHeaders=${signedHeaders}, Signature=${signature}`;

  const res = await fetch(`https://${HOST}${PATH}`, {
    method: 'POST',
    headers: { ...headers, Authorization: authorization },
    body: payload,
  });
  const text = await res.text();
  let json;
  try { json = JSON.parse(text); } catch { json = null; }
  if (!res.ok) {
    const msg = json?.Errors?.map((e) => e.Code + ': ' + e.Message).join('; ') || text.slice(0, 200);
    throw new Error(`PA-API ${res.status} — ${msg}`);
  }
  return json;
}

const chunk = (arr, n) => Array.from({ length: Math.ceil(arr.length / n) }, (_, i) => arr.slice(i * n, i * n + n));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * @param {Array<{url:string}>} links  Amazon store links from products
 * @returns {Promise<Map<string, number>>}  url -> price (USD). Empty if unconfigured.
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

  // 2. Batch GetItems (max 10 ASINs/request, ~1 req/sec to respect throttling).
  const asinToPrice = new Map();
  for (const group of chunk(asins, 10)) {
    const json = await getItems(group);
    const items = json?.ItemsResult?.Items || [];
    for (const it of items) {
      const amount = it?.Offers?.Listings?.[0]?.Price?.Amount;
      if (typeof amount === 'number' && amount > 0) asinToPrice.set(it.ASIN, Math.round(amount * 100) / 100);
    }
    await sleep(1100);
  }

  // 3. Map prices back onto the original link URLs.
  for (const [url, asin] of urlToAsin) {
    if (asinToPrice.has(asin)) out.set(url, asinToPrice.get(asin));
  }
  return out;
}
