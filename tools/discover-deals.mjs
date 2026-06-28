#!/usr/bin/env node
/**
 * discover-deals.mjs — find cheaper deals from TRUSTED stores and queue them for review.
 *
 * For each catalogue product it keyword-searches Amazon, keeps listings that
 * (a) come from a trusted seller, (b) plausibly match the product (model token
 * in title), (c) sit in a sane price band, and (d) beat our current price.
 * New matches are appended to static/proposals.json (status "new") and a review
 * digest is posted to Discord (+ an email body is written for the workflow).
 * Nothing goes live until you approve a proposal (status -> "approved").
 *
 * AliExpress discovery will plug in here once the Advanced API is approved.
 *
 * Run: node tools/discover-deals.mjs
 */
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
loadEnv(join(ROOT, '.env'));
function loadEnv(p) {
  if (!existsSync(p)) return;
  for (const line of readFileSync(p, 'utf8').split('\n')) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/);
    if (m && process.env[m[1]] === undefined) process.env[m[1]] = m[2].replace(/^["']|["']$/g, '');
  }
}
const { searchAmazonDeals, amazonConfigured } = await import('./lib/amazon.mjs');

const PROPOSALS = join(ROOT, 'static', 'proposals.json');
const TODAY = new Date().toISOString().slice(0, 10);

function loadProducts() {
  const src = readFileSync(join(ROOT, 'static', 'shop.html'), 'utf8');
  return Function(`"use strict"; return ([${src.match(/const products = \[([\s\S]*?)\n\s*\];/)[1]}]);`)();
}
const prices = JSON.parse(readFileSync(join(ROOT, 'static', 'prices.json'), 'utf8'));

const norm = (s) => (s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
const modelTokens = (name) => name.split(/\s+/).filter((t) => /\d/.test(t)).map(norm).filter(Boolean);
function titleMatches(title, name) {
  const t = norm(title);
  const models = modelTokens(name);
  if (models.length) return models.every((m) => t.includes(m));
  const brand = norm(name.split(/\s+/)[0]);
  return brand.length > 2 && t.includes(brand);
}
const round2 = (n) => Math.round(n * 100) / 100;
const currentBest = (id) => {
  const e = prices[id]; if (!e) return null;
  const vals = Object.keys(e).filter((k) => k !== '_meta').map((s) => e[s]?.current).filter((n) => typeof n === 'number');
  return vals.length ? Math.min(...vals) : null;
};

async function discover() {
  const products = loadProducts();
  const out = [];
  for (const p of products) {
    const ref = currentBest(p.id);
    let results = [];
    try { results = await searchAmazonDeals(p.name, 8); } catch (e) { console.error(`  ! ${p.name}: ${e.message}`); }
    for (const r of results) {
      if (!r.trusted || !titleMatches(r.title, p.name) || !(r.price > 0)) continue;
      if (ref && (r.price < ref * 0.3 || r.price > ref * 2)) continue;  // sane band
      if (ref && r.price >= ref - 0.01) continue;                       // must be cheaper
      out.push({
        key: `${p.id}:${r.asin}`, productId: p.id, productName: p.name, image: p.image,
        store: 'Amazon', merchant: r.merchant, asin: r.asin, title: r.title,
        price: round2(r.price), currentBest: ref, savings: ref ? round2(ref - r.price) : null,
        url: r.url, status: 'new', firstSeen: TODAY,
      });
    }
    await new Promise((res) => setTimeout(res, 1200));
  }
  return out;
}

async function postDiscord(items) {
  if (!items.length || !process.env.DISCORD_WEBHOOK_URL) return;
  const embeds = items.slice(0, 10).map((d) => ({
    title: `${d.productName} — $${d.price} at ${d.merchant}`,
    url: d.url,
    description: `Beats current $${d.currentBest} (save $${d.savings}). Trusted store.\n_${d.title.slice(0, 90)}_\nApprove in proposals.json to add.`,
    thumbnail: d.image ? { url: d.image } : undefined,
    color: 0x6254a4,
  }));
  const res = await fetch(process.env.DISCORD_WEBHOOK_URL, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: 'Ryan Retro Deal Finder', content: `🛒 ${items.length} new deal(s) to review`, embeds }),
  });
  if (res.ok) console.log('Posted review digest to Discord.');
}

function emailHtml(items) {
  if (!items.length) return '<p>No new deals to review.</p>';
  const rows = items.map((d) => `<tr>
    <td style="padding:8px;border-bottom:1px solid #eee;"><a href="${d.url}" style="color:#6254A4;font-weight:bold;">${d.productName}</a>
      <div style="color:#888;font-size:12px;">${d.merchant} · ${d.title.slice(0, 60)}</div></td>
    <td style="padding:8px;border-bottom:1px solid #eee;text-align:right;white-space:nowrap;">
      <b style="color:#2e7d32;">$${d.price}</b> <span style="color:#aaa;text-decoration:line-through;">$${d.currentBest}</span>
      <div style="color:#2e7d32;font-size:12px;">save $${d.savings}</div></td></tr>`).join('');
  return `<div style="font-family:Arial;max-width:600px;margin:auto;">
    <h2 style="color:#6254A4;">🛒 ${items.length} new deal(s) to review</h2>
    <table style="width:100%;border-collapse:collapse;">${rows}</table>
    <p style="font-size:13px;">Approve in <code>static/proposals.json</code> (set <code>"status":"approved"</code>) to add to the site.</p></div>`;
}

// --- main ---
if (!amazonConfigured()) { console.log('Amazon not configured — skipping discovery.'); process.exit(0); }
const existing = existsSync(PROPOSALS) ? JSON.parse(readFileSync(PROPOSALS, 'utf8')) : [];
const seen = new Set(existing.map((e) => e.key));
const fresh = (await discover()).filter((d) => !seen.has(d.key));

writeFileSync(PROPOSALS, JSON.stringify([...fresh, ...existing].slice(0, 200), null, 2) + '\n');
writeFileSync(join(ROOT, 'tools', 'proposals-email.html'), emailHtml(fresh));
writeFileSync(join(ROOT, 'tools', 'last-proposals.json'), JSON.stringify(fresh, null, 2) + '\n');

console.log(`\nDiscovery: ${fresh.length} NEW deal(s) queued for review (proposals.json now has ${[...fresh, ...existing].length}).`);
fresh.forEach((d) => console.log(`  • ${d.productName}: $${d.price} ${d.merchant} (was $${d.currentBest})`));
if (fresh.length) await postDiscord(fresh);
