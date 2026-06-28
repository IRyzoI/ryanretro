/**
 * Alert builders for price drops — Discord webhook + email HTML.
 * No dependencies. Each `drop` is:
 *   { id, name, store, merchant, from, to, pct, atLow, url, image, date }
 */

const money = (n) => '$' + Number(n).toFixed(2);

/** Post up to 10 drops as rich embeds to a Discord webhook. */
export async function postDiscord(webhookUrl, drops) {
  if (!drops.length) return;
  const embeds = drops.slice(0, 10).map((d) => ({
    title: `${d.name} — ${d.pct}% off`,
    url: d.url,
    description: `**${money(d.to)}**  ~~${money(d.from)}~~  ·  ${d.merchant}${d.atLow ? '  ·  🔥 lowest yet' : ''}`,
    thumbnail: d.image ? { url: d.image } : undefined,
    color: 0x57f287,
  }));
  const body = {
    username: 'Ryan Retro Price Bot',
    content: `🔻 ${drops.length} price drop${drops.length > 1 ? 's' : ''}`,
    embeds,
  };
  const res = await fetch(webhookUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Discord ${res.status}: ${(await res.text()).slice(0, 200)}`);
}

/** Build an HTML email body listing the drops (used by the GitHub Actions mail step). */
export function buildEmailHtml(drops) {
  if (!drops.length) return '<p>No new price drops this run.</p>';
  const rows = drops
    .map(
      (d) => `
    <tr>
      <td style="padding:10px;border-bottom:1px solid #eee;">
        <a href="${d.url}" style="color:#6254A4;font-weight:bold;text-decoration:none;">${d.name}</a>
        <div style="color:#888;font-size:12px;">${d.merchant}${d.atLow ? ' · 🔥 lowest yet' : ''}</div>
      </td>
      <td style="padding:10px;border-bottom:1px solid #eee;text-align:right;white-space:nowrap;">
        <span style="color:#2e7d32;font-weight:bold;font-size:16px;">${money(d.to)}</span>
        <span style="color:#aaa;text-decoration:line-through;margin-left:6px;">${money(d.from)}</span>
        <div style="color:#2e7d32;font-size:12px;font-weight:bold;">-${d.pct}%</div>
      </td>
    </tr>`
    )
    .join('');
  return `
  <div style="font-family:Arial,Helvetica,sans-serif;max-width:560px;margin:0 auto;">
    <h2 style="color:#6254A4;">🔻 ${drops.length} price drop${drops.length > 1 ? 's' : ''} on Ryan Retro</h2>
    <table style="width:100%;border-collapse:collapse;">${rows}</table>
    <p style="margin-top:16px;"><a href="https://ryanretro.com/deals" style="color:#6254A4;">See all deals →</a></p>
    <p style="color:#aaa;font-size:11px;">Automated alert from the Ryan Retro price tracker.</p>
  </div>`;
}
