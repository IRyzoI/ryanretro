/* =====================================================================
   Ryan Retro Creator Planner — sync API
   Mount in your existing Express app:

     const plannerApi = require('./planner-api');
     app.use('/api/planner', plannerApi);

   Requires two things on Railway:
   1. A Volume mounted at /data  (Service -> Volumes -> New Volume, mount path: /data)
      Railway's normal filesystem is wiped on every deploy; the volume persists.
   2. An environment variable PLANNER_TOKEN set to a long random string.
      This is the password your devices use. Generate one with:
      node -e "console.log(require('crypto').randomBytes(24).toString('hex'))"
   ===================================================================== */

const express = require('express');
const fs = require('fs/promises');
const path = require('path');

const router = express.Router();

const DATA_DIR = process.env.PLANNER_DATA_DIR || '/data/planner';
const IMG_DIR = path.join(DATA_DIR, 'img');
const STATE_FILE = path.join(DATA_DIR, 'state.json');
const TOKEN = process.env.PLANNER_TOKEN;

async function ensureDirs(){
  await fs.mkdir(IMG_DIR, { recursive: true });
}
ensureDirs().catch(err => console.error('planner-api: could not create data dir', err));

/* ---- auth: every request needs Authorization: Bearer <PLANNER_TOKEN> ---- */
router.use((req, res, next) => {
  if (!TOKEN){
    return res.status(500).json({ error: 'PLANNER_TOKEN is not configured on the server' });
  }
  const auth = req.get('Authorization') || '';
  if (auth !== 'Bearer ' + TOKEN){
    return res.status(401).json({ error: 'unauthorized' });
  }
  next();
});

/* JSON bodies; images are base64 data URLs so allow a generous limit */
router.use(express.json({ limit: '10mb' }));

const safeId = id => /^[a-z0-9]+$/i.test(id);

/* ---- state ---- */
router.get('/', async (req, res) => {
  try {
    const raw = await fs.readFile(STATE_FILE, 'utf8');
    res.json({ value: raw });
  } catch (e){
    if (e.code === 'ENOENT') return res.json({ value: null });
    res.status(500).json({ error: 'read failed' });
  }
});

router.put('/', async (req, res) => {
  if (typeof req.body.value !== 'string') return res.status(400).json({ error: 'value must be a string' });
  try {
    await ensureDirs();
    const tmp = STATE_FILE + '.tmp';
    await fs.writeFile(tmp, req.body.value, 'utf8');   // atomic-ish: write then rename
    await fs.rename(tmp, STATE_FILE);
    res.json({ ok: true });
  } catch (e){
    console.error('planner-api state write failed', e);
    res.status(500).json({ error: 'write failed' });
  }
});

/* ---- images ---- */
router.get('/img/:id', async (req, res) => {
  if (!safeId(req.params.id)) return res.status(400).json({ error: 'bad id' });
  try {
    const raw = await fs.readFile(path.join(IMG_DIR, req.params.id), 'utf8');
    res.json({ value: raw });
  } catch (e){
    if (e.code === 'ENOENT') return res.status(404).json({ error: 'not found' });
    res.status(500).json({ error: 'read failed' });
  }
});

router.put('/img/:id', async (req, res) => {
  if (!safeId(req.params.id)) return res.status(400).json({ error: 'bad id' });
  if (typeof req.body.value !== 'string') return res.status(400).json({ error: 'value must be a string' });
  try {
    await ensureDirs();
    await fs.writeFile(path.join(IMG_DIR, req.params.id), req.body.value, 'utf8');
    res.json({ ok: true });
  } catch (e){
    console.error('planner-api image write failed', e);
    res.status(500).json({ error: 'write failed' });
  }
});

router.delete('/img/:id', async (req, res) => {
  if (!safeId(req.params.id)) return res.status(400).json({ error: 'bad id' });
  try {
    await fs.unlink(path.join(IMG_DIR, req.params.id));
    res.json({ ok: true });
  } catch (e){
    if (e.code === 'ENOENT') return res.json({ ok: true });   // already gone = success
    res.status(500).json({ error: 'delete failed' });
  }
});

module.exports = router;
