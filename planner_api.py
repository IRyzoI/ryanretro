# =====================================================================
# Ryan Retro Creator Planner — sync API (Flask)
#
# Setup:
# 1. Put this file next to app.py as: planner_api.py
# 2. In app.py, after your `app = Flask(...)` line, add:
#
#      from planner_api import planner_api
#      app.register_blueprint(planner_api, url_prefix='/api/planner')
#
# 3. On Railway (ryanretro service -> Variables) add:
#      PLANNER_TOKEN      = a long random string (your devices' password)
#      PLANNER_DATA_DIR   = <your volume mount path>/planner
#                           (only needed if the data-vol mount path is NOT /data)
# =====================================================================

import os
import re
from flask import Blueprint, request, jsonify

planner_api = Blueprint('planner_api', __name__)

DATA_DIR = os.environ.get('PLANNER_DATA_DIR', '/data/planner')
IMG_DIR = os.path.join(DATA_DIR, 'img')
STATE_FILE = os.path.join(DATA_DIR, 'state.json')
TOKEN = os.environ.get('PLANNER_TOKEN')

SAFE_ID = re.compile(r'^[a-zA-Z0-9]+$')


def _ensure_dirs():
    try:
        os.makedirs(IMG_DIR, exist_ok=True)
    except OSError:
        pass  # volume not mounted (e.g. running locally) — requests will report errors


_ensure_dirs()


@planner_api.before_request
def _auth():
    if not TOKEN:
        return jsonify(error='PLANNER_TOKEN is not configured on the server'), 500
    if request.headers.get('Authorization') != 'Bearer ' + TOKEN:
        return jsonify(error='unauthorized'), 401


# ---------------- state ----------------

@planner_api.route('/', methods=['GET'], strict_slashes=False)
def get_state():
    try:
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return jsonify(value=f.read())
    except FileNotFoundError:
        return jsonify(value=None)
    except OSError:
        return jsonify(error='read failed'), 500


@planner_api.route('/', methods=['PUT'], strict_slashes=False)
def put_state():
    body = request.get_json(silent=True) or {}
    value = body.get('value')
    if not isinstance(value, str):
        return jsonify(error='value must be a string'), 400
    try:
        _ensure_dirs()
        tmp = STATE_FILE + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(value)
        os.replace(tmp, STATE_FILE)  # atomic swap, no half-written saves
        return jsonify(ok=True)
    except OSError:
        return jsonify(error='write failed (is the volume mounted?)'), 500


# ---------------- images ----------------

@planner_api.route('/img/<img_id>', methods=['GET'])
def get_img(img_id):
    if not SAFE_ID.match(img_id):
        return jsonify(error='bad id'), 400
    try:
        with open(os.path.join(IMG_DIR, img_id), 'r', encoding='utf-8') as f:
            return jsonify(value=f.read())
    except FileNotFoundError:
        return jsonify(error='not found'), 404
    except OSError:
        return jsonify(error='read failed'), 500


@planner_api.route('/img/<img_id>', methods=['PUT'])
def put_img(img_id):
    if not SAFE_ID.match(img_id):
        return jsonify(error='bad id'), 400
    body = request.get_json(silent=True) or {}
    value = body.get('value')
    if not isinstance(value, str):
        return jsonify(error='value must be a string'), 400
    try:
        _ensure_dirs()
        with open(os.path.join(IMG_DIR, img_id), 'w', encoding='utf-8') as f:
            f.write(value)
        return jsonify(ok=True)
    except OSError:
        return jsonify(error='write failed (is the volume mounted?)'), 500


@planner_api.route('/img/<img_id>', methods=['DELETE'])
def delete_img(img_id):
    if not SAFE_ID.match(img_id):
        return jsonify(error='bad id'), 400
    try:
        os.remove(os.path.join(IMG_DIR, img_id))
    except FileNotFoundError:
        pass  # already gone — that's a success
    except OSError:
        return jsonify(error='delete failed'), 500
    return jsonify(ok=True)
