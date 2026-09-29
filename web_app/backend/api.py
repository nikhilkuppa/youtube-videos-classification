"""
Flask API backend for serving tSNE visualization data.

Coordinates/colors/categories are served from precomputed artifacts built by
build_data.py (web_app/data/precomputed/) - no TSV parsing or per-row work
happens at request time. Per-video metadata (title/channel/url/...) lives in
a SQLite file and is queried by id/LIKE on demand rather than held in RAM as
a Python dict - a dict of ~270k rich records measured over 1GB RSS in this
process, which is what exceeded Render's 512MB instance limit. Only a small
id -> row-index map (used to find a point's x/y for search) stays resident.
"""
import hashlib
import json
import logging
import os
import sqlite3
import struct
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

limiter = Limiter(get_remote_address, app=app, default_limits=[])

PRECOMPUTED_DIR = Path(__file__).parent.parent / 'data' / 'precomputed'
RECORD_SIZE = 20  # 11-byte ascii id, float32 x, float32 y, uint8 category_id

# classification -> {'coords_bytes': bytes, 'categories': [...], 'db': Connection,
#                     'id_index': {id: row_index}, 'count': int}
DATA_CACHE = {}

VIDEO_COLUMNS = [
    'id', 'title', 'channel', 'url', 'viewCount', 'likeCount', 'videoName',
    'category', 'contentCategory', 'contentTags', 'formatCategory', 'formatTags',
]


def row_to_dict(row) -> dict:
    d = dict(zip(VIDEO_COLUMNS, row))
    d['contentTags'] = json.loads(d['contentTags'] or '[]')
    d['formatTags'] = json.loads(d['formatTags'] or '[]')
    return d


def load_classification(classification: str):
    """
    Load coords/categories/id_index (small, pure-Python data). Safe to call
    before gunicorn's --preload forks workers, so the resulting dicts are
    shared copy-on-write across workers instead of duplicated per process.
    Does NOT open the SQLite connection - see db_for().
    """
    if classification in DATA_CACHE:
        return DATA_CACHE[classification]

    bin_path = PRECOMPUTED_DIR / f'{classification}.bin'
    cat_path = PRECOMPUTED_DIR / f'{classification}_categories.json'
    db_path = PRECOMPUTED_DIR / f'{classification}_meta.sqlite'

    if not (bin_path.exists() and cat_path.exists() and db_path.exists()):
        logger.error(f"Precomputed data missing for '{classification}' in {PRECOMPUTED_DIR}. "
                      f"Run `python web_app/build_data.py` first.")
        return None

    logger.info(f"Loading precomputed data for {classification} ...")

    coords_bytes = bin_path.read_bytes()
    categories_bytes = cat_path.read_bytes()
    categories = json.loads(categories_bytes)

    count = len(coords_bytes) // RECORD_SIZE

    # id -> row index in the .bin, decoded once from the buffer itself (not
    # from the DB) - a plain small-string-keyed dict, ~50MB combined across
    # both classifications, unlike the full metadata dicts this replaces.
    id_index = {}
    for i in range(count):
        off = i * RECORD_SIZE
        vid = coords_bytes[off:off + 11].decode('ascii')
        id_index[vid] = i

    DATA_CACHE[classification] = {
        'coords_bytes': coords_bytes,
        'coords_etag': hashlib.md5(coords_bytes).hexdigest(),
        'categories': categories,
        'categories_bytes': categories_bytes,
        'categories_etag': hashlib.md5(categories_bytes).hexdigest(),
        'db_path': db_path,
        'db': None,  # opened lazily, per worker process - see db_for()
        'id_index': id_index,
        'count': count,
    }
    logger.info(f"Loaded {count} videos for {classification} ({len(categories)} categories)")
    return DATA_CACHE[classification]


def db_for(data):
    """
    Open (or reuse) this worker process's own SQLite connection. A single
    Connection object isn't safe to share across processes, so this must
    happen after gunicorn forks, not at module-level preload time - unlike
    coords_bytes/id_index, which are plain read-only Python objects and
    share fine via copy-on-write.
    """
    if data['db'] is None:
        data['db'] = sqlite3.connect(data['db_path'], check_same_thread=False)
    return data['db']


def point_xy(data, video_id):
    """Read x, y for a single video id directly out of the packed buffer."""
    idx = data['id_index'].get(video_id)
    if idx is None:
        return None
    offset = idx * RECORD_SIZE
    _id, x, y, _cat = struct.unpack_from('<11sffB', data['coords_bytes'], offset)
    return x, y


@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/coordinates/<classification>')
def get_coordinates(classification):
    """
    Raw packed binary: N * <11s id><float32 x><float32 y><uint8 category_id>.
    No titles/URLs/channel names in this payload - just positions and a
    category id, resolved against /api/categories/<classification>.
    """
    data = load_classification(classification)
    if data is None:
        return jsonify({'error': f'Classification {classification} not found'}), 404

    # Conditional caching instead of a blind long-lived cache: an
    # `immutable`/1-year Cache-Control here previously meant that once this
    # data changed (e.g. adding the subcategory breakdown), every browser
    # that had already loaded the page kept serving its stale cached copy
    # forever, with no way to pick up the update short of a hard-refresh.
    # An ETag lets the browser cheaply re-check on each load and only
    # re-download the ~2-3MB payload when it actually changed.
    resp = Response(data['coords_bytes'], mimetype='application/octet-stream')
    resp.headers['Cache-Control'] = 'public, max-age=0, must-revalidate'
    resp.headers['X-Point-Count'] = str(data['count'])
    resp.set_etag(data['coords_etag'])
    return resp.make_conditional(request)


@app.route('/api/categories/<classification>')
def get_categories(classification):
    """Palette + legend + centroids (small; ETag so updates aren't stuck
    behind a stale cache - see get_coordinates for why)."""
    data = load_classification(classification)
    if data is None:
        return jsonify({'error': f'Classification {classification} not found'}), 404

    resp = Response(data['categories_bytes'], mimetype='application/json')
    resp.headers['Cache-Control'] = 'public, max-age=0, must-revalidate'
    resp.set_etag(data['categories_etag'])
    return resp.make_conditional(request)


@app.route('/api/metadata/<classification>/<video_id>')
@limiter.limit('120 per minute')
def get_video_metadata(classification, video_id):
    """Full metadata for a specific video - fetched on demand only."""
    data = load_classification(classification)
    if data is None:
        return jsonify({'error': f'Classification {classification} not found'}), 404

    row = db_for(data).execute(
        'SELECT * FROM videos WHERE id = ?', (video_id,)
    ).fetchone()
    if row is None:
        return jsonify({'error': f'Video {video_id} not found'}), 404

    return jsonify(row_to_dict(row))


@app.route('/api/batch_metadata/<classification>', methods=['POST'])
@limiter.limit('30 per minute')
def get_batch_metadata(classification):
    """Metadata for multiple videos at once (e.g. preloading a viewport)."""
    data = load_classification(classification)
    if data is None:
        return jsonify({'error': f'Classification {classification} not found'}), 404

    video_ids = (request.json or {}).get('video_ids', [])[:200]
    if not video_ids:
        return jsonify({})

    placeholders = ','.join('?' * len(video_ids))
    rows = db_for(data).execute(
        f'SELECT * FROM videos WHERE id IN ({placeholders})', video_ids
    ).fetchall()

    results = {row[0]: row_to_dict(row) for row in rows}
    return jsonify(results)


@app.route('/api/search/<classification>')
@limiter.limit('60 per minute')
def search_videos(classification):
    """Search videos by title, channel, or category."""
    query = request.args.get('q', '').strip()
    data = load_classification(classification)
    if data is None:
        return jsonify({'error': f'Classification {classification} not found'}), 404

    if not query:
        return jsonify({'results': []})

    like = f'%{query}%'
    rows = db_for(data).execute(
        '''SELECT * FROM videos
           WHERE title LIKE ? OR channel LIKE ? OR category LIKE ?
              OR contentCategory LIKE ? OR formatCategory LIKE ?
           LIMIT 50''',
        (like, like, like, like, like),
    ).fetchall()

    results = []
    for row in rows:
        info = row_to_dict(row)
        xy = point_xy(data, info['id'])
        if xy is None:
            continue
        x, y = xy
        results.append({**info, 'x': x, 'y': y})

    return jsonify({'results': results})


@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'})


# Runs at import time (both under `python backend/api.py` and under gunicorn
# with --preload), i.e. once in the master process before workers fork. The
# coords_bytes/id_index dicts built here are then shared copy-on-write by
# every forked worker instead of each worker reloading and duplicating them.
logger.info("Preloading precomputed data...")
load_classification('format')
load_classification('content')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
