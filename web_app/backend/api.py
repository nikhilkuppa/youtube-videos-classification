"""
Flask API backend for serving tSNE visualization data.

Coordinates/colors/categories are served from precomputed artifacts built by
build_data.py (web_app/data/precomputed/) - no TSV parsing or per-row work
happens at request time. Per-video metadata (title/channel/url/...) is only
served on demand, by id, so bulk scraping requires one request per video.
"""
import gzip
import json
import logging
import os
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

# classification -> {'bin': bytes, 'categories': [...], 'meta': {id: {...}},
#                     'id_index': {id: row_index}, 'count': int}
DATA_CACHE = {}


def load_classification(classification: str):
    if classification in DATA_CACHE:
        return DATA_CACHE[classification]

    bin_path = PRECOMPUTED_DIR / f'{classification}.bin'
    cat_path = PRECOMPUTED_DIR / f'{classification}_categories.json'
    meta_path = PRECOMPUTED_DIR / f'{classification}_meta.json.gz'

    if not (bin_path.exists() and cat_path.exists() and meta_path.exists()):
        logger.error(f"Precomputed data missing for '{classification}' in {PRECOMPUTED_DIR}. "
                      f"Run `python web_app/build_data.py` first.")
        return None

    logger.info(f"Loading precomputed data for {classification} ...")

    coords_bytes = bin_path.read_bytes()
    with open(cat_path, encoding='utf-8') as f:
        categories = json.load(f)
    with gzip.open(meta_path, 'rt', encoding='utf-8') as f:
        meta = json.load(f)

    # id -> row index in the .bin, so per-id lookups (metadata, search) can
    # find x/y without a linear scan.
    id_index = {vid: i for i, vid in enumerate(meta.keys())}

    record_size = 20  # 11-byte ascii id, float32 x, float32 y, uint8 category_id
    count = len(coords_bytes) // record_size

    DATA_CACHE[classification] = {
        'coords_bytes': coords_bytes,
        'categories': categories,
        'meta': meta,
        'id_index': id_index,
        'count': count,
    }
    logger.info(f"Loaded {count} videos for {classification} ({len(categories)} categories)")
    return DATA_CACHE[classification]


def point_xy(data, video_id):
    """Read x, y for a single video id directly out of the packed buffer."""
    idx = data['id_index'].get(video_id)
    if idx is None:
        return None
    import struct
    offset = idx * 20
    _id, x, y, _cat = struct.unpack_from('<11sffB', data['coords_bytes'], offset)
    return x, y


@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/coordinates/<classification>')
def get_coordinates(classification):
    """
    Raw packed binary: N * <float32 x><float32 y><uint8 category_id>.
    No titles/URLs/channel names in this payload - just positions and a
    category id, resolved against /api/categories/<classification>.
    """
    data = load_classification(classification)
    if data is None:
        return jsonify({'error': f'Classification {classification} not found'}), 404

    resp = Response(data['coords_bytes'], mimetype='application/octet-stream')
    resp.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
    resp.headers['X-Point-Count'] = str(data['count'])
    return resp


@app.route('/api/categories/<classification>')
def get_categories(classification):
    """Palette + legend + centroids (small, fine to cache aggressively)."""
    data = load_classification(classification)
    if data is None:
        return jsonify({'error': f'Classification {classification} not found'}), 404

    resp = jsonify(data['categories'])
    resp.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
    return resp


@app.route('/api/metadata/<classification>/<video_id>')
@limiter.limit('120 per minute')
def get_video_metadata(classification, video_id):
    """Full metadata for a specific video - fetched on demand only."""
    data = load_classification(classification)
    if data is None:
        return jsonify({'error': f'Classification {classification} not found'}), 404

    info = data['meta'].get(video_id)
    if info is None:
        return jsonify({'error': f'Video {video_id} not found'}), 404

    return jsonify({'id': video_id, **info})


@app.route('/api/batch_metadata/<classification>', methods=['POST'])
@limiter.limit('30 per minute')
def get_batch_metadata(classification):
    """Metadata for multiple videos at once (e.g. preloading a viewport)."""
    data = load_classification(classification)
    if data is None:
        return jsonify({'error': f'Classification {classification} not found'}), 404

    video_ids = (request.json or {}).get('video_ids', [])[:200]

    results = {}
    for vid in video_ids:
        info = data['meta'].get(vid)
        if info is not None:
            results[vid] = {'id': vid, **info}

    return jsonify(results)


@app.route('/api/search/<classification>')
@limiter.limit('60 per minute')
def search_videos(classification):
    """Search videos by title, channel, or category."""
    query = request.args.get('q', '').lower().strip()
    data = load_classification(classification)
    if data is None:
        return jsonify({'error': f'Classification {classification} not found'}), 404

    if not query:
        return jsonify({'results': []})

    results = []
    for video_id, info in data['meta'].items():
        if (query in info['title'].lower()
                or query in info['channel'].lower()
                or query in info['category'].lower()):
            xy = point_xy(data, video_id)
            if xy is None:
                continue
            x, y = xy
            results.append({'id': video_id, **info, 'x': x, 'y': y})
            if len(results) >= 50:
                break

    return jsonify({'results': results})


@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    logger.info("Preloading precomputed data...")
    load_classification('format')
    load_classification('content')

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
