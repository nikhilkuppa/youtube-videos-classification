"""
Flask API backend for serving tSNE visualization metadata on-demand.
This prevents scraping by only serving data via API requests.
"""

import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import re
import ast
from pathlib import Path
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)  # Enable CORS for frontend-backend communication

# Global data stores
DATA_CACHE = {}
LABEL_MAPS = {}


def parse_color(color_data):
    """Parse color data from various formats into RGB tuple."""
    if isinstance(color_data, str):
        color_str = re.sub(r'np\.float64\((.*?)\)', r'\1', str(color_data))
        try:
            color_tuple = ast.literal_eval(color_str)
        except (ValueError, SyntaxError):
            numbers = re.findall(r'[\d.]+', color_str)
            if len(numbers) >= 3:
                color_tuple = (float(numbers[0]), float(numbers[1]), float(numbers[2]))
            else:
                color_tuple = (0.2, 0.6, 1.0)
    elif isinstance(color_data, (list, tuple, np.ndarray)):
        color_tuple = tuple(color_data[:3])
    else:
        color_tuple = (0.2, 0.6, 1.0)

    color_tuple = tuple(max(0, min(1, float(c))) for c in color_tuple[:3])
    return color_tuple


def load_label_maps():
    """Load mapping tables for label display names (cached)."""
    if LABEL_MAPS:
        return LABEL_MAPS

    # Label map files are bundled in web_app/data/
    data_path = Path(__file__).parent.parent / "data"

    try:
        # Educational content: mapUNESCO column from edu_content_map.tsv
        edu_df = pd.read_csv(data_path / "edu_content_map.tsv", sep='\t')
        edu_map = {}
        for _, row in edu_df.iterrows():
            tag = str(row.get('tag', '')).strip().rstrip(':')
            val = str(row.get('mapUNESCO', '')).strip()
            if tag and val and val.lower() != 'nan':
                edu_map[tag] = val.title()
        LABEL_MAPS['edu'] = edu_map

        # Non-educational content: map2 column from content-map-with-unesco.tsv
        content_df = pd.read_csv(data_path / "content-map-with-unesco.tsv", sep='\t')
        content_map = {}
        for _, row in content_df.iterrows():
            tag = str(row.get('tag', '')).strip().rstrip(':')
            val = str(row.get('map2', '')).strip()
            if tag and val and val.lower() != 'nan':
                content_map[tag] = val.title()
        LABEL_MAPS['content'] = content_map

        # Format: map2 column from format-map.tsv
        fmt_df = pd.read_csv(data_path / "format-map.tsv", sep='\t')
        fmt_map = {}
        for _, row in fmt_df.iterrows():
            tag = str(row.get('tag', '')).strip()
            val = str(row.get('map2', '')).strip()
            if tag and val and val.lower() != 'nan':
                fmt_map[tag] = val.title()
        LABEL_MAPS['format'] = fmt_map

        logger.info(
            f"Label maps loaded: {len(edu_map)} edu, "
            f"{len(content_map)} content, {len(fmt_map)} format"
        )
    except Exception as e:
        logger.error(f"Error loading label maps: {e}")

    return LABEL_MAPS


def generate_label_colors(label_list):
    """
    Assign a distinct RGB color to each unique label using golden-angle HSL
    so every dot for the same mapped category gets the same color.
    """
    unique_labels = sorted(set(label_list))
    n = len(unique_labels)
    color_map = {}
    for i, label in enumerate(unique_labels):
        hue = (i * 137.508) % 360  # golden angle → well-spaced hues
        s = 0.70
        l = 0.55
        c = (1 - abs(2 * l - 1)) * s
        h = hue / 60.0
        x = c * (1 - abs(h % 2 - 1))
        if 0 <= h < 1:   r, g, b = c, x, 0
        elif 1 <= h < 2: r, g, b = x, c, 0
        elif 2 <= h < 3: r, g, b = 0, c, x
        elif 3 <= h < 4: r, g, b = 0, x, c
        elif 4 <= h < 5: r, g, b = x, 0, c
        else:             r, g, b = c, 0, x
        m = l - c / 2
        color_map[label] = [int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)]
    return color_map


def parse_top_categories(top_cats_str):
    """Parse 'Name:score; Name:score' into [(name, score), ...] tuples."""
    result = []
    if not top_cats_str or str(top_cats_str).lower() in ('nan', ''):
        return result
    for part in str(top_cats_str).split(';'):
        part = part.strip()
        if ':' in part:
            name, score_str = part.rsplit(':', 1)
            try:
                result.append((name.strip(), float(score_str.strip())))
            except ValueError:
                pass
    return result


def blend_label_colors(label_scores, label_colors, fallback):
    """Weighted-average RGB blend of label colors by score."""
    if not label_scores:
        return fallback
    # Filter to labels that have a color assigned
    pairs = [(name, score) for name, score in label_scores if name in label_colors]
    if not pairs:
        return fallback
    total = sum(s for _, s in pairs)
    if total == 0:
        return label_colors.get(pairs[0][0], fallback)
    r = g = b = 0.0
    for name, score in pairs:
        w = score / total
        c = label_colors[name]
        r += w * c[0]
        g += w * c[1]
        b += w * c[2]
    return [int(r), int(g), int(b)]


def get_display_label(raw_label, classification):
    """Map raw tag to capitalized display name."""
    maps = load_label_maps()
    tag = str(raw_label).strip()

    if classification == 'format':
        return maps.get('format', {}).get(tag, tag.title())
    else:
        edu_map = maps.get('edu', {})
        if tag in edu_map:
            return edu_map[tag]
        return maps.get('content', {}).get(tag, tag.title())


def finalize_cross_labels():
    """Cross-reference format and content labels once both caches are ready."""
    if DATA_CACHE.get('_cross_labels_done'):
        return
    fmt_cache = DATA_CACHE.get('metadata_format')
    con_cache = DATA_CACHE.get('metadata_content')
    if not (fmt_cache and con_cache):
        return

    format_labels = {vid: d['label'] for vid, d in fmt_cache['full_data'].items()}
    content_labels = {vid: d['label'] for vid, d in con_cache['full_data'].items()}

    for vid, meta in con_cache['full_data'].items():
        meta['cross_label'] = format_labels.get(vid, 'N/A')
    for vid, meta in fmt_cache['full_data'].items():
        meta['cross_label'] = content_labels.get(vid, 'N/A')

    DATA_CACHE['_cross_labels_done'] = True
    logger.info("Cross labels finalized")


def load_metadata(classification='format'):
    """Load and cache metadata for a given classification."""
    cache_key = f'metadata_{classification}'

    if cache_key in DATA_CACHE:
        logger.info(f"Using cached data for {classification}")
        return DATA_CACHE[cache_key]

    metadata_path = Path(__file__).parent.parent / 'data' / 'tsne_metadata' / classification / f'metadata_{classification}.tsv'

    logger.info(f"Loading metadata from {metadata_path}")

    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        return None

    df = pd.read_csv(metadata_path, sep='\t')
    df = df.dropna(subset=['id'])

    # Map every raw label to its display name up front
    df['display_label'] = df['label'].apply(
        lambda l: get_display_label(l, classification)
    )

    # One color per unique mapped label (not per fine-grained original tag)
    label_colors = generate_label_colors(df['display_label'].tolist())

    full_data = {}
    coordinates = []

    for _, row in df.iterrows():
        video_id = str(row['id'])
        display_label = row['display_label']

        # Score-weighted blend of category colors → gradient at cluster boundaries
        label_scores = parse_top_categories(row.get('top_categories', ''))
        fallback = label_colors.get(display_label, [128, 128, 128])
        color = blend_label_colors(label_scores, label_colors, fallback)

        coordinates.append({
            'id': video_id,
            'x': float(row['tsne_x']),
            'y': float(row['tsne_y']),
            'color': color,
            'label': display_label
        })

        full_data[video_id] = {
            'id': video_id,
            'title': str(row.get('title', 'N/A')),
            'channel': str(row.get('Channel', 'N/A')),
            'label': display_label,
            'url': str(row.get('URL', '')),
            'viewCount': str(row.get('viewCount', 'N/A')),
            'likeCount': str(row.get('likeCount', 'N/A')),
            'topCategories': str(row.get('top_categories', '')),
            'videoName': str(row.get('video_name', 'N/A')),
            'cross_label': ''  # filled by finalize_cross_labels()
        }

    # Category centroids grouped by display label (already computed above)
    cat_groups = df.groupby('display_label').agg(
        tsne_x=('tsne_x', 'median'),
        tsne_y=('tsne_y', 'median')
    ).reset_index()

    category_labels = [
        {'label': str(r['display_label']), 'x': float(r['tsne_x']), 'y': float(r['tsne_y'])}
        for _, r in cat_groups.iterrows()
    ]

    DATA_CACHE[cache_key] = {
        'coordinates': coordinates,
        'full_data': full_data,
        'categories': category_labels,
        'bounds': {
            'x_min': float(df['tsne_x'].min()),
            'x_max': float(df['tsne_x'].max()),
            'y_min': float(df['tsne_y'].min()),
            'y_max': float(df['tsne_y'].max())
        },
        'count': len(df)
    }

    logger.info(f"Loaded {len(df)} videos for {classification}")

    # Attempt cross-label finalization whenever a classification loads
    finalize_cross_labels()

    return DATA_CACHE[cache_key]


@app.route('/')
def serve_index():
    """Serve the main HTML page."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/coordinates/<classification>')
def get_coordinates(classification):
    """Get minimal coordinate data for rendering."""
    data = load_metadata(classification)

    if data is None:
        return jsonify({'error': f'Classification {classification} not found'}), 404

    return jsonify({
        'coordinates': data['coordinates'],
        'categories': data['categories'],
        'bounds': data['bounds'],
        'count': data['count']
    })


@app.route('/api/metadata/<classification>/<video_id>')
def get_video_metadata(classification, video_id):
    """Get full metadata for a specific video."""
    data = load_metadata(classification)

    if data is None:
        return jsonify({'error': f'Classification {classification} not found'}), 404

    if video_id not in data['full_data']:
        return jsonify({'error': f'Video {video_id} not found'}), 404

    return jsonify(data['full_data'][video_id])


@app.route('/api/batch_metadata/<classification>', methods=['POST'])
def get_batch_metadata(classification):
    """Get metadata for multiple videos at once."""
    data = load_metadata(classification)

    if data is None:
        return jsonify({'error': f'Classification {classification} not found'}), 404

    video_ids = request.json.get('video_ids', [])

    results = {}
    for vid in video_ids:
        if vid in data['full_data']:
            results[vid] = data['full_data'][vid]

    return jsonify(results)


@app.route('/api/search/<classification>')
def search_videos(classification):
    """Search videos by title, channel, or category."""
    query = request.args.get('q', '').lower()
    data = load_metadata(classification)

    if data is None:
        return jsonify({'error': f'Classification {classification} not found'}), 404

    if not query:
        return jsonify({'results': []})

    results = []
    for video_id, metadata in data['full_data'].items():
        if (query in metadata['title'].lower() or
                query in metadata['channel'].lower() or
                query in metadata['label'].lower()):
            coord = next((c for c in data['coordinates'] if c['id'] == video_id), None)
            if coord:
                results.append({
                    **metadata,
                    'x': coord['x'],
                    'y': coord['y']
                })

                if len(results) >= 50:
                    break

    return jsonify({'results': results})


@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    logger.info("Preloading metadata...")
    load_metadata('format')
    load_metadata('content')
    finalize_cross_labels()

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
