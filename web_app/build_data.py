#!/usr/bin/env python3
"""
Build compact, precomputed artifacts for the t-SNE web app.

Combines:
  - the existing per-classification metadata TSVs (t-SNE coordinates + video
    info, in data/tsne_metadata/) which are NOT regenerated here, and
  - taxonomy/outputs/master_retagged.tsv (the unified ISCED-F + NE taxonomy
    this project builds separately) for clean, discrete category labels.

Replaces patch_metadata.py: instead of patching the old fine-grained
map2/mapUNESCO labels in place, this produces small, fast-loading artifacts
that the API can serve directly with no per-request work.

Run once whenever classifications/taxonomy change:
    python web_app/build_data.py [--master-retagged PATH]

Outputs, per classification, into web_app/data/precomputed/:
    {classification}.bin              packed <11s id><float32 x><float32 y><uint8 category_id>
    {classification}_categories.json  [{id, name, colorHex, count, centroidX, centroidY}, ...]
    {classification}_meta.json.gz     {video_id: {title, channel, url, viewCount, likeCount,
                                        videoName, category, contentCategory, contentTags,
                                        formatCategory, formatTags}}
"""
import argparse
import csv
import gzip
import json
import statistics
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent  # web_app/
DATA_DIR = BASE / 'data'
OUT_DIR = DATA_DIR / 'precomputed'

# youtube_tags_app/ and youtube/youtube_project/ are sibling directories
# under .../CCNY/. This is a build-time-only dependency (not needed at
# request time / deploy time) since its output is committed into
# data/precomputed/.
DEFAULT_MASTER_RETAGGED = (
    BASE.parent.parent / 'youtube' / 'youtube_project' / 'taxonomy' / 'outputs' / 'master_retagged.tsv'
)

METADATA_PATHS = {
    'content': DATA_DIR / 'tsne_metadata' / 'content' / 'metadata_content.tsv',
    'format': DATA_DIR / 'tsne_metadata' / 'format' / 'metadata_format.tsv',
}

UNCATEGORIZED = 'Uncategorized'
UNCATEGORIZED_COLOR = '#555555'

# Fixed, hand-picked palette. For the 11 ISCED-derived (education) buckets,
# colors are carried over from the paper's get_edu_colormap()
# (classify_content/figures/tsne_plot.py) under their new supercategory
# names. The 10 "NE" (non-education) buckets get new hand-picked colors in
# the same spirit (distinct, saturated, legible on black).
CONTENT_PALETTE = {
    'Natural sciences': '#e6194b',
    'Social sciences': '#4363d8',
    'Humanities': '#f58231',
    'Engineering & manufacturing': '#911eb4',
    'Health & welfare': '#46f0f0',
    'Education & teaching': '#f032e6',
    'Arts & culture': '#bcf60c',
    'Technology': '#fabebe',
    'Agriculture & environment': '#008080',
    'Business & law': '#e6beff',
    'Languages & literature': '#fffac8',
    # NE extension (non-education)
    'Entertainment & media': '#ff6f61',
    'News & current affairs': '#ffd700',
    'Skills & how-to': '#00b894',
    'Lifestyle & personal': '#ff2ba8',
    'Growth & career': '#6c5ce7',
    'Wellbeing & fitness': '#55efc4',
    'Gaming & digital culture': '#1e90ff',
    'Misc & internet culture': '#a29bfe',
    'Sports & athletics': '#00cec9',
    'Religion & spirituality': '#800000',
    UNCATEGORIZED: UNCATEGORIZED_COLOR,
}

FORMAT_PALETTE = {
    'Talking Head': '#e6194b',
    'Documentaries': '#3cb44b',
    'Infographics': '#4363d8',
    'Interviews': '#f58231',
    'Writing Cursor': '#911eb4',
    'Stage Talks': '#46f0f0',
    'Demonstrations': '#f032e6',
    'Lectures': '#bcf60c',
    'Animations': '#fabebe',
    'Skit': '#008080',
    'Writing Board': '#e6beff',
    'Screencasts': '#9a6324',
    'Others': '#808080',
    'Gaming': '#800000',
    'Slidedecks': '#aaffc3',
    'Panel': '#ffd700',
    'Writing Hand': '#ff6f61',
    'Vlog': '#6c5ce7',
    UNCATEGORIZED: UNCATEGORIZED_COLOR,
}


def load_master_retagged(path: Path) -> dict:
    """Index master_retagged.tsv by video ID for a single-pass join."""
    print(f"Loading unified taxonomy from {path} ...")
    if not path.exists():
        sys.exit(f"ERROR: master_retagged.tsv not found at {path}\n"
                  f"Pass --master-retagged to point at it explicitly.")

    index = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            vid = row.get('ID', '').strip()
            if not vid:
                continue
            index[vid] = row
    print(f"  {len(index)} rows indexed")
    return index


def _dedupe_path(parts):
    """Join non-empty parts with '>', dropping consecutive duplicates
    (e.g. category == label happens a lot in the unified taxonomy)."""
    path = []
    for p in parts:
        p = (p or '').strip()
        if p and (not path or path[-1] != p):
            path.append(p)
    return ' > '.join(path)


def content_category(row: dict) -> str:
    return row.get('content_Tag1_supercategory', '').strip() or UNCATEGORIZED


def content_tag_paths(row: dict) -> list:
    """Up to 3 unique 'Supercategory > Category > Label' breadcrumbs."""
    paths = []
    for i in (1, 2, 3):
        path = _dedupe_path([
            row.get(f'content_Tag{i}_supercategory', ''),
            row.get(f'content_Tag{i}_category', ''),
            row.get(f'content_Tag{i}_label', ''),
        ])
        if path and path not in paths:
            paths.append(path)
    return paths


def format_category(row: dict) -> str:
    return row.get('format_Tag1_analysis_label', '').strip() or UNCATEGORIZED


def format_tag_paths(row: dict) -> list:
    """Up to 3 unique 'Family > Analysis label' breadcrumbs."""
    paths = []
    for i in (1, 2, 3):
        path = _dedupe_path([
            row.get(f'format_Tag{i}_family', ''),
            row.get(f'format_Tag{i}_analysis_label', ''),
        ])
        if path and path not in paths:
            paths.append(path)
    return paths


CLASSIFIERS = {
    'content': (content_category, CONTENT_PALETTE),
    'format': (format_category, FORMAT_PALETTE),
}


def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def clean_channel_name(video_name: str, video_id: str, fallback: str) -> str:
    """
    The precomputed 'Channel' column in the old metadata TSVs was derived by
    chopping a fixed number of '_'-separated tokens off the end of `Video`
    (see tsne_data_reduction/code/{content,format}_tsne.py), which mangles
    any channel name with more than one or two words (e.g. "The Graham
    Norton Show" -> "The_Graham"). video_name is "{channel}_{video_id}" with
    spaces already normalized to underscores upstream, and video_id is
    known and fixed-length, so strip it off directly instead.
    """
    video_name = (video_name or '').strip()
    suffix = f'_{video_id}'
    if video_id and video_name.endswith(suffix):
        channel_raw = video_name[: -len(suffix)]
    else:
        channel_raw = fallback or ''
    return channel_raw.replace('_', ' ').strip() or (fallback or 'N/A')


def build_classification(classification: str, retag_index: dict):
    meta_path = METADATA_PATHS[classification]
    print(f"\nBuilding {classification} from {meta_path} ...")
    if not meta_path.exists():
        sys.exit(f"ERROR: metadata file not found: {meta_path}")

    classify_fn, palette = CLASSIFIERS[classification]

    ids, xs, ys, cat_names = [], [], [], []
    meta = {}
    skipped = 0
    malformed_id = 0

    with open(meta_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            vid = str(row.get('id', '')).strip()
            # Standard YouTube video IDs are exactly 11 chars; a handful of
            # rows upstream have a channel name or other garbage leaked into
            # this column instead - drop those rather than ship bad ids.
            if len(vid) != 11:
                if vid:
                    malformed_id += 1
                continue
            retag_row = retag_index.get(vid)
            if retag_row is None:
                skipped += 1
                category = UNCATEGORIZED
                content_cat, content_tags = UNCATEGORIZED, []
                format_cat, format_tags = UNCATEGORIZED, []
            else:
                category = classify_fn(retag_row)
                content_cat, content_tags = content_category(retag_row), content_tag_paths(retag_row)
                format_cat, format_tags = format_category(retag_row), format_tag_paths(retag_row)

            try:
                x = float(row['tsne_x'])
                y = float(row['tsne_y'])
            except (KeyError, ValueError):
                continue

            ids.append(vid)
            xs.append(x)
            ys.append(y)
            cat_names.append(category)

            video_name = row.get('video_name', 'N/A')
            meta[vid] = {
                'title': row.get('title', 'N/A'),
                'channel': clean_channel_name(video_name, vid, row.get('Channel', '')),
                'url': row.get('URL', ''),
                'viewCount': row.get('viewCount', 'N/A'),
                'likeCount': row.get('likeCount', 'N/A'),
                'videoName': video_name,
                'category': category,
                'contentCategory': content_cat,
                'contentTags': content_tags,
                'formatCategory': format_cat,
                'formatTags': format_tags,
            }

    n = len(xs)
    print(f"  {n} videos ({skipped} not found in unified taxonomy -> Uncategorized, "
          f"{malformed_id} dropped for a malformed id)")

    # Assign stable, deterministic category ids: most frequent first,
    # Uncategorized always last.
    counts = Counter(cat_names)
    ordered = sorted(
        (c for c in counts if c != UNCATEGORIZED),
        key=lambda c: -counts[c]
    )
    if UNCATEGORIZED in counts:
        ordered.append(UNCATEGORIZED)
    if len(ordered) > 255:
        sys.exit(f"ERROR: {len(ordered)} categories exceeds uint8 range")
    cat_id_of = {name: i for i, name in enumerate(ordered)}

    # Centroids (median position) per category, for legend/label placement.
    xs_by_cat = defaultdict(list)
    ys_by_cat = defaultdict(list)
    for x, y, name in zip(xs, ys, cat_names):
        xs_by_cat[name].append(x)
        ys_by_cat[name].append(y)

    categories_out = []
    for name in ordered:
        color_hex = palette.get(name, UNCATEGORIZED_COLOR)
        categories_out.append({
            'id': cat_id_of[name],
            'name': name,
            'colorHex': color_hex,
            'count': counts[name],
            'centroidX': statistics.median(xs_by_cat[name]),
            'centroidY': statistics.median(ys_by_cat[name]),
        })

    # Binary coordinates: <11-byte ascii id><float32 x><float32 y><uint8 category_id>
    # The id travels with each record (not just an index) so the frontend can
    # look up per-video metadata / build hover-cache keys without a second
    # id-list fetch to keep in sync.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bin_path = OUT_DIR / f'{classification}.bin'
    packer = struct.Struct('<11sffB')
    with open(bin_path, 'wb') as f:
        for vid, x, y, name in zip(ids, xs, ys, cat_names):
            f.write(packer.pack(vid.encode('ascii'), x, y, cat_id_of[name]))
    print(f"  wrote {bin_path} ({bin_path.stat().st_size / 1024:.0f} KB)")

    cat_path = OUT_DIR / f'{classification}_categories.json'
    with open(cat_path, 'w', encoding='utf-8') as f:
        json.dump(categories_out, f, indent=2)
    print(f"  wrote {cat_path} ({len(categories_out)} categories)")

    meta_path_out = OUT_DIR / f'{classification}_meta.json.gz'
    with gzip.open(meta_path_out, 'wt', encoding='utf-8') as f:
        json.dump(meta, f)
    print(f"  wrote {meta_path_out} ({meta_path_out.stat().st_size / 1024:.0f} KB gzipped)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master-retagged', type=Path, default=DEFAULT_MASTER_RETAGGED,
                         help='Path to taxonomy/outputs/master_retagged.tsv')
    args = parser.parse_args()

    retag_index = load_master_retagged(args.master_retagged)
    for classification in ('content', 'format'):
        build_classification(classification, retag_index)

    print("\nDone. Precomputed artifacts are in web_app/data/precomputed/")


if __name__ == '__main__':
    main()
