#!/usr/bin/env python3
"""
One-time script to patch metadata files:
  - Fixes top_categories (was all 'None') with proper mapped display names
"""

import pandas as pd
import numpy as np
from pathlib import Path

DROPBOX = Path("/Users/nikhilkuppa/Dropbox (Personal)/youtube_video_tagging/youtube_project")
LOCAL = Path("/Users/nikhilkuppa/Documents/Research/CCNY/youtube/youtube_project")

CONTENT_CLASS = DROPBOX / "classify_content/output/content_classifications.tsv"
FORMAT_CLASS  = DROPBOX / "classify_format/output/format_classifications.tsv"
META_CONTENT  = LOCAL / "tsne_data_reduction/code/tsne_metadata/content/metadata_content.tsv"
META_FORMAT   = LOCAL / "tsne_data_reduction/code/tsne_metadata/format/metadata_format.tsv"
EDU_MAP       = LOCAL / "classify_content/input/edu_content_map.tsv"
CONTENT_MAP   = LOCAL / "classify_content/input/content-map-with-unesco.tsv"
FORMAT_MAP    = LOCAL / "classify_format/input/format-map.tsv"


def load_mappings():
    # Edu content map: mapUNESCO column
    edu_df = pd.read_csv(EDU_MAP, sep='\t')
    edu_map = {}
    for _, row in edu_df.iterrows():
        tag = str(row.get('tag', '')).strip().rstrip(':')
        val = str(row.get('mapUNESCO', '')).strip()
        if tag and val and val.lower() != 'nan':
            edu_map[tag] = val.title()

    # Non-edu content map: map2 column
    content_df = pd.read_csv(CONTENT_MAP, sep='\t')
    content_map = {}
    for _, row in content_df.iterrows():
        tag = str(row.get('tag', '')).strip().rstrip(':')
        val = str(row.get('map2', '')).strip()
        if tag and val and val.lower() != 'nan':
            content_map[tag] = val.title()

    # Format map: map2 column
    fmt_df = pd.read_csv(FORMAT_MAP, sep='\t')
    fmt_map = {}
    for _, row in fmt_df.iterrows():
        tag = str(row.get('tag', '')).strip()
        val = str(row.get('map2', '')).strip()
        if tag and val and val.lower() != 'nan':
            fmt_map[tag] = val.title()

    return edu_map, content_map, fmt_map


def content_display(tag, edu_map, content_map):
    tag = tag.strip()
    if tag in edu_map:
        return edu_map[tag]
    if tag in content_map:
        return content_map[tag]
    return tag.title()


def format_display(tag, fmt_map):
    tag = tag.strip()
    return fmt_map.get(tag, tag.title())


def compute_top_categories(video_key, scores_df, display_fn, top_n=5):
    """
    Return 'Name:score' pairs for the top_n unique mapped categories with non-zero scores.
    Scores are stored so api.py can blend colors via weighted average.
    Display logic (showing first 3) is handled in the frontend.
    """
    if video_key not in scores_df.index:
        return ''
    row = scores_df.loc[video_key]
    row_num = pd.to_numeric(row, errors='coerce').fillna(0)
    nonzero = row_num[row_num > 0].sort_values(ascending=False)

    # Deduplicate: keep highest score per mapped name
    seen = {}  # name -> score
    for tag in nonzero.index:
        name = display_fn(tag)
        if name not in seen:
            seen[name] = float(nonzero[tag])
        if len(seen) >= top_n:
            break

    return '; '.join(f"{name}:{score:.1f}" for name, score in seen.items())


def patch(meta_path, class_path, video_suffix, display_fn):
    print(f"\nPatching {meta_path.name} ...")
    meta = pd.read_csv(meta_path, sep='\t')

    print(f"  Loading classifications from {class_path} ...")
    scores = pd.read_csv(class_path, sep='\t')
    scores['_key'] = scores['Video'].str.removesuffix(video_suffix)
    scores = scores.set_index('_key')
    drop = [c for c in ['channel_name', 'Video'] if c in scores.columns]
    scores = scores.drop(columns=drop)

    meta['top_categories'] = meta['video_name'].apply(
        lambda vn: compute_top_categories(str(vn), scores, display_fn)
    )

    covered = (meta['top_categories'] != '').sum()
    print(f"  {covered}/{len(meta)} videos with top_categories")
    meta.to_csv(meta_path, sep='\t', index=False)
    print(f"  Saved.")


def main():
    edu_map, content_map, fmt_map = load_mappings()
    print(f"Mappings loaded: {len(edu_map)} edu, {len(content_map)} content, {len(fmt_map)} format")

    patch(
        META_CONTENT,
        CONTENT_CLASS,
        '_classification',
        lambda tag: content_display(tag, edu_map, content_map)
    )

    patch(
        META_FORMAT,
        FORMAT_CLASS,
        '_format_classification',
        lambda tag: format_display(tag, fmt_map)
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
