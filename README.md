## Architecture

```
web_app/
├── backend/
│   └── api.py              # Flask API server
├── frontend/
│   ├── index.html          # Main HTML interface
│   └── app.js              # Canvas-based visualization engine
├── data/
│   ├── format-map.tsv              # Format tag → display name mappings
│   ├── content-map-with-unesco.tsv # Content tag → UNESCO mappings
│   ├── edu_content_map.tsv         # Educational content mappings
│   └── tsne_metadata/
│       ├── format/
│       │   └── metadata_format.tsv   # ~130k videos with format classification
│       └── content/
│           └── metadata_content.tsv  # ~140k videos with content classification
├── .python-version         # Python 3.11.0 (used by Render)
└── requirements.txt        # Python dependencies
```

## Quick Start

```bash
cd web_app
bash run.sh
```

## Step-by-step Run

### 1. Install Dependencies

```bash
cd web_app
pip install -r requirements.txt
```

### 2. Verify Metadata Files

Ensure your metadata files exist:
```
web_app/data/tsne_metadata/
├── format/
│   └── metadata_format.tsv
└── content/
    └── metadata_content.tsv
```

## Running the Application

### Development Mode

```bash
cd web_app
python backend/api.py
```

Then open your browser to: `http://localhost:5000`

### Production Mode (with Gunicorn)

```bash
cd web_app
gunicorn -w 4 -b 0.0.0.0:5000 backend.api:app
```

### Cloud Platform (Render)

Deployment is configured via `render.yaml` at the repo root:
- `rootDir` is set to `./web_app`
- Python version is pinned via `web_app/.python-version`
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn --workers 4 --bind 0.0.0.0:$PORT backend.api:app`
