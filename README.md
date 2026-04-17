## Architecture

```
web_app/
├── backend/
│   └── api.py              # Flask API server
├── frontend/
│   ├── index.html          # Main HTML interface
│   └── app.js              # Canvas-based visualization engine
├── requirements.txt        # Python dependencies
└── README.md              # This file
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
tsne_metadata/
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

### Cloud Platform (Heroku, Railway, Render)

1. Add a `Procfile`:
```
web: gunicorn -w 4 backend.api:app
```