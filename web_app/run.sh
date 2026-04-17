#!/bin/bash

# Simple launcher script for t-SNE Visualization Web App

echo "==================================="
echo "t-SNE YouTube Visualization"
echo "==================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Check if metadata files exist
if [ ! -f "data/tsne_metadata/format/metadata_format.tsv" ]; then
    echo "WARNING: Format metadata file not found!"
    echo "Expected: data/tsne_metadata/format/metadata_format.tsv"
fi

if [ ! -f "data/tsne_metadata/content/metadata_content.tsv" ]; then
    echo "WARNING: Content metadata file not found!"
    echo "Expected: data/tsne_metadata/content/metadata_content.tsv"
fi

echo ""
echo "==================================="
echo "Starting server..."
echo "==================================="
echo ""
echo "Server will be available at:"
echo "http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the Flask app
python backend/api.py
