/**
 * High-performance t-SNE visualization using Canvas and spatial indexing
 * Handles 100k+ points efficiently with lazy metadata loading
 */

const API_BASE = window.location.hostname === 'localhost' ? 'http://localhost:5000' : '';

class SpatialIndex {
    constructor(gridSize = 50) {
        this.gridSize = gridSize;
        this.grid = new Map();
    }

    clear() {
        this.grid.clear();
    }

    insert(point, x, y) {
        const cellX = Math.floor(x / this.gridSize);
        const cellY = Math.floor(y / this.gridSize);
        const key = `${cellX},${cellY}`;

        if (!this.grid.has(key)) {
            this.grid.set(key, []);
        }
        this.grid.get(key).push({ ...point, canvasX: x, canvasY: y });
    }

    query(x, y, radius = 10) {
        const cellX = Math.floor(x / this.gridSize);
        const cellY = Math.floor(y / this.gridSize);
        const cellRadius = Math.ceil(radius / this.gridSize);

        const candidates = [];

        for (let dx = -cellRadius; dx <= cellRadius; dx++) {
            for (let dy = -cellRadius; dy <= cellRadius; dy++) {
                const key = `${cellX + dx},${cellY + dy}`;
                const cell = this.grid.get(key);
                if (cell) {
                    candidates.push(...cell);
                }
            }
        }

        // Find closest point within radius
        let closest = null;
        let minDist = radius;

        for (const point of candidates) {
            const dist = Math.sqrt(
                Math.pow(point.canvasX - x, 2) +
                Math.pow(point.canvasY - y, 2)
            );
            if (dist < minDist) {
                minDist = dist;
                closest = point;
            }
        }

        return closest;
    }
}

class TSNEVisualization {
    constructor() {
        this.canvas = document.getElementById('tsne-canvas');
        this.ctx = this.canvas.getContext('2d', { alpha: false });
        this.container = document.getElementById('canvas-container');

        // State
        this.data = null;
        this.classification = 'format';
        this.scale = 1;
        this.translateX = 0;
        this.translateY = 0;
        this.isDragging = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;
        this.hoveredPoint = null;
        this.selectedPoint = null;

        // Performance
        this.spatialIndex = new SpatialIndex(50);
        this.metadataCache = new Map();
        this.renderRequested = false;

        // Constants
        this.DOT_RADIUS = 3;
        this.HOVER_RADIUS = 15;
        this.MIN_SCALE = 0.3;
        this.MAX_SCALE = 10;

        this.init();
    }

    init() {
        this.setupCanvas();
        this.setupEventListeners();
        this.loadData(this.classification);
    }

    setupCanvas() {
        const updateSize = () => {
            const dpr = window.devicePixelRatio || 1;
            const rect = this.container.getBoundingClientRect();

            this.canvas.width = rect.width * dpr;
            this.canvas.height = rect.height * dpr;
            this.canvas.style.width = rect.width + 'px';
            this.canvas.style.height = rect.height + 'px';

            this.ctx.scale(dpr, dpr);

            this.canvasWidth = rect.width;
            this.canvasHeight = rect.height;

            this.requestRender();
        };

        updateSize();
        window.addEventListener('resize', updateSize);
    }

    setupEventListeners() {
        // Mouse events for pan and zoom
        this.canvas.addEventListener('mousedown', this.handleMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.handleMouseUp.bind(this));
        this.canvas.addEventListener('mouseleave', this.handleMouseUp.bind(this));
        this.canvas.addEventListener('wheel', this.handleWheel.bind(this));
        this.canvas.addEventListener('click', this.handleClick.bind(this));

        // Controls
        document.getElementById('zoom-in').addEventListener('click', () => this.zoom(1.3));
        document.getElementById('zoom-out').addEventListener('click', () => this.zoom(0.7));
        document.getElementById('reset-view').addEventListener('click', () => this.resetView());

        // Classification toggle
        document.querySelectorAll('.toggle-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const classification = e.target.dataset.classification;
                if (classification !== this.classification) {
                    this.switchClassification(classification);
                }
            });
        });

        // Search
        const searchInput = document.getElementById('search-input');
        let searchTimeout;
        searchInput.addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => this.handleSearch(e.target.value), 300);
        });
    }

    async loadData(classification) {
        try {
            document.getElementById('loading').style.display = 'block';

            const response = await fetch(`${API_BASE}/api/coordinates/${classification}`);
            const data = await response.json();

            this.data = data;
            this.classification = classification;

            // Update UI
            document.getElementById('video-count').textContent = `${data.count.toLocaleString()} videos`;
            document.querySelectorAll('.toggle-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.classification === classification);
            });

            // Build spatial index
            this.buildSpatialIndex();

            // Reset view
            this.resetView();

            document.getElementById('loading').style.display = 'none';
        } catch (error) {
            console.error('Error loading data:', error);
            document.getElementById('loading').innerHTML = '<div class="placeholder">Error loading data</div>';
        }
    }

    buildSpatialIndex() {
        this.spatialIndex.clear();

        if (!this.data || !this.data.coordinates) return;

        const { coordinates } = this.data;

        coordinates.forEach(point => {
            const [x, y] = this.projectToCanvas(point.x, point.y);
            this.spatialIndex.insert(point, x, y);
        });
    }

    projectToCanvas(x, y) {
        if (!this.data) return [0, 0];

        const { bounds } = this.data;

        const margin = 50;
        const plotWidth = this.canvasWidth - 2 * margin;
        const plotHeight = this.canvasHeight - 2 * margin;

        const normalizedX = (x - bounds.x_min) / (bounds.x_max - bounds.x_min);
        const normalizedY = (y - bounds.y_min) / (bounds.y_max - bounds.y_min);

        const canvasX = margin + normalizedX * plotWidth;
        const canvasY = this.canvasHeight - (margin + normalizedY * plotHeight);

        return [canvasX, canvasY];
    }

    screenToWorld(screenX, screenY) {
        const worldX = (screenX - this.translateX) / this.scale;
        const worldY = (screenY - this.translateY) / this.scale;
        return [worldX, worldY];
    }

    worldToScreen(worldX, worldY) {
        const screenX = worldX * this.scale + this.translateX;
        const screenY = worldY * this.scale + this.translateY;
        return [screenX, screenY];
    }

    requestRender() {
        if (!this.renderRequested) {
            this.renderRequested = true;
            requestAnimationFrame(() => this.render());
        }
    }

    render() {
        this.renderRequested = false;

        if (!this.data) return;

        const ctx = this.ctx;
        const { coordinates, categories } = this.data;

        // Clear canvas
        ctx.fillStyle = '#0a0a0a';
        ctx.fillRect(0, 0, this.canvasWidth, this.canvasHeight);

        ctx.save();

        // Apply transformations
        ctx.translate(this.translateX, this.translateY);
        ctx.scale(this.scale, this.scale);

        // Calculate visible bounds for culling
        const [minWorldX, minWorldY] = this.screenToWorld(0, 0);
        const [maxWorldX, maxWorldY] = this.screenToWorld(this.canvasWidth, this.canvasHeight);

        // Render points
        const dotRadius = this.DOT_RADIUS / this.scale;

        coordinates.forEach(point => {
            const [x, y] = this.projectToCanvas(point.x, point.y);

            if (x < minWorldX - 50 || x > maxWorldX + 50 ||
                y < minWorldY - 50 || y > maxWorldY + 50) {
                return;
            }

            ctx.fillStyle = `rgb(${point.color[0]}, ${point.color[1]}, ${point.color[2]})`;
            ctx.globalAlpha = 0.7;
            ctx.beginPath();
            ctx.arc(x, y, dotRadius, 0, Math.PI * 2);
            ctx.fill();
        });

        // Draw category labels (always shown, fully opaque white)
        ctx.globalAlpha = 1.0;
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = 'rgba(0,0,0,0.8)';
        ctx.lineWidth = 4 / this.scale;
        ctx.font = `bold ${14 / this.scale}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        categories.forEach(cat => {
            const [x, y] = this.projectToCanvas(cat.x, cat.y);
            ctx.strokeText(cat.label, x, y);
            ctx.fillText(cat.label, x, y);
        });

        // Highlight hovered point
        if (this.hoveredPoint) {
            const [x, y] = this.projectToCanvas(this.hoveredPoint.x, this.hoveredPoint.y);
            ctx.globalAlpha = 1;
            ctx.strokeStyle = '#00d4ff';
            ctx.lineWidth = 2 / this.scale;
            ctx.beginPath();
            ctx.arc(x, y, (dotRadius + 3 / this.scale), 0, Math.PI * 2);
            ctx.stroke();
        }

        // Highlight selected point
        if (this.selectedPoint) {
            const [x, y] = this.projectToCanvas(this.selectedPoint.x, this.selectedPoint.y);
            ctx.globalAlpha = 1;
            ctx.strokeStyle = '#ff00ff';
            ctx.lineWidth = 3 / this.scale;
            ctx.beginPath();
            ctx.arc(x, y, (dotRadius + 5 / this.scale), 0, Math.PI * 2);
            ctx.stroke();
        }

        ctx.restore();
    }

    handleMouseDown(e) {
        this.isDragging = true;
        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;
        this.container.classList.add('grabbing');
    }

    handleMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        if (this.isDragging) {
            const dx = e.clientX - this.lastMouseX;
            const dy = e.clientY - this.lastMouseY;

            this.translateX += dx;
            this.translateY += dy;

            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;

            this.requestRender();
        } else {
            const [worldX, worldY] = this.screenToWorld(mouseX, mouseY);
            const point = this.spatialIndex.query(worldX, worldY, this.HOVER_RADIUS / this.scale);

            if (point !== this.hoveredPoint) {
                this.hoveredPoint = point;
                this.requestRender();

                if (point) {
                    this.showHoverInfo(point);
                } else {
                    this.clearHoverInfo();
                }
            }
        }
    }

    handleMouseUp() {
        this.isDragging = false;
        this.container.classList.remove('grabbing');
    }

    handleWheel(e) {
        e.preventDefault();

        const rect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;

        const [worldX, worldY] = this.screenToWorld(mouseX, mouseY);

        this.scale = Math.max(this.MIN_SCALE, Math.min(this.MAX_SCALE, this.scale * zoomFactor));

        const [newScreenX, newScreenY] = this.worldToScreen(worldX, worldY);

        this.translateX += mouseX - newScreenX;
        this.translateY += mouseY - newScreenY;

        this.requestRender();
    }

    handleClick(e) {
        if (this.hoveredPoint) {
            this.selectedPoint = this.hoveredPoint;
            this.showSelectedInfo(this.selectedPoint);
            this.requestRender();
        }
    }

    zoom(factor) {
        const centerX = this.canvasWidth / 2;
        const centerY = this.canvasHeight / 2;

        const [worldX, worldY] = this.screenToWorld(centerX, centerY);

        this.scale = Math.max(this.MIN_SCALE, Math.min(this.MAX_SCALE, this.scale * factor));

        const [newScreenX, newScreenY] = this.worldToScreen(worldX, worldY);

        this.translateX += centerX - newScreenX;
        this.translateY += centerY - newScreenY;

        this.requestRender();
    }

    resetView() {
        this.scale = 1;
        this.translateX = 0;
        this.translateY = 0;
        this.hoveredPoint = null;
        this.selectedPoint = null;

        this.buildSpatialIndex();
        this.clearHoverInfo();
        this.clearSelectedInfo();
        this.requestRender();
    }

    /** Render top 3 category names as badge spans (strips ':score' suffixes). */
    renderCategoryBadges(topCategories) {
        if (!topCategories || topCategories === 'N/A' || topCategories.toLowerCase() === 'nan') {
            return '<span style="color:#555;font-size:12px">N/A</span>';
        }
        // Each entry may be "Name:score" — strip the score part for display
        const badges = topCategories.split(';')
            .map(s => s.trim().replace(/:\d+(\.\d+)?$/, '').trim())
            .filter(Boolean)
            .slice(0, 3);  // show at most 3
        if (badges.length === 0) {
            return '<span style="color:#555;font-size:12px">N/A</span>';
        }
        return badges
            .map(b => `<span class="category-badge">${this.escapeHtml(b)}</span>`)
            .join(' ');
    }

    /** Render a single badge for cross-classification label (format ↔ content). */
    renderCrossBadge(crossLabel) {
        if (!crossLabel || crossLabel === 'N/A' || crossLabel.toLowerCase() === 'nan') {
            return '<span style="color:#555;font-size:12px">N/A</span>';
        }
        return `<span class="category-badge cross-badge">${this.escapeHtml(crossLabel)}</span>`;
    }

    crossLabelName() {
        return this.classification === 'content' ? 'Format' : 'Content';
    }

    async showHoverInfo(point) {
        const metadata = await this.fetchMetadata(point.id);

        if (!metadata) return;

        // Update thumbnail
        const thumbnailContainer = document.getElementById('thumbnail-preview');
        thumbnailContainer.innerHTML = `
            <img src="https://img.youtube.com/vi/${point.id}/maxresdefault.jpg"
                 alt="Thumbnail"
                 onerror="this.src='https://img.youtube.com/vi/${point.id}/hqdefault.jpg'">
        `;

        // Update metadata
        const metadataContainer = document.getElementById('hover-metadata');
        metadataContainer.innerHTML = `
            <div class="metadata-item">
                <div class="metadata-label">Title</div>
                <div class="metadata-value">${this.escapeHtml(metadata.title)}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Channel</div>
                <div class="metadata-value">${this.escapeHtml(metadata.channel)}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Category</div>
                <div class="metadata-value badge-row">
                    ${this.renderCategoryBadges(metadata.topCategories)}
                </div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">${this.crossLabelName()}</div>
                <div class="metadata-value">
                    ${this.renderCrossBadge(metadata.cross_label)}
                </div>
            </div>
            <div class="stats-row">
                <div class="stat-item">
                    <div class="stat-value">${this.formatNumber(metadata.viewCount)}</div>
                    <div class="stat-label">Views</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${this.formatNumber(metadata.likeCount)}</div>
                    <div class="stat-label">Likes</div>
                </div>
            </div>
        `;
    }

    clearHoverInfo() {
        document.getElementById('thumbnail-preview').innerHTML = '<div class="placeholder">Hover over a dot</div>';
        document.getElementById('hover-metadata').innerHTML = '<div class="placeholder">Hover over a dot to see details</div>';
    }

    async showSelectedInfo(point) {
        const metadata = await this.fetchMetadata(point.id);

        if (!metadata) return;

        // Update video embed
        const videoContainer = document.getElementById('video-embed');
        videoContainer.innerHTML = `
            <iframe
                src="https://www.youtube.com/embed/${point.id}?autoplay=1"
                title="YouTube video player"
                frameborder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowfullscreen>
            </iframe>
        `;

        // Update metadata
        const metadataContainer = document.getElementById('selected-metadata');
        metadataContainer.innerHTML = `
            <div class="metadata-item">
                <div class="metadata-label">Title</div>
                <div class="metadata-value">${this.escapeHtml(metadata.title)}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Channel</div>
                <div class="metadata-value">${this.escapeHtml(metadata.channel)}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Category</div>
                <div class="metadata-value badge-row">
                    ${this.renderCategoryBadges(metadata.topCategories)}
                </div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">${this.crossLabelName()}</div>
                <div class="metadata-value">
                    ${this.renderCrossBadge(metadata.cross_label)}
                </div>
            </div>
            <div class="stats-row">
                <div class="stat-item">
                    <div class="stat-value">${this.formatNumber(metadata.viewCount)}</div>
                    <div class="stat-label">Views</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${this.formatNumber(metadata.likeCount)}</div>
                    <div class="stat-label">Likes</div>
                </div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">URL</div>
                <div class="metadata-value">
                    <a href="${metadata.url}" target="_blank">${metadata.url}</a>
                </div>
            </div>
        `;
    }

    clearSelectedInfo() {
        document.getElementById('video-embed').innerHTML = '<div class="placeholder">Click a dot to play video</div>';
        document.getElementById('selected-metadata').innerHTML = '<div class="placeholder">Click a dot to see details</div>';
    }

    async fetchMetadata(videoId) {
        if (this.metadataCache.has(videoId)) {
            return this.metadataCache.get(videoId);
        }

        try {
            const response = await fetch(`${API_BASE}/api/metadata/${this.classification}/${videoId}`);
            const metadata = await response.json();

            this.metadataCache.set(videoId, metadata);
            return metadata;
        } catch (error) {
            console.error('Error fetching metadata:', error);
            return null;
        }
    }

    async handleSearch(query) {
        if (!query || query.length < 2) return;

        try {
            const response = await fetch(`${API_BASE}/api/search/${this.classification}?q=${encodeURIComponent(query)}`);
            const data = await response.json();

            if (data.results && data.results.length > 0) {
                const firstResult = data.results[0];

                const [worldX, worldY] = this.projectToCanvas(firstResult.x, firstResult.y);

                const centerX = this.canvasWidth / 2;
                const centerY = this.canvasHeight / 2;

                this.scale = 3;
                this.translateX = centerX - worldX * this.scale;
                this.translateY = centerY - worldY * this.scale;

                this.selectedPoint = {
                    id: firstResult.id,
                    x: firstResult.x,
                    y: firstResult.y
                };

                this.showSelectedInfo(this.selectedPoint);
                this.requestRender();
            }
        } catch (error) {
            console.error('Search error:', error);
        }
    }

    switchClassification(classification) {
        this.metadataCache.clear();
        this.loadData(classification);
    }

    formatNumber(num) {
        const n = parseInt(num);
        if (isNaN(n)) return num;
        if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
        if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
        return n.toLocaleString();
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize app
const app = new TSNEVisualization();
