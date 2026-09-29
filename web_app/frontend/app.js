/**
 * High-performance t-SNE visualization using Canvas and spatial indexing.
 *
 * Coordinates are fetched as a packed binary buffer (11-byte id + float32 x
 * + float32 y + uint8 category_id per point) and read into typed arrays -
 * no JSON parsing of 100k+ point objects. Category colors/names are a small
 * fixed palette fetched separately and looked up by id.
 */

const API_BASE = window.location.hostname === 'localhost' ? 'http://localhost:5000' : '';
const RECORD_SIZE = 20; // 11s + f + f + B

function hexToRgb(hex) {
    hex = hex.replace('#', '');
    return [
        parseInt(hex.substring(0, 2), 16),
        parseInt(hex.substring(2, 4), 16),
        parseInt(hex.substring(4, 6), 16),
    ];
}

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
        this.classification = 'format';
        this.pointCount = 0;
        this.ids = null;       // string[]
        this.xs = null;        // Float32Array
        this.ys = null;        // Float32Array
        this.catIds = null;    // Uint8Array
        this.categories = [];  // [{id, name, colorHex, colorRGB, count, centroidX, centroidY}]
        this.bounds = null;

        this.scale = 1;
        this.translateX = 0;
        this.translateY = 0;
        this.isDragging = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;
        this.hoveredPoint = null;
        this.selectedPoint = null;
        this.highlightCategoryId = null; // set while hovering a legend item
        this.isolatedCategoryId = null;  // set by clicking a legend item - sticky until cleared

        // Performance
        this.spatialIndex = new SpatialIndex(50);
        this.metadataCache = new Map();
        this.renderRequested = false;

        // Constants
        this.DOT_RADIUS = 2.5;
        this.HOVER_RADIUS = 15;
        this.MIN_SCALE = 0.3;
        this.MAX_SCALE = 10;
        this.LABEL_MIN_SCALE = 1.3;

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

            this.ctx.setTransform(1, 0, 0, 1, 0, 0);
            this.ctx.scale(dpr, dpr);

            this.canvasWidth = rect.width;
            this.canvasHeight = rect.height;

            this.requestRender();
        };

        updateSize();
        window.addEventListener('resize', updateSize);
    }

    setupEventListeners() {
        this.canvas.addEventListener('mousedown', this.handleMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.handleMouseUp.bind(this));
        this.canvas.addEventListener('mouseleave', this.handleMouseUp.bind(this));
        this.canvas.addEventListener('wheel', this.handleWheel.bind(this));
        this.canvas.addEventListener('click', this.handleClick.bind(this));

        document.getElementById('zoom-in').addEventListener('click', () => this.zoom(1.3));
        document.getElementById('zoom-out').addEventListener('click', () => this.zoom(0.7));
        document.getElementById('reset-view').addEventListener('click', () => this.resetView());

        document.querySelectorAll('.toggle-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const classification = e.target.dataset.classification;
                if (classification !== this.classification) {
                    this.switchClassification(classification);
                }
            });
        });

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

            // cache: 'no-cache' forces a real revalidation request every
            // load instead of trusting a previously cached response's
            // freshness - needed once, since these URLs were briefly served
            // with an `immutable` Cache-Control and some browsers will
            // honor that forever otherwise, never re-checking even after
            // the server starts sending correct ETags. Cheap: the server
            // still answers with a 304 when nothing actually changed.
            const [coordsResp, catsResp] = await Promise.all([
                fetch(`${API_BASE}/api/coordinates/${classification}`, { cache: 'no-cache' }),
                fetch(`${API_BASE}/api/categories/${classification}`, { cache: 'no-cache' }),
            ]);

            const buffer = await coordsResp.arrayBuffer();
            const categories = await catsResp.json();

            this.parseBuffer(buffer);
            this.categories = categories.map(c => ({ ...c, colorRGB: hexToRgb(c.colorHex) }));
            this.classification = classification;

            document.getElementById('video-count').textContent = `${this.pointCount.toLocaleString()} videos`;
            document.querySelectorAll('.toggle-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.classification === classification);
            });

            this.renderLegend();
            this.buildSpatialIndex();
            this.resetView();

            document.getElementById('loading').style.display = 'none';
        } catch (error) {
            console.error('Error loading data:', error);
            document.getElementById('loading').innerHTML = '<div class="placeholder">Error loading data</div>';
        }
    }

    /** Parse the packed binary buffer into typed arrays (structure-of-arrays). */
    parseBuffer(buffer) {
        const n = Math.floor(buffer.byteLength / RECORD_SIZE);
        this.pointCount = n;
        this.ids = new Array(n);
        this.xs = new Float32Array(n);
        this.ys = new Float32Array(n);
        this.catIds = new Uint8Array(n);

        const bytes = new Uint8Array(buffer);
        const view = new DataView(buffer);
        const decoder = new TextDecoder('ascii');

        let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;

        for (let i = 0; i < n; i++) {
            const off = i * RECORD_SIZE;
            this.ids[i] = decoder.decode(bytes.subarray(off, off + 11));
            const x = view.getFloat32(off + 11, true);
            const y = view.getFloat32(off + 15, true);
            this.xs[i] = x;
            this.ys[i] = y;
            this.catIds[i] = view.getUint8(off + 19);

            if (x < xMin) xMin = x;
            if (x > xMax) xMax = x;
            if (y < yMin) yMin = y;
            if (y > yMax) yMax = y;
        }

        this.bounds = { x_min: xMin, x_max: xMax, y_min: yMin, y_max: yMax };
    }

    renderLegend() {
        const container = document.getElementById('legend-list');
        const sorted = [...this.categories].sort((a, b) => b.count - a.count);
        this.categoriesById = new Map(this.categories.map(c => [c.id, c]));
        this.isolatedCategoryId = null;

        container.innerHTML = sorted.map(c => `
            <div class="legend-item" data-category-id="${c.id}">
                <span class="legend-swatch" style="background:${c.colorHex}"></span>
                <span class="legend-name">${this.escapeHtml(c.name)}</span>
                <span class="legend-count">${c.count.toLocaleString()}</span>
            </div>
        `).join('');

        const tooltip = document.getElementById('legend-tooltip');

        container.querySelectorAll('.legend-item').forEach(el => {
            const catId = parseInt(el.dataset.categoryId, 10);

            el.addEventListener('mouseenter', (e) => {
                this.highlightCategoryId = catId;
                this.requestRender();
                this.showLegendTooltip(catId, e.clientX, e.clientY);
            });
            el.addEventListener('mousemove', (e) => {
                this.positionLegendTooltip(e.clientX, e.clientY);
            });
            el.addEventListener('mouseleave', () => {
                this.highlightCategoryId = null;
                this.requestRender();
                tooltip.style.display = 'none';
            });
            // Click pins the isolation: only this category's dots stay
            // visible until you click it again or click empty canvas space.
            el.addEventListener('click', () => {
                this.isolatedCategoryId = this.isolatedCategoryId === catId ? null : catId;
                this.updateLegendActiveStates();
                this.requestRender();
            });
        });

        this.updateLegendActiveStates();
    }

    updateLegendActiveStates() {
        document.querySelectorAll('.legend-item').forEach(el => {
            const catId = parseInt(el.dataset.categoryId, 10);
            el.classList.toggle('isolated', catId === this.isolatedCategoryId);
        });
    }

    /** Breakdown tooltip: shows the finer-grained tags inside a legend
     *  color, so it's clear why one color can appear as separate blobs. */
    showLegendTooltip(categoryId, clientX, clientY) {
        const cat = this.categoriesById.get(categoryId);
        const tooltip = document.getElementById('legend-tooltip');
        if (!cat) return;

        const subs = cat.subcategories || [];
        const rows = subs.length
            ? subs.map(s => `
                <div class="legend-tooltip-row">
                    <span class="legend-tooltip-name">${this.escapeHtml(s.name)}</span>
                    <span class="legend-tooltip-count">${s.count.toLocaleString()}</span>
                </div>
            `).join('')
            : '<div class="placeholder-inline">No further breakdown</div>';

        tooltip.innerHTML = `
            <div class="legend-tooltip-header">
                <span class="legend-swatch" style="background:${cat.colorHex}"></span>
                ${this.escapeHtml(cat.name)}
            </div>
            ${rows}
        `;
        tooltip.style.display = 'block';
        this.positionLegendTooltip(clientX, clientY);
    }

    positionLegendTooltip(clientX, clientY) {
        const tooltip = document.getElementById('legend-tooltip');
        if (tooltip.style.display === 'none') return;

        const gap = 14;
        const maxX = window.innerWidth - tooltip.offsetWidth - 8;
        const maxY = window.innerHeight - tooltip.offsetHeight - 8;

        tooltip.style.left = `${Math.min(clientX + gap, maxX)}px`;
        tooltip.style.top = `${Math.min(clientY, maxY)}px`;
    }

    buildSpatialIndex() {
        this.spatialIndex.clear();
        if (!this.pointCount) return;

        for (let i = 0; i < this.pointCount; i++) {
            const [x, y] = this.projectToCanvas(this.xs[i], this.ys[i]);
            this.spatialIndex.insert({ index: i }, x, y);
        }
    }

    projectToCanvas(x, y) {
        if (!this.bounds) return [0, 0];

        const margin = 50;
        const plotWidth = this.canvasWidth - 2 * margin;
        const plotHeight = this.canvasHeight - 2 * margin;

        const normalizedX = (x - this.bounds.x_min) / (this.bounds.x_max - this.bounds.x_min);
        const normalizedY = (y - this.bounds.y_min) / (this.bounds.y_max - this.bounds.y_min);

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
        if (!this.pointCount) return;

        const ctx = this.ctx;

        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, this.canvasWidth, this.canvasHeight);

        ctx.save();
        ctx.translate(this.translateX, this.translateY);
        ctx.scale(this.scale, this.scale);

        const [minWorldX, minWorldY] = this.screenToWorld(0, 0);
        const [maxWorldX, maxWorldY] = this.screenToWorld(this.canvasWidth, this.canvasHeight);

        const dotRadius = this.DOT_RADIUS / this.scale;
        const categories = this.categories;
        // A click-isolated category wins over a merely-hovered one; either
        // way this is the id whose dots get pulled out and emphasized.
        const isolating = this.isolatedCategoryId !== null;
        const highlightId = isolating ? this.isolatedCategoryId : this.highlightCategoryId;

        // Hovering a legend item dims every other category to gray so the
        // highlighted one's footprint stands out. Clicking it goes further
        // and hides the rest entirely, so only that category's dots remain
        // clickable/visible until it's un-isolated.
        const highlightedIndices = highlightId !== null ? [] : null;

        for (let i = 0; i < this.pointCount; i++) {
            const [x, y] = this.projectToCanvas(this.xs[i], this.ys[i]);

            if (x < minWorldX - 50 || x > maxWorldX + 50 ||
                y < minWorldY - 50 || y > maxWorldY + 50) {
                continue;
            }

            const catId = this.catIds[i];

            if (highlightId !== null) {
                if (catId === highlightId) {
                    highlightedIndices.push(i);
                    continue; // drawn in a second, on-top pass below
                }
                if (isolating) {
                    continue; // hidden entirely while isolated
                }
                ctx.fillStyle = '#3a3a3a';
                ctx.globalAlpha = 0.25;
                ctx.beginPath();
                ctx.arc(x, y, dotRadius, 0, Math.PI * 2);
                ctx.fill();
                continue;
            }

            const cat = categories[catId];
            const rgb = cat ? cat.colorRGB : [128, 128, 128];

            ctx.fillStyle = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
            ctx.globalAlpha = 0.75;
            ctx.beginPath();
            ctx.arc(x, y, dotRadius, 0, Math.PI * 2);
            ctx.fill();
        }

        if (highlightedIndices) {
            const cat = categories[highlightId];
            const rgb = cat ? cat.colorRGB : [255, 255, 255];
            ctx.fillStyle = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
            ctx.globalAlpha = 1;
            const r = dotRadius * 1.4;
            for (const i of highlightedIndices) {
                const [x, y] = this.projectToCanvas(this.xs[i], this.ys[i]);
                ctx.beginPath();
                ctx.arc(x, y, r, 0, Math.PI * 2);
                ctx.fill();
            }

            // One label per sub-blob, at its own centroid (not the parent
            // category's, which can land in the gap between two blobs) -
            // sized by how much of the category it accounts for, so the
            // biggest chunk reads first.
            const subs = (cat && cat.subcategories) || [];
            if (subs.length) {
                const maxCount = Math.max(...subs.map(s => s.count));
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.strokeStyle = 'rgba(0,0,0,0.9)';
                ctx.fillStyle = '#ffffff';
                for (const sub of subs) {
                    const [x, y] = this.projectToCanvas(sub.centroidX, sub.centroidY);
                    const weight = Math.sqrt(sub.count / maxCount); // 0..1
                    const fontPx = (12 + 12 * weight) / this.scale;
                    ctx.font = `${weight > 0.55 ? 'bold ' : ''}${fontPx}px sans-serif`;
                    ctx.lineWidth = (3 + 2 * weight) / this.scale;
                    ctx.strokeText(sub.name, x, y);
                    ctx.fillText(sub.name, x, y);
                }
            }
        } else if (this.scale > this.LABEL_MIN_SCALE) {
            // Otherwise, category labels only at higher zoom, fading in -
            // keeps the base view clean like the paper figure (legend
            // carries names at rest).
            ctx.globalAlpha = Math.min(1, (this.scale - this.LABEL_MIN_SCALE) / 0.6);
            ctx.fillStyle = '#ffffff';
            ctx.strokeStyle = 'rgba(0,0,0,0.85)';
            ctx.lineWidth = 3 / this.scale;
            ctx.font = `${13 / this.scale}px sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            for (const cat of categories) {
                const [x, y] = this.projectToCanvas(cat.centroidX, cat.centroidY);
                ctx.strokeText(cat.name, x, y);
                ctx.fillText(cat.name, x, y);
            }
        }

        if (this.hoveredPoint) {
            const [x, y] = this.projectToCanvas(this.xs[this.hoveredPoint.index], this.ys[this.hoveredPoint.index]);
            ctx.globalAlpha = 1;
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2 / this.scale;
            ctx.beginPath();
            ctx.arc(x, y, (dotRadius + 3 / this.scale), 0, Math.PI * 2);
            ctx.stroke();
        }

        if (this.selectedPoint) {
            const [x, y] = this.projectToCanvas(this.xs[this.selectedPoint.index], this.ys[this.selectedPoint.index]);
            ctx.globalAlpha = 1;
            ctx.strokeStyle = '#00d4ff';
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
            let point = this.spatialIndex.query(worldX, worldY, this.HOVER_RADIUS / this.scale);

            // While isolated, hidden dots (any other category) shouldn't be
            // hoverable/clickable even if the cursor lands where one used
            // to be visually - otherwise a click there would look like it
            // did nothing instead of un-isolating.
            if (point && this.isolatedCategoryId !== null && this.catIds[point.index] !== this.isolatedCategoryId) {
                point = null;
            }

            const changed = (point?.index) !== (this.hoveredPoint?.index);
            if (changed) {
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

    handleClick() {
        if (this.hoveredPoint) {
            this.selectedPoint = this.hoveredPoint;
            this.showSelectedInfo(this.selectedPoint);
            this.requestRender();
        } else if (this.isolatedCategoryId !== null) {
            // Clicked empty space (or a now-hidden dot's old spot) while
            // isolated - bring the full plot back.
            this.isolatedCategoryId = null;
            this.updateLegendActiveStates();
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
        this.isolatedCategoryId = null;
        this.updateLegendActiveStates();

        this.buildSpatialIndex();
        this.clearHoverInfo();
        this.clearSelectedInfo();
        this.requestRender();
    }

    categoryBadge(name) {
        return `<span class="category-badge">${this.escapeHtml(name)}</span>`;
    }

    /**
     * Render up to 3 hierarchy breadcrumbs (each a "Top > Mid > Leaf"
     * string from the server) as nested chip chains, instead of a flat,
     * same-looking pile of badges.
     */
    renderTagPaths(paths) {
        if (!paths || !paths.length) {
            return '<span class="placeholder-inline">N/A</span>';
        }
        return paths.slice(0, 3).map(path => {
            const parts = path.split('>').map(p => p.trim()).filter(Boolean);
            return `<div class="tag-path">${parts
                .map(p => `<span class="tag-crumb">${this.escapeHtml(p)}</span>`)
                .join('<span class="tag-sep">›</span>')}</div>`;
        }).join('');
    }

    /** Standardized "Format: ... / Content: ..." block shown regardless of
     *  which classification is currently being browsed. */
    renderClassificationBlock(label, category, tags) {
        return `
            <div class="metadata-item">
                <div class="metadata-label">${label}</div>
                <div class="metadata-value">${this.categoryBadge(category)}</div>
                <div class="tag-path-list">${this.renderTagPaths(tags)}</div>
            </div>
        `;
    }

    async showHoverInfo(point) {
        const videoId = this.ids[point.index];
        const metadata = await this.fetchMetadata(videoId);
        if (!metadata) return;

        const thumbnailContainer = document.getElementById('thumbnail-preview');
        thumbnailContainer.innerHTML = `
            <img src="https://img.youtube.com/vi/${videoId}/maxresdefault.jpg"
                 alt="Thumbnail"
                 onerror="this.src='https://img.youtube.com/vi/${videoId}/hqdefault.jpg'">
        `;

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
                <div class="metadata-label">Format</div>
                <div class="metadata-value">${this.categoryBadge(metadata.formatCategory)}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Content</div>
                <div class="metadata-value">${this.categoryBadge(metadata.contentCategory)}</div>
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
        const videoId = this.ids[point.index];
        const metadata = await this.fetchMetadata(videoId);
        if (!metadata) return;

        const videoContainer = document.getElementById('video-embed');
        videoContainer.innerHTML = `
            <iframe
                src="https://www.youtube.com/embed/${videoId}?autoplay=1"
                title="YouTube video player"
                frameborder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowfullscreen>
            </iframe>
        `;

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
            ${this.renderClassificationBlock('Format', metadata.formatCategory, metadata.formatTags)}
            ${this.renderClassificationBlock('Content', metadata.contentCategory, metadata.contentTags)}
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

                // Find the local index for this id so canvas highlighting works.
                const idx = this.ids.indexOf(firstResult.id);
                this.selectedPoint = idx >= 0 ? { index: idx } : null;

                this.showSelectedInfoFromMetadata(firstResult);
                this.requestRender();
            }
        } catch (error) {
            console.error('Search error:', error);
        }
    }

    showSelectedInfoFromMetadata(metadata) {
        this.metadataCache.set(metadata.id, metadata);
        this.showSelectedInfo({ index: this.ids.indexOf(metadata.id) });
    }

    switchClassification(classification) {
        this.metadataCache.clear();
        this.loadData(classification);
    }

    formatNumber(num) {
        const n = parseInt(String(num).replace(/,/g, ''), 10);
        if (isNaN(n)) return num;
        if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
        if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
        return n.toLocaleString();
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text == null ? '' : text;
        return div.innerHTML;
    }
}

const app = new TSNEVisualization();
