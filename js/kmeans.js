/**
 * K-Means Logic
 */

const canvas = document.getElementById('main-canvas');
const ctx = canvas.getContext('2d');

// State
const state = {
    points: [],
    centroids: [],
    k: 3,
    shape: 'blobs',
    iteration: 0,
    converged: false,
    draggingIndex: -1,
    history: [] // For animation if needed
};

// Elements
const kSlider = document.getElementById('k-slider');
const kDisplay = document.getElementById('k-display');
const shapeSelect = document.getElementById('shape-select');
const iterDisplay = document.getElementById('iter-display');
const inertiaDisplay = document.getElementById('inertia-display');
const stepBtn = document.getElementById('step-btn');
const runBtn = document.getElementById('run-btn');
const resetBtn = document.getElementById('reset-centroids-btn');
const regenBtn = document.getElementById('regen-data-btn');
const elbowChart = document.getElementById('elbow-chart');
const calcElbowBtn = document.getElementById('calc-elbow-btn');

const colors = ['#ef4444', '#3b82f6', '#22c55e', '#fbbf24', '#a855f7', '#ec4899', '#06b6d4', '#f97316'];

function init() {
    resizeCanvas();
    generateData();
    initCentroids();
    draw();
}

function generateData() {
    state.points = [];
    const n = 60;

    if (state.shape === 'blobs') {
        // Generate K blobs roughly
        for (let i = 0; i < state.k; i++) {
            const cx = 0.2 + Math.random() * 0.6;
            const cy = 0.2 + Math.random() * 0.6;
            for (let j = 0; j < n / state.k; j++) {
                state.points.push({
                    x: cx + (Math.random() - 0.5) * 0.15,
                    y: cy + (Math.random() - 0.5) * 0.15,
                    cluster: -1
                });
            }
        }
    } else if (state.shape === 'circles') {
        for (let i = 0; i < n; i++) {
            const angle = Math.random() * Math.PI * 2;
            const r = i < n / 2 ? 0.2 : 0.4;
            state.points.push({
                x: 0.5 + Math.cos(angle) * r + (Math.random() - 0.5) * 0.05,
                y: 0.5 + Math.sin(angle) * r + (Math.random() - 0.5) * 0.05,
                cluster: -1
            });
        }
    } else if (state.shape === 'moons') {
        for (let i = 0; i < n / 2; i++) {
            const x = i / (n / 2) * Math.PI;
            state.points.push({
                x: 0.3 + x / Math.PI * 0.4 + (Math.random() - 0.5) * 0.05,
                y: 0.3 + Math.sin(x) * 0.2 + (Math.random() - 0.5) * 0.05,
                cluster: -1
            });
        }
        for (let i = 0; i < n / 2; i++) {
            const x = i / (n / 2) * Math.PI;
            state.points.push({
                x: 0.7 - x / Math.PI * 0.4 + (Math.random() - 0.5) * 0.05,
                y: 0.7 - Math.sin(x) * 0.2 + (Math.random() - 0.5) * 0.05,
                cluster: -1
            });
        }
    } else {
        for (let i = 0; i < n; i++) {
            state.points.push({ x: Math.random(), y: Math.random(), cluster: -1 });
        }
    }
}

function initCentroids() {
    state.centroids = [];
    // Random initialization
    for (let i = 0; i < state.k; i++) {
        state.centroids.push({
            x: Math.random(),
            y: Math.random(),
            color: colors[i % colors.length]
        });
    }
    state.iteration = 0;
    state.converged = false;
    assignClusters(); // Initial assignment
}

function assignClusters() {
    let changed = false;
    state.points.forEach(p => {
        let minDist = Infinity;
        let bestK = -1;
        state.centroids.forEach((c, i) => {
            const d = (p.x - c.x) ** 2 + (p.y - c.y) ** 2;
            if (d < minDist) {
                minDist = d;
                bestK = i;
            }
        });
        if (p.cluster !== bestK) changed = true;
        p.cluster = bestK;
    });
    return changed;
}

function updateCentroids() {
    const sums = new Array(state.k).fill(0).map(() => ({ x: 0, y: 0, count: 0 }));

    state.points.forEach(p => {
        if (p.cluster !== -1) {
            sums[p.cluster].x += p.x;
            sums[p.cluster].y += p.y;
            sums[p.cluster].count++;
        }
    });

    let maxShift = 0;
    state.centroids.forEach((c, i) => {
        if (sums[i].count > 0) {
            const newX = sums[i].x / sums[i].count;
            const newY = sums[i].y / sums[i].count;
            const shift = Math.sqrt((c.x - newX) ** 2 + (c.y - newY) ** 2);
            if (shift > maxShift) maxShift = shift;
            c.x = newX;
            c.y = newY;
        } else {
            // Re-init empty cluster
            c.x = Math.random();
            c.y = Math.random();
        }
    });

    state.iteration++;
    if (maxShift < 0.001) state.converged = true;
}

function calculateInertia() {
    let inertia = 0;
    state.points.forEach(p => {
        if (p.cluster !== -1) {
            const c = state.centroids[p.cluster];
            inertia += (p.x - c.x) ** 2 + (p.y - c.y) ** 2;
        }
    });
    return inertia;
}

function drawVoronoi() {
    const w = canvas.width, h = canvas.height;
    const imgData = ctx.createImageData(w, h);
    const data = imgData.data;

    const step = 4;
    for (let py = 0; py < h; py += step) {
        for (let px = 0; px < w; px += step) {
            const { x, y } = Utils.fromCanvasCoords(px, py, canvas);

            let minDist = Infinity;
            let bestK = -1;
            for (let i = 0; i < state.k; i++) {
                const d = (x - state.centroids[i].x) ** 2 + (y - state.centroids[i].y) ** 2;
                if (d < minDist) {
                    minDist = d;
                    bestK = i;
                }
            }

            // Get color
            const hex = colors[bestK % colors.length];
            const r = parseInt(hex.slice(1, 3), 16);
            const g = parseInt(hex.slice(3, 5), 16);
            const b = parseInt(hex.slice(5, 7), 16);

            for (let dy = 0; dy < step; dy++) {
                for (let dx = 0; dx < step; dx++) {
                    if (py + dy >= h || px + dx >= w) continue;
                    const idx = ((py + dy) * w + (px + dx)) * 4;
                    data[idx] = r; data[idx + 1] = g; data[idx + 2] = b; data[idx + 3] = 20; // Low opacity
                }
            }
        }
    }
    ctx.putImageData(imgData, 0, 0);
}

function draw() {
    Utils.clearCanvas(ctx, canvas);
    if (state.centroids.length > 0) drawVoronoi();
    Utils.drawGrid(ctx, canvas);

    // Draw Connections
    ctx.lineWidth = 1;
    state.points.forEach(p => {
        if (p.cluster !== -1) {
            const pos = Utils.toCanvasCoords(p.x, p.y, canvas);
            const cPos = Utils.toCanvasCoords(state.centroids[p.cluster].x, state.centroids[p.cluster].y, canvas);
            ctx.beginPath();
            ctx.moveTo(pos.cx, pos.cy);
            ctx.lineTo(cPos.cx, cPos.cy);
            ctx.strokeStyle = state.centroids[p.cluster].color + '44'; // Transparent
            ctx.stroke();
        }
    });

    // Draw Points
    state.points.forEach(p => {
        const pos = Utils.toCanvasCoords(p.x, p.y, canvas);
        ctx.beginPath();
        ctx.arc(pos.cx, pos.cy, 5, 0, Math.PI * 2);
        ctx.fillStyle = p.cluster === -1 ? '#fff' : state.centroids[p.cluster].color;
        ctx.fill();
    });

    // Draw Centroids
    state.centroids.forEach((c, i) => {
        const pos = Utils.toCanvasCoords(c.x, c.y, canvas);
        ctx.beginPath();
        ctx.arc(pos.cx, pos.cy, 10, 0, Math.PI * 2);
        ctx.fillStyle = c.color;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 12px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('X', pos.cx, pos.cy);
    });

    iterDisplay.textContent = state.iteration;
    inertiaDisplay.textContent = calculateInertia().toFixed(4);
}

function resizeCanvas() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
    draw();
}

// Elbow Method
async function calculateElbow() {
    elbowChart.innerHTML = '<div style="color: #aaa; font-size: 0.8rem; width: 100%; text-align: center;">Calculating...</div>';
    const inertias = [];
    const maxK = 8;

    // Save state
    const savedK = state.k;
    const savedCentroids = [...state.centroids];
    const savedPoints = JSON.parse(JSON.stringify(state.points));

    for (let k = 1; k <= maxK; k++) {
        state.k = k;
        initCentroids();
        for (let i = 0; i < 10; i++) {
            assignClusters();
            updateCentroids();
        }
        inertias.push(calculateInertia());
    }

    // Restore state
    state.k = savedK;
    state.centroids = savedCentroids;
    state.points = savedPoints;
    draw();

    // Render Chart
    elbowChart.innerHTML = '';
    const maxInertia = Math.max(...inertias);
    inertias.forEach((val, i) => {
        const bar = document.createElement('div');
        bar.style.flex = '1';
        bar.style.background = i + 1 === state.k ? 'var(--accent-color)' : 'rgba(255,255,255,0.2)';
        bar.style.height = (val / maxInertia * 100) + '%';
        bar.style.borderRadius = '2px 2px 0 0';
        bar.title = `K=${i + 1}, Inertia=${val.toFixed(2)}`;
        elbowChart.appendChild(bar);
    });
}

// Interactions
canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const { x, y } = Utils.fromCanvasCoords(cx, cy, canvas);
    state.points.push({ x, y, cluster: -1 });
    assignClusters();
    draw();
});

window.addEventListener('resize', resizeCanvas);

kSlider.addEventListener('input', (e) => {
    state.k = parseInt(e.target.value);
    kDisplay.textContent = state.k;
    initCentroids();
    draw();
});

shapeSelect.addEventListener('change', (e) => {
    state.shape = e.target.value;
    generateData();
    initCentroids();
    draw();
});

stepBtn.addEventListener('click', () => {
    if (!state.converged) {
        updateCentroids();
        assignClusters();
        draw();
    }
});

runBtn.addEventListener('click', () => {
    let limit = 20;
    const interval = setInterval(() => {
        if (state.converged || limit-- <= 0) {
            clearInterval(interval);
        } else {
            updateCentroids();
            assignClusters();
            draw();
        }
    }, 100);
});

resetBtn.addEventListener('click', () => {
    initCentroids();
    draw();
});

regenBtn.addEventListener('click', () => {
    generateData();
    initCentroids();
    draw();
});

calcElbowBtn.addEventListener('click', calculateElbow);

init();
