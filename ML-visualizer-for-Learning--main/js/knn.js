/**
 * KNN Logic
 */

const canvas = document.getElementById('main-canvas');
const ctx = canvas.getContext('2d');

// State
const state = {
    points: [],
    currentClass: 0,
    k: 1,
    metric: 'euclidean',
    noise: 0.1,
    draggingIndex: -1
};

// Elements
const kSlider = document.getElementById('k-slider');
const kDisplay = document.getElementById('k-display');
const kHint = document.getElementById('k-hint');
const metricSelect = document.getElementById('metric-select');
const noiseSlider = document.getElementById('noise-slider');
const noiseDisplay = document.getElementById('noise-display');
const regenBtn = document.getElementById('regen-data-btn');
const classBtns = document.querySelectorAll('.class-btn');
const accEl = document.getElementById('acc-display');
const biasEl = document.getElementById('bias-display');
const varEl = document.getElementById('var-display');

function init() {
    resizeCanvas();
    generateData();
    draw();
}

function generateData() {
    state.points = Utils.generateClassificationData(40, state.noise, 0.3);
}

function getDistance(p1, x, y) {
    const dx = Math.abs(p1.x - x);
    const dy = Math.abs(p1.y - y);
    if (state.metric === 'euclidean') return Math.sqrt(dx * dx + dy * dy);
    return dx + dy;
}

function predict(x, y) {
    if (state.points.length === 0) return 0;

    const distances = state.points.map(p => ({
        dist: getDistance(p, x, y),
        label: p.label
    }));

    distances.sort((a, b) => a.dist - b.dist);

    const kNearest = distances.slice(0, state.k);
    const count0 = kNearest.filter(d => d.label === 0).length;

    // Return probability of class 1
    return 1 - (count0 / state.k);
}

function drawDecisionBoundary() {
    const w = canvas.width, h = canvas.height;
    const imgData = ctx.createImageData(w, h);
    const data = imgData.data;

    // Low res scan
    const step = 4;
    for (let py = 0; py < h; py += step) {
        for (let px = 0; px < w; px += step) {
            const { x, y } = Utils.fromCanvasCoords(px, py, canvas);
            const prob = predict(x, y);

            let r, g, b, a;
            if (prob < 0.5) {
                r = 239; g = 68; b = 68;
                a = 40 + (1 - prob) * 20;
            } else {
                r = 59; g = 130; b = 246;
                a = 40 + prob * 20;
            }

            for (let dy = 0; dy < step; dy++) {
                for (let dx = 0; dx < step; dx++) {
                    if (py + dy >= h || px + dx >= w) continue;
                    const idx = ((py + dy) * w + (px + dx)) * 4;
                    data[idx] = r; data[idx + 1] = g; data[idx + 2] = b; data[idx + 3] = a;
                }
            }
        }
    }
    ctx.putImageData(imgData, 0, 0);
}

function draw() {
    Utils.clearCanvas(ctx, canvas);
    if (state.points.length > 0) drawDecisionBoundary();
    Utils.drawGrid(ctx, canvas);

    state.points.forEach((p, i) => {
        const pos = Utils.toCanvasCoords(p.x, p.y, canvas);
        ctx.beginPath();
        ctx.arc(pos.cx, pos.cy, 7, 0, Math.PI * 2);
        ctx.fillStyle = p.label === 0 ? '#ef4444' : '#3b82f6';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
    });

    updateMetrics();
}

function updateMetrics() {
    // Leave-One-Out Cross Validation for Accuracy estimate
    let correct = 0;
    if (state.points.length > 1) {
        state.points.forEach((p, i) => {
            // Predict using all OTHER points
            const others = state.points.filter((_, idx) => idx !== i);

            const distances = others.map(o => ({
                dist: getDistance(o, p.x, p.y),
                label: o.label
            })).sort((a, b) => a.dist - b.dist).slice(0, state.k);

            const c0 = distances.filter(d => d.label === 0).length;
            const pred = c0 > distances.length / 2 ? 0 : 1;

            if (pred === p.label) correct++;
        });
        accEl.textContent = Math.round((correct / state.points.length) * 100) + '%';
    } else {
        accEl.textContent = '0%';
    }

    // Bias/Variance Heuristic
    if (state.k === 1) {
        biasEl.textContent = "Low";
        varEl.textContent = "High (Overfitting)";
        kHint.textContent = "K=1: Captures noise, jagged boundaries.";
    } else if (state.k > 10) {
        biasEl.textContent = "High";
        varEl.textContent = "Low (Underfitting)";
        kHint.textContent = `K=${state.k}: Smooth boundaries, ignores local details.`;
    } else {
        biasEl.textContent = "Balanced";
        varEl.textContent = "Balanced";
        kHint.textContent = `K=${state.k}: Good balance.`;
    }
}

function resizeCanvas() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
    draw();
}

// Interactions
canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const { x, y } = Utils.fromCanvasCoords(cx, cy, canvas);

    let found = -1;
    for (let i = 0; i < state.points.length; i++) {
        const pos = Utils.toCanvasCoords(state.points[i].x, state.points[i].y, canvas);
        if (Math.hypot(pos.cx - cx, pos.cy - cy) < 15) { found = i; break; }
    }

    if (found !== -1) {
        state.draggingIndex = found;
    } else {
        state.points.push({ x, y, label: state.currentClass });
        draw();
    }
});

window.addEventListener('mousemove', (e) => {
    if (state.draggingIndex !== -1) {
        const rect = canvas.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        const { x, y } = Utils.fromCanvasCoords(cx, cy, canvas);
        state.points[state.draggingIndex].x = x;
        state.points[state.draggingIndex].y = y;
        draw();
    }
});

window.addEventListener('mouseup', () => state.draggingIndex = -1);
window.addEventListener('resize', resizeCanvas);

kSlider.addEventListener('input', (e) => {
    state.k = parseInt(e.target.value);
    kDisplay.textContent = state.k;
    draw();
});

metricSelect.addEventListener('change', (e) => {
    state.metric = e.target.value;
    draw();
});

noiseSlider.addEventListener('input', (e) => {
    state.noise = parseFloat(e.target.value);
    noiseDisplay.textContent = state.noise;
    generateData();
    draw();
});

regenBtn.addEventListener('click', () => {
    generateData();
    draw();
});

classBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        classBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.currentClass = parseInt(btn.dataset.class);
    });
});

init();
