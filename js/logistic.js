/**
 * Logistic Regression Logic
 */

const canvas = document.getElementById('main-canvas');
const ctx = canvas.getContext('2d');

// State
const state = {
    points: [],
    currentClass: 0,
    threshold: 0.5,
    learningRate: 0.1,
    regularization: 0,
    noise: 0.1,
    draggingIndex: -1,
    weights: { w1: 0, w2: 0, b: 0 }
};

// Elements
const threshSlider = document.getElementById('thresh-slider');
const threshDisplay = document.getElementById('thresh-display');
const lrSlider = document.getElementById('lr-slider');
const lrDisplay = document.getElementById('lr-display');
const regSlider = document.getElementById('reg-slider');
const regDisplay = document.getElementById('reg-display');
const noiseSlider = document.getElementById('noise-slider');
const noiseDisplay = document.getElementById('noise-display');
const regenBtn = document.getElementById('regen-data-btn');
const classBtns = document.querySelectorAll('.class-btn');

// Metrics Elements
const accEl = document.getElementById('acc-display');
const precEl = document.getElementById('prec-display');
const recEl = document.getElementById('rec-display');
const cmTn = document.getElementById('cm-tn');
const cmFp = document.getElementById('cm-fp');
const cmFn = document.getElementById('cm-fn');
const cmTp = document.getElementById('cm-tp');

function init() {
    resizeCanvas();
    generateData();
    train();
    draw();
}

function generateData() {
    state.points = Utils.generateClassificationData(40, state.noise, 0.3);
}

function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

function train() {
    if (state.points.length < 2) return;

    // Reset weights
    let w1 = 0, w2 = 0, b = 0;
    const lr = state.learningRate;
    const lambda = state.regularization;
    const epochs = 2000; // Fast enough for JS

    for (let i = 0; i < epochs; i++) {
        let dw1 = 0, dw2 = 0, db = 0;
        for (let p of state.points) {
            const z = w1 * p.x + w2 * p.y + b;
            const y_pred = sigmoid(z);
            const error = y_pred - p.label;

            dw1 += error * p.x;
            dw2 += error * p.y;
            db += error;
        }

        // L2 Regularization gradients
        dw1 += lambda * w1;
        dw2 += lambda * w2;

        w1 -= lr * dw1 / state.points.length;
        w2 -= lr * dw2 / state.points.length;
        b -= lr * db / state.points.length;
    }
    state.weights = { w1, w2, b };
}

function predict(x, y) {
    const { w1, w2, b } = state.weights;
    return sigmoid(w1 * x + w2 * y + b);
}

function drawDecisionBoundary() {
    const w = canvas.width, h = canvas.height, p = 40;
    const imgData = ctx.createImageData(w, h);
    const data = imgData.data;

    // Low res scan for performance (skip pixels)
    const step = 4;
    for (let py = 0; py < h; py += step) {
        for (let px = 0; px < w; px += step) {
            const { x, y } = Utils.fromCanvasCoords(px, py, canvas);
            const prob = predict(x, y);

            let r, g, b, a;
            if (prob < state.threshold) {
                // Red Class 0
                r = 239; g = 68; b = 68;
                a = 40 * (1 - prob); // Fade near boundary
            } else {
                // Blue Class 1
                r = 59; g = 130; b = 246;
                a = 40 * prob;
            }

            // Fill block
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

    if (state.points.length > 1) drawDecisionBoundary();
    Utils.drawGrid(ctx, canvas);

    // Draw Points
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

    // Draw Decision Line (w1*x + w2*y + b = logit(threshold))
    // logit(p) = ln(p / (1-p))
    // w1*x + w2*y + b = T  => y = (T - b - w1*x) / w2
    if (state.points.length > 1 && Math.abs(state.weights.w2) > 0.001) {
        const T = Math.log(state.threshold / (1 - state.threshold));
        const { w1, w2, b } = state.weights;

        ctx.beginPath();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);

        // Calculate two points to draw line
        const x1 = 0, y1 = (T - b - w1 * x1) / w2;
        const x2 = 1, y2 = (T - b - w1 * x2) / w2;

        const p1 = Utils.toCanvasCoords(x1, y1, canvas);
        const p2 = Utils.toCanvasCoords(x2, y2, canvas);

        ctx.moveTo(p1.cx, p1.cy);
        ctx.lineTo(p2.cx, p2.cy);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    updateMetrics();
}

function updateMetrics() {
    const metrics = Utils.calculateClassificationMetrics(state.points, (x, y) => {
        return predict(x, y) >= state.threshold ? 1 : 0; // Use threshold for binary pred
    });

    accEl.textContent = Math.round(metrics.accuracy * 100) + '%';
    precEl.textContent = metrics.precision.toFixed(2);
    recEl.textContent = metrics.recall.toFixed(2);

    cmTn.textContent = metrics.confusion[0];
    cmFp.textContent = metrics.confusion[1];
    cmFn.textContent = metrics.confusion[2];
    cmTp.textContent = metrics.confusion[3];
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
        train();
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
        train();
        draw();
    }
});

window.addEventListener('mouseup', () => state.draggingIndex = -1);
window.addEventListener('resize', resizeCanvas);

threshSlider.addEventListener('input', (e) => {
    state.threshold = parseFloat(e.target.value);
    threshDisplay.textContent = state.threshold.toFixed(2);
    draw();
});

lrSlider.addEventListener('input', (e) => {
    state.learningRate = parseFloat(e.target.value);
    lrDisplay.textContent = state.learningRate;
    train();
    draw();
});

regSlider.addEventListener('input', (e) => {
    state.regularization = parseFloat(e.target.value);
    regDisplay.textContent = state.regularization;
    train();
    draw();
});

noiseSlider.addEventListener('input', (e) => {
    state.noise = parseFloat(e.target.value);
    noiseDisplay.textContent = state.noise;
    generateData();
    train();
    draw();
});

regenBtn.addEventListener('click', () => {
    generateData();
    train();
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
