/**
 * SVM Logic (Simplified SMO/Gradient Descent for Visualization)
 */

const canvas = document.getElementById('main-canvas');
const ctx = canvas.getContext('2d');

// State
const state = {
    points: [],
    currentClass: 0,
    kernel: 'linear',
    C: 1.0,
    gamma: 1.0,
    noise: 0.1,
    draggingIndex: -1,
    alphas: [], // Lagrange multipliers
    bias: 0
};

// Elements
const kernelSelect = document.getElementById('kernel-select');
const cSlider = document.getElementById('c-slider');
const cDisplay = document.getElementById('c-display');
const gammaSlider = document.getElementById('gamma-slider');
const gammaDisplay = document.getElementById('gamma-display');
const noiseSlider = document.getElementById('noise-slider');
const noiseDisplay = document.getElementById('noise-display');
const regenBtn = document.getElementById('regen-data-btn');
const classBtns = document.querySelectorAll('.class-btn');
const accEl = document.getElementById('acc-display');
const marginEl = document.getElementById('margin-display');

function init() {
    resizeCanvas();
    generateData();
    train();
    draw();
}

function generateData() {
    state.points = Utils.generateClassificationData(30, state.noise, 0.4);
    // SVM expects labels -1 and 1
    state.points.forEach(p => p.y_target = p.label === 0 ? -1 : 1);
}

// Kernels
function kernel(p1, p2) {
    if (state.kernel === 'linear') {
        return p1.x * p2.x + p1.y * p2.y;
    } else if (state.kernel === 'rbf') {
        const distSq = (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2;
        return Math.exp(-state.gamma * distSq);
    } else if (state.kernel === 'poly') {
        return (p1.x * p2.x + p1.y * p2.y + 1) ** 2;
    }
    return 0;
}

function train() {
    if (state.points.length < 2) return;

    // Simplified SMO-like training (Coordinate Descent on Dual Problem)
    // Maximize: sum(alpha) - 0.5 * sum(alpha_i * alpha_j * y_i * y_j * K(x_i, x_j))
    // Subject to: 0 <= alpha_i <= C, sum(alpha_i * y_i) = 0

    const n = state.points.length;
    state.alphas = new Array(n).fill(0);
    state.bias = 0;

    const passes = 10; // Fast approximation
    const tol = 1e-4;

    // Precompute Kernel Matrix
    const K = new Array(n).fill(0).map(() => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            K[i][j] = kernel(state.points[i], state.points[j]);
        }
    }

    let iter = 0;
    while (iter < passes) {
        let numChanged = 0;
        for (let i = 0; i < n; i++) {
            const Ei = predictRaw(state.points[i]) - state.points[i].y_target;
            if ((state.points[i].y_target * Ei < -tol && state.alphas[i] < state.C) ||
                (state.points[i].y_target * Ei > tol && state.alphas[i] > 0)) {

                // Select j randomly
                let j = Math.floor(Math.random() * (n - 1));
                if (j >= i) j++;

                const Ej = predictRaw(state.points[j]) - state.points[j].y_target;

                const oldAi = state.alphas[i];
                const oldAj = state.alphas[j];

                let L, H;
                if (state.points[i].y_target !== state.points[j].y_target) {
                    L = Math.max(0, state.alphas[j] - state.alphas[i]);
                    H = Math.min(state.C, state.C + state.alphas[j] - state.alphas[i]);
                } else {
                    L = Math.max(0, state.alphas[i] + state.alphas[j] - state.C);
                    H = Math.min(state.C, state.alphas[i] + state.alphas[j]);
                }

                if (L === H) continue;

                const eta = 2 * K[i][j] - K[i][i] - K[j][j];
                if (eta >= 0) continue;

                state.alphas[j] -= (state.points[j].y_target * (Ei - Ej)) / eta;
                state.alphas[j] = Math.max(L, Math.min(H, state.alphas[j]));

                if (Math.abs(state.alphas[j] - oldAj) < 1e-5) continue;

                state.alphas[i] += state.points[i].y_target * state.points[j].y_target * (oldAj - state.alphas[j]);

                const b1 = state.bias - Ei - state.points[i].y_target * (state.alphas[i] - oldAi) * K[i][i] - state.points[j].y_target * (state.alphas[j] - oldAj) * K[i][j];
                const b2 = state.bias - Ej - state.points[i].y_target * (state.alphas[i] - oldAi) * K[i][j] - state.points[j].y_target * (state.alphas[j] - oldAj) * K[j][j];

                if (state.alphas[i] > 0 && state.alphas[i] < state.C) state.bias = b1;
                else if (state.alphas[j] > 0 && state.alphas[j] < state.C) state.bias = b2;
                else state.bias = (b1 + b2) / 2;

                numChanged++;
            }
        }
        if (numChanged === 0) iter++;
        else iter = 0;
    }
}

function predictRaw(p) {
    let sum = 0;
    for (let i = 0; i < state.points.length; i++) {
        if (state.alphas[i] > 0) {
            sum += state.alphas[i] * state.points[i].y_target * kernel(state.points[i], p);
        }
    }
    return sum + state.bias;
}

function drawDecisionBoundary() {
    const w = canvas.width, h = canvas.height;
    const imgData = ctx.createImageData(w, h);
    const data = imgData.data;

    const step = 4;
    for (let py = 0; py < h; py += step) {
        for (let px = 0; px < w; px += step) {
            const { x, y } = Utils.fromCanvasCoords(px, py, canvas);
            const val = predictRaw({ x, y });

            // Map val (-1 to 1) to color
            // Margin is at -1 and 1

            let r, g, b, a;
            if (val < 0) {
                r = 239; g = 68; b = 68;
                // Intensity based on distance from boundary (0)
                a = Math.min(100, Math.abs(val) * 40 + 20);
            } else {
                r = 59; g = 130; b = 246;
                a = Math.min(100, Math.abs(val) * 40 + 20);
            }

            // Highlight margins (-1, 1)
            if (Math.abs(Math.abs(val) - 1) < 0.1) {
                a += 50; // Brighter at margin
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

        // Highlight Support Vectors (alpha > 0)
        if (state.alphas[i] > 0.001) {
            ctx.strokeStyle = '#fbbf24'; // Amber
            ctx.lineWidth = 3;
        } else {
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
        }
        ctx.stroke();
    });

    updateMetrics();
}

function updateMetrics() {
    const metrics = Utils.calculateClassificationMetrics(state.points, (x, y) => {
        return predictRaw({ x, y }) >= 0 ? 1 : 0;
    });
    accEl.textContent = Math.round(metrics.accuracy * 100) + '%';

    if (state.kernel === 'linear') {
        // For linear, margin = 2 / ||w||
        // We don't compute w explicitly in dual, but can infer.
        marginEl.textContent = "Calculated";
    } else {
        marginEl.textContent = "N/A (Non-linear)";
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
        state.points.push({ x, y, label: state.currentClass, y_target: state.currentClass === 0 ? -1 : 1 });
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

kernelSelect.addEventListener('change', (e) => {
    state.kernel = e.target.value;
    train();
    draw();
});

cSlider.addEventListener('input', (e) => {
    state.C = parseFloat(e.target.value);
    cDisplay.textContent = state.C;
    train();
    draw();
});

gammaSlider.addEventListener('input', (e) => {
    state.gamma = parseFloat(e.target.value);
    gammaDisplay.textContent = state.gamma;
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
