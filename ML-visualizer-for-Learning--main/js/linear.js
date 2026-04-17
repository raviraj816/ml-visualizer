/**
 * Linear Regression Logic
 */

const canvas = document.getElementById('main-canvas');
const ctx = canvas.getContext('2d');

// State
const state = {
    points: [],
    degree: 1,
    lambda: 0,
    noise: 0.2,
    splitRatio: 0.8, // 80% train
    draggingIndex: -1
};

// Elements
const degreeSlider = document.getElementById('degree-slider');
const degreeDisplay = document.getElementById('degree-display');
const lambdaSlider = document.getElementById('lambda-slider');
const lambdaDisplay = document.getElementById('lambda-display');
const noiseSlider = document.getElementById('noise-slider');
const noiseDisplay = document.getElementById('noise-display');
const splitSlider = document.getElementById('split-slider');
const splitDisplay = document.getElementById('split-display');
const regenBtn = document.getElementById('regen-data-btn');
const mseTrainEl = document.getElementById('mse-train');
const mseTestEl = document.getElementById('mse-test');
const eqEl = document.getElementById('equation-value');

// Logic
function init() {
    resizeCanvas();
    generateData();
    draw();
}

function generateData() {
    state.points = Utils.generateRegressionData(20, state.noise, 'sine');
    // Shuffle for random split
    state.points.sort(() => Math.random() - 0.5);
}

function getSplitData() {
    const splitIdx = Math.floor(state.points.length * state.splitRatio);
    return {
        train: state.points.slice(0, splitIdx),
        test: state.points.slice(splitIdx)
    };
}

function solvePolynomialRegression(points, degree, lambda) {
    if (points.length < 1) return { predict: () => 0, equation: "No data" };

    const X_data = [];
    const Y_data = [];
    for (let p of points) {
        for (let d = 0; d <= degree; d++) X_data.push(Math.pow(p.x, d));
        Y_data.push(p.y);
    }

    const X = new Utils.Matrix(points.length, degree + 1, X_data);
    const Y = new Utils.Matrix(points.length, 1, Y_data);
    const XT = X.transpose();
    let XTX = XT.multiply(X);

    // Add L2 Regularization (Ridge): (X^T X + lambda * I)^-1
    if (lambda > 0) {
        for (let i = 0; i < XTX.rows; i++) {
            // Don't regularize bias term (usually index 0)
            if (i > 0) {
                XTX.set(i, i, XTX.get(i, i) + lambda * points.length); // Scale lambda by N
            }
        }
    }

    const XTY = XT.multiply(Y);

    try {
        const W = Utils.Matrix.solve(XTX, XTY);
        const weights = W.data;
        return {
            predict: (x) => {
                let y = 0;
                for (let i = 0; i < weights.length; i++) y += weights[i] * Math.pow(x, i);
                return y;
            },
            equation: degree === 1 ? `y = ${weights[1].toFixed(2)}x + ${weights[0].toFixed(2)}` : `Poly (deg ${degree})`
        };
    } catch (e) {
        return { predict: () => 0, equation: "Error" };
    }
}

function draw() {
    Utils.clearCanvas(ctx, canvas);
    Utils.drawGrid(ctx, canvas);

    const { train, test } = getSplitData();
    const model = solvePolynomialRegression(train, state.degree, state.lambda);

    // Draw Curve
    ctx.beginPath();
    ctx.strokeStyle = '#6366f1';
    ctx.lineWidth = 3;
    const steps = 100;
    for (let i = 0; i <= steps; i++) {
        const x = i / steps;
        const y = model.predict(x);
        const pos = Utils.toCanvasCoords(x, y, canvas);
        if (i === 0) ctx.moveTo(pos.cx, pos.cy);
        else ctx.lineTo(pos.cx, pos.cy);
    }
    ctx.stroke();

    // Draw Train Points
    train.forEach(p => {
        const pos = Utils.toCanvasCoords(p.x, p.y, canvas);
        ctx.beginPath();
        ctx.arc(pos.cx, pos.cy, 6, 0, Math.PI * 2);
        ctx.fillStyle = '#a855f7'; // Purple
        ctx.fill();
    });

    // Draw Test Points
    test.forEach(p => {
        const pos = Utils.toCanvasCoords(p.x, p.y, canvas);
        ctx.beginPath();
        ctx.arc(pos.cx, pos.cy, 6, 0, Math.PI * 2);
        ctx.fillStyle = '#22c55e'; // Green
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.stroke();
    });

    // Metrics
    const trainMetrics = Utils.calculateRegressionMetrics(train, model.predict);
    const testMetrics = Utils.calculateRegressionMetrics(test, model.predict);

    mseTrainEl.textContent = trainMetrics.mse.toFixed(4);
    mseTestEl.textContent = testMetrics.mse.toFixed(4);
    eqEl.textContent = model.equation;
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

    // Check click
    let found = -1;
    for (let i = 0; i < state.points.length; i++) {
        const pos = Utils.toCanvasCoords(state.points[i].x, state.points[i].y, canvas);
        if (Math.hypot(pos.cx - cx, pos.cy - cy) < 15) { found = i; break; }
    }

    if (found !== -1) {
        state.draggingIndex = found;
    } else {
        state.points.push({ x, y });
        draw();
    }
});

window.addEventListener('mousemove', (e) => {
    if (state.draggingIndex !== -1) {
        const rect = canvas.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        const { x, y } = Utils.fromCanvasCoords(cx, cy, canvas);
        state.points[state.draggingIndex] = { x, y };
        draw();
    }
});

window.addEventListener('mouseup', () => state.draggingIndex = -1);
window.addEventListener('resize', resizeCanvas);

degreeSlider.addEventListener('input', (e) => {
    state.degree = parseInt(e.target.value);
    degreeDisplay.textContent = state.degree;
    draw();
});

lambdaSlider.addEventListener('input', (e) => {
    state.lambda = parseFloat(e.target.value);
    lambdaDisplay.textContent = state.lambda.toFixed(2);
    draw();
});

noiseSlider.addEventListener('input', (e) => {
    state.noise = parseFloat(e.target.value);
    noiseDisplay.textContent = state.noise;
    generateData();
    draw();
});

splitSlider.addEventListener('input', (e) => {
    state.splitRatio = parseInt(e.target.value) / 100;
    splitDisplay.textContent = e.target.value + '%';
    draw();
});

regenBtn.addEventListener('click', () => {
    generateData();
    draw();
});

init();
