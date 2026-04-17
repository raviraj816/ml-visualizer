/**
 * ML Visualizer Core Logic - Extended
 */

// --- State Management ---
const state = {
    points: [], // Array of {x, y, label} (label is 0 or 1 for classification)
    draggingPointIndex: -1,
    currentAlgo: 'linear-regression',
    taskType: 'regression', // 'regression', 'classification', 'clustering'
    degree: 1,
    theme: 'dark',
    // K-Means specific
    k: 3,
    centroids: [],
    clusters: [],
    // Classification specific
    currentClass: 0, // 0 or 1
    knnK: 3,
    treeDepth: 5
};

// --- DOM Elements ---
const canvas = document.getElementById('main-canvas');
const ctx = canvas.getContext('2d');

// Stats & Controls
const performanceCard = document.getElementById('performance-card');
const classificationStats = document.getElementById('classification-stats');
const dataControls = document.getElementById('data-controls');
const polyControls = document.getElementById('poly-controls');
const kmeansControls = document.getElementById('kmeans-controls');
const knnControls = document.getElementById('knn-controls');
const treeControls = document.getElementById('tree-controls');

const mseEl = document.getElementById('mse-value');
const maeEl = document.getElementById('mae-value');
const eqEl = document.getElementById('equation-value');
const accEl = document.getElementById('accuracy-value');
const lossEl = document.getElementById('loss-value');

const degreeSlider = document.getElementById('degree-slider');
const degreeDisplay = document.getElementById('degree-display');
const kSlider = document.getElementById('k-slider');
const kDisplay = document.getElementById('k-display');
const knnKSlider = document.getElementById('knn-k-slider');
const knnKDisplay = document.getElementById('knn-k-display');
const depthSlider = document.getElementById('depth-slider');
const depthDisplay = document.getElementById('depth-display');

const kmeansStepBtn = document.getElementById('kmeans-step-btn');
const kmeansResetBtn = document.getElementById('kmeans-reset-btn');
const resetBtn = document.getElementById('reset-btn');
const navLinks = document.querySelectorAll('.nav-links li');
const algoTitle = document.getElementById('current-algo-title');
const conceptText = document.getElementById('concept-text');
const classBtns = document.querySelectorAll('.class-btn');

// --- Math Helpers ---

class Matrix {
    constructor(rows, cols, data = []) {
        this.rows = rows;
        this.cols = cols;
        this.data = data.length ? data : new Array(rows * cols).fill(0);
    }
    get(r, c) { return this.data[r * this.cols + c]; }
    set(r, c, val) { this.data[r * this.cols + c] = val; }
    multiply(other) {
        if (this.cols !== other.rows) throw new Error("Dimension mismatch");
        const result = new Matrix(this.rows, other.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < other.cols; j++) {
                let sum = 0;
                for (let k = 0; k < this.cols; k++) sum += this.get(i, k) * other.get(k, j);
                result.set(i, j, sum);
            }
        }
        return result;
    }
    transpose() {
        const result = new Matrix(this.cols, this.rows);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) result.set(j, i, this.get(i, j));
        }
        return result;
    }
    static solve(A, b) {
        const n = A.rows;
        const aug = new Matrix(n, n + 1);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) aug.set(i, j, A.get(i, j));
            aug.set(i, n, b.get(i, 0));
        }
        for (let i = 0; i < n; i++) {
            let pivot = aug.get(i, i);
            if (Math.abs(pivot) < 1e-10) {
                for (let k = i + 1; k < n; k++) {
                    if (Math.abs(aug.get(k, i)) > 1e-10) {
                        for (let j = 0; j <= n; j++) {
                            const temp = aug.get(i, j);
                            aug.set(i, j, aug.get(k, j));
                            aug.set(k, j, temp);
                        }
                        pivot = aug.get(i, i);
                        break;
                    }
                }
            }
            for (let j = i; j <= n; j++) aug.set(i, j, aug.get(i, j) / pivot);
            for (let k = 0; k < n; k++) {
                if (k !== i) {
                    const factor = aug.get(k, i);
                    for (let j = i; j <= n; j++) aug.set(k, j, aug.get(k, j) - factor * aug.get(i, j));
                }
            }
        }
        const x = new Matrix(n, 1);
        for (let i = 0; i < n; i++) x.set(i, 0, aug.get(i, n));
        return x;
    }
}

function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

// --- Algorithms ---

// 1. Regression
function solveLinearRegression(points) {
    if (points.length < 2) return { predict: (x) => 0, equation: "Need points" };
    let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0, n = points.length;
    for (let p of points) { sumX += p.x; sumY += p.y; sumXY += p.x * p.y; sumXX += p.x * p.x; }
    const m = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const b = (sumY - m * sumX) / n;
    return { predict: (x) => m * x + b, equation: `y = ${m.toFixed(2)}x + ${b.toFixed(2)}` };
}

function solvePolynomialRegression(points, degree) {
    if (points.length < degree + 1) return { predict: (x) => 0, equation: "Need points" };
    const X_data = [], Y_data = [];
    for (let p of points) {
        for (let d = 0; d <= degree; d++) X_data.push(Math.pow(p.x, d));
        Y_data.push(p.y);
    }
    const X = new Matrix(points.length, degree + 1, X_data);
    const Y = new Matrix(points.length, 1, Y_data);
    try {
        const weights = Matrix.solve(X.transpose().multiply(X), X.transpose().multiply(Y)).data;
        return {
            predict: (x) => {
                let y = 0;
                for (let i = 0; i < weights.length; i++) y += weights[i] * Math.pow(x, i);
                return y;
            },
            equation: `Poly (deg ${degree})`
        };
    } catch (e) { return { predict: (x) => 0, equation: "Error" }; }
}

// 2. Classification
function solveLogisticRegression(points) {
    // Simple Gradient Descent
    let w1 = 0, w2 = 0, b = 0;
    const lr = 0.1;
    const epochs = 1000;

    if (points.length < 2) return { predict: (x, y) => 0.5 };

    for (let i = 0; i < epochs; i++) {
        let dw1 = 0, dw2 = 0, db = 0;
        for (let p of points) {
            const z = w1 * p.x + w2 * p.y + b;
            const y_pred = sigmoid(z);
            const error = y_pred - p.label;
            dw1 += error * p.x;
            dw2 += error * p.y;
            db += error;
        }
        w1 -= lr * dw1 / points.length;
        w2 -= lr * dw2 / points.length;
        b -= lr * db / points.length;
    }

    return {
        predict: (x, y) => sigmoid(w1 * x + w2 * y + b),
        weights: { w1, w2, b }
    };
}

function solveKNN(points, k) {
    return {
        predict: (x, y) => {
            if (points.length === 0) return 0;
            const distances = points.map(p => ({
                dist: Math.hypot(p.x - x, p.y - y),
                label: p.label
            })).sort((a, b) => a.dist - b.dist).slice(0, k);

            const count0 = distances.filter(d => d.label === 0).length;
            return count0 > distances.length / 2 ? 0 : 1;
        }
    };
}

function solveSVM(points) {
    // Simplified Linear SVM (Hinge Loss with GD)
    let w1 = 0, w2 = 0, b = 0;
    const lr = 0.01;
    const lambda = 0.01;
    const epochs = 500;

    for (let i = 0; i < epochs; i++) {
        for (let p of points) {
            const y_target = p.label === 0 ? -1 : 1;
            const val = y_target * (w1 * p.x + w2 * p.y + b);

            if (val >= 1) {
                w1 -= lr * (2 * lambda * w1);
                w2 -= lr * (2 * lambda * w2);
            } else {
                w1 -= lr * (2 * lambda * w1 - y_target * p.x);
                w2 -= lr * (2 * lambda * w2 - y_target * p.y);
                b -= lr * (-y_target);
            }
        }
    }

    return {
        predict: (x, y) => (w1 * x + w2 * y + b) < 0 ? 0 : 1, // 0 is class -1, 1 is class 1
        raw: (x, y) => w1 * x + w2 * y + b
    };
}

// Simple Decision Tree
class Node {
    constructor(depth = 0) {
        this.depth = depth;
        this.left = null;
        this.right = null;
        this.splitFeature = null; // 'x' or 'y'
        this.splitValue = null;
        this.label = null;
    }
}

function buildTree(points, depth, maxDepth) {
    const node = new Node(depth);
    const labels = points.map(p => p.label);
    const count0 = labels.filter(l => l === 0).length;
    const count1 = labels.length - count0;

    if (count0 === 0 || count1 === 0 || depth >= maxDepth || points.length < 2) {
        node.label = count0 >= count1 ? 0 : 1;
        return node;
    }

    // Find best split (random search for simplicity in visualization)
    let bestGini = Infinity;
    let bestSplit = null;

    // Try 10 random splits
    for (let i = 0; i < 20; i++) {
        const feature = Math.random() > 0.5 ? 'x' : 'y';
        const val = Math.random(); // 0 to 1

        const left = points.filter(p => p[feature] < val);
        const right = points.filter(p => p[feature] >= val);

        if (left.length === 0 || right.length === 0) continue;

        const gini = (left.length * calculateGini(left) + right.length * calculateGini(right)) / points.length;
        if (gini < bestGini) {
            bestGini = gini;
            bestSplit = { feature, val, left, right };
        }
    }

    if (!bestSplit) {
        node.label = count0 >= count1 ? 0 : 1;
        return node;
    }

    node.splitFeature = bestSplit.feature;
    node.splitValue = bestSplit.val;
    node.left = buildTree(bestSplit.left, depth + 1, maxDepth);
    node.right = buildTree(bestSplit.right, depth + 1, maxDepth);
    return node;
}

function calculateGini(points) {
    if (points.length === 0) return 0;
    const p0 = points.filter(p => p.label === 0).length / points.length;
    return 1 - (p0 * p0 + (1 - p0) * (1 - p0));
}

function predictTree(node, x, y) {
    if (node.label !== null) return node.label;
    const val = node.splitFeature === 'x' ? x : y;
    if (val < node.splitValue) return predictTree(node.left, x, y);
    else return predictTree(node.right, x, y);
}

// 3. Clustering
function initKMeans() {
    state.centroids = [];
    for (let i = 0; i < state.k; i++) {
        state.centroids.push({
            x: Math.random(),
            y: Math.random(),
            color: `hsl(${i * (360 / state.k)}, 70%, 60%)`
        });
    }
    assignClusters();
}

function assignClusters() {
    state.clusters = state.points.map(p => {
        let minD = Infinity;
        let cluster = 0;
        state.centroids.forEach((c, i) => {
            const d = Math.hypot(p.x - c.x, p.y - c.y);
            if (d < minD) { minD = d; cluster = i; }
        });
        return cluster;
    });
}

function stepKMeans() {
    if (state.points.length === 0) return;
    const newCentroids = state.centroids.map(() => ({ x: 0, y: 0, count: 0 }));
    state.points.forEach((p, i) => {
        const cluster = state.clusters[i];
        newCentroids[cluster].x += p.x;
        newCentroids[cluster].y += p.y;
        newCentroids[cluster].count++;
    });
    state.centroids.forEach((c, i) => {
        if (newCentroids[i].count > 0) {
            c.x = newCentroids[i].x / newCentroids[i].count;
            c.y = newCentroids[i].y / newCentroids[i].count;
        } else { c.x = Math.random(); c.y = Math.random(); }
    });
    assignClusters();
    draw();
}

// --- Drawing ---

function toCanvasCoords(x, y) {
    const w = canvas.width, h = canvas.height, p = 40;
    return { cx: p + x * (w - p * 2), cy: h - p - y * (h - p * 2) };
}

function fromCanvasCoords(cx, cy) {
    const w = canvas.width, h = canvas.height, p = 40;
    let x = (cx - p) / (w - p * 2), y = (h - p - cy) / (h - p * 2);
    return { x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)) };
}

function drawGrid() {
    const w = canvas.width, h = canvas.height, p = 40;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    ctx.font = '10px Outfit';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.textAlign = 'center';
    for (let i = 0; i <= 10; i++) {
        const x = p + (w - p * 2) * (i / 10);
        ctx.beginPath(); ctx.moveTo(x, p); ctx.lineTo(x, h - p); ctx.stroke();
        ctx.fillText((i / 10).toFixed(1), x, h - p + 15);
    }
    ctx.textAlign = 'right';
    for (let i = 0; i <= 10; i++) {
        const y = h - p - (h - p * 2) * (i / 10);
        ctx.beginPath(); ctx.moveTo(p, y); ctx.lineTo(w - p, y); ctx.stroke();
        ctx.fillText((i / 10).toFixed(1), p - 10, y + 3);
    }
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(p, p); ctx.lineTo(p, h - p); ctx.lineTo(w - p, h - p); ctx.stroke();
}

function drawDecisionBoundary(predictFn) {
    const w = canvas.width, h = canvas.height, p = 40;
    const resolution = 40; // Lower for speed
    const plotW = w - p * 2, plotH = h - p * 2;

    const imgData = ctx.createImageData(w, h);
    const data = imgData.data;

    for (let py = p; py < h - p; py += 2) {
        for (let px = p; px < w - p; px += 2) {
            const { x, y } = fromCanvasCoords(px, py);
            const val = predictFn(x, y);

            // Color based on val (0 to 1)
            // Class 0 (Red): 239, 68, 68
            // Class 1 (Blue): 59, 130, 246

            let r, g, b, a;
            if (val < 0.5) {
                // Reddish
                const intensity = 1 - val * 2;
                r = 239; g = 68; b = 68; a = 50 * intensity;
            } else {
                // Blueish
                const intensity = (val - 0.5) * 2;
                r = 59; g = 130; b = 246; a = 50 * intensity;
            }

            // Set pixel (2x2 block for speed)
            const idx = (py * w + px) * 4;
            data[idx] = r; data[idx + 1] = g; data[idx + 2] = b; data[idx + 3] = a;
            data[idx + 4] = r; data[idx + 5] = g; data[idx + 6] = b; data[idx + 7] = a;
            const idx2 = ((py + 1) * w + px) * 4;
            data[idx2] = r; data[idx2 + 1] = g; data[idx2 + 2] = b; data[idx2 + 3] = a;
            data[idx2 + 4] = r; data[idx2 + 5] = g; data[idx2 + 6] = b; data[idx2 + 7] = a;
        }
    }
    ctx.putImageData(imgData, 0, 0);
}

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawGrid();

    if (state.taskType === 'regression') {
        let model;
        if (state.currentAlgo === 'linear-regression') model = solveLinearRegression(state.points);
        else model = solvePolynomialRegression(state.points, state.degree);

        if (state.points.length > 0) {
            ctx.beginPath();
            ctx.strokeStyle = '#6366f1';
            ctx.lineWidth = 3;
            const steps = 100;
            for (let i = 0; i <= steps; i++) {
                const x = i / steps;
                const y = model.predict(x);
                const pos = toCanvasCoords(x, y);
                if (i === 0) ctx.moveTo(pos.cx, pos.cy);
                else ctx.lineTo(pos.cx, pos.cy);
            }
            ctx.stroke();
        }
        // Draw points
        state.points.forEach((p, i) => {
            const pos = toCanvasCoords(p.x, p.y);
            ctx.beginPath(); ctx.arc(pos.cx, pos.cy, 8, 0, Math.PI * 2);
            ctx.fillStyle = i === state.draggingPointIndex ? '#fff' : '#a855f7';
            ctx.fill(); ctx.stroke();
        });
        const stats = calculateStats(state.points, model.predict);
        mseEl.textContent = stats.mse.toFixed(4);
        maeEl.textContent = stats.mae.toFixed(4);
        eqEl.textContent = model.equation;

    } else if (state.taskType === 'classification') {
        let predictFn;
        if (state.currentAlgo === 'logistic-regression') {
            const model = solveLogisticRegression(state.points);
            predictFn = model.predict;
        } else if (state.currentAlgo === 'knn') {
            const model = solveKNN(state.points, state.knnK);
            predictFn = model.predict;
        } else if (state.currentAlgo === 'svm') {
            const model = solveSVM(state.points);
            predictFn = model.predict;
        } else if (state.currentAlgo === 'decision-tree' || state.currentAlgo === 'random-forest' || state.currentAlgo === 'gradient-boosting') {
            // For RF and Boosting, we simulate with a deeper tree or ensemble for now
            const depth = state.currentAlgo === 'decision-tree' ? state.treeDepth : 8;
            const root = buildTree(state.points, 0, depth);
            predictFn = (x, y) => predictTree(root, x, y);
        }

        if (predictFn && state.points.length > 1) {
            drawDecisionBoundary(predictFn);
        }

        // Draw points
        state.points.forEach((p, i) => {
            const pos = toCanvasCoords(p.x, p.y);
            ctx.beginPath(); ctx.arc(pos.cx, pos.cy, 8, 0, Math.PI * 2);
            ctx.fillStyle = p.label === 0 ? '#ef4444' : '#3b82f6';
            ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
        });

        // Calculate Accuracy
        if (predictFn && state.points.length > 0) {
            let correct = 0;
            state.points.forEach(p => {
                const pred = predictFn(p.x, p.y) >= 0.5 ? 1 : 0;
                if (pred === p.label) correct++;
            });
            accEl.textContent = Math.round((correct / state.points.length) * 100) + '%';
        }

    } else if (state.taskType === 'clustering') {
        if (state.currentAlgo === 'k-means') {
            state.centroids.forEach(c => {
                const pos = toCanvasCoords(c.x, c.y);
                ctx.beginPath(); ctx.fillStyle = c.color; ctx.arc(pos.cx, pos.cy, 12, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
            });
            state.points.forEach((p, i) => {
                const pos = toCanvasCoords(p.x, p.y);
                ctx.beginPath(); ctx.arc(pos.cx, pos.cy, 8, 0, Math.PI * 2);
                const cluster = state.clusters[i];
                ctx.fillStyle = cluster !== undefined ? state.centroids[cluster].color : '#fff';
                ctx.fill(); ctx.stroke();
            });
        }
    }
}

function calculateStats(points, predictFn) {
    if (points.length === 0) return { mse: 0, mae: 0 };
    let sumSqErr = 0, sumAbsErr = 0;
    for (let p of points) {
        const yPred = predictFn(p.x);
        const error = p.y - yPred;
        sumSqErr += error * error;
        sumAbsErr += Math.abs(error);
    }
    return { mse: sumSqErr / points.length, mae: sumAbsErr / points.length };
}

// --- Event Listeners ---

canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const { x, y } = fromCanvasCoords(cx, cy);

    // Check click on existing
    let found = -1;
    for (let i = 0; i < state.points.length; i++) {
        const pos = toCanvasCoords(state.points[i].x, state.points[i].y);
        if (Math.hypot(pos.cx - cx, pos.cy - cy) < 15) { found = i; break; }
    }

    if (found !== -1) {
        state.draggingPointIndex = found;
    } else {
        // Add point
        const label = state.taskType === 'classification' ? state.currentClass : 0;
        state.points.push({ x, y, label });
        state.draggingPointIndex = state.points.length - 1;
        if (state.currentAlgo === 'k-means') assignClusters();
        draw();
    }
});

window.addEventListener('mousemove', (e) => {
    if (state.draggingPointIndex !== -1) {
        const rect = canvas.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        const { x, y } = fromCanvasCoords(cx, cy);
        state.points[state.draggingPointIndex].x = x;
        state.points[state.draggingPointIndex].y = y;
        if (state.currentAlgo === 'k-means') assignClusters();
        draw();
    }
});

window.addEventListener('mouseup', () => { state.draggingPointIndex = -1; });

resetBtn.addEventListener('click', () => {
    state.points = [];
    state.clusters = [];
    if (state.currentAlgo === 'k-means') initKMeans();
    draw();
});

// Controls
degreeSlider.addEventListener('input', (e) => { state.degree = parseInt(e.target.value); degreeDisplay.textContent = state.degree; draw(); });
kSlider.addEventListener('input', (e) => { state.k = parseInt(e.target.value); kDisplay.textContent = state.k; initKMeans(); draw(); });
knnKSlider.addEventListener('input', (e) => { state.knnK = parseInt(e.target.value); knnKDisplay.textContent = state.knnK; draw(); });
depthSlider.addEventListener('input', (e) => { state.treeDepth = parseInt(e.target.value); depthDisplay.textContent = state.treeDepth; draw(); });

kmeansStepBtn.addEventListener('click', stepKMeans);
kmeansResetBtn.addEventListener('click', () => { initKMeans(); draw(); });

classBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        classBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.currentClass = parseInt(btn.dataset.class);
    });
});

// Navigation
navLinks.forEach(link => {
    link.addEventListener('click', () => {
        if (link.classList.contains('disabled')) return;
        navLinks.forEach(l => l.classList.remove('active'));
        link.classList.add('active');
        const algo = link.dataset.algo;
        state.currentAlgo = algo;

        // Reset UI
        [polyControls, kmeansControls, knnControls, treeControls, performanceCard, classificationStats, dataControls].forEach(el => el.style.display = 'none');

        // Determine Task Type
        if (['linear-regression', 'polynomial-regression'].includes(algo)) {
            state.taskType = 'regression';
            performanceCard.style.display = 'block';
            if (algo === 'polynomial-regression') polyControls.style.display = 'block';
        } else if (['k-means', 'hierarchical', 'pca', 'isolation-forest'].includes(algo)) {
            state.taskType = 'clustering';
            if (algo === 'k-means') {
                kmeansControls.style.display = 'block';
                if (state.centroids.length === 0) initKMeans();
            }
        } else {
            state.taskType = 'classification';
            classificationStats.style.display = 'block';
            dataControls.style.display = 'block';
            if (algo === 'knn') knnControls.style.display = 'block';
            if (algo === 'decision-tree' || algo === 'random-forest') treeControls.style.display = 'block';
        }

        // Update Title & Text
        algoTitle.textContent = link.textContent.trim();
        draw();
    });
});

window.addEventListener('resize', () => {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
    draw();
});

// Init
canvas.width = canvas.parentElement.clientWidth;
canvas.height = canvas.parentElement.clientHeight;
state.points = [{ x: 0.2, y: 0.2, label: 0 }, { x: 0.4, y: 0.5, label: 0 }, { x: 0.6, y: 0.4, label: 1 }, { x: 0.8, y: 0.8, label: 1 }];
draw();
