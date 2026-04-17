/**
 * Random Forest Logic
 */

const canvas = document.getElementById('main-canvas');
const ctx = canvas.getContext('2d');

// State
const state = {
    points: [],
    currentClass: 0,
    nTrees: 5,
    maxDepth: 5,
    bootstrap: true,
    noise: 0.1,
    draggingIndex: -1,
    forest: []
};

// Elements
const treesSlider = document.getElementById('trees-slider');
const treesDisplay = document.getElementById('trees-display');
const depthSlider = document.getElementById('depth-slider');
const depthDisplay = document.getElementById('depth-display');
const bootstrapCheck = document.getElementById('bootstrap-check');
const noiseSlider = document.getElementById('noise-slider');
const noiseDisplay = document.getElementById('noise-display');
const regenBtn = document.getElementById('regen-data-btn');
const classBtns = document.querySelectorAll('.class-btn');
const accEl = document.getElementById('acc-display');
const oobEl = document.getElementById('oob-display');
const fiX = document.getElementById('fi-x');
const fiY = document.getElementById('fi-y');

function init() {
    resizeCanvas();
    generateData();
    train();
    draw();
}

function generateData() {
    state.points = Utils.generateClassificationData(50, state.noise, 0.3);
}

// Reuse Decision Tree Logic (Simplified)
class Node {
    constructor(depth) {
        this.depth = depth;
        this.feature = null;
        this.threshold = null;
        this.left = null;
        this.right = null;
        this.prediction = null;
        this.isLeaf = false;
    }
}

function buildTree(points, depth) {
    const node = new Node(depth);
    const count0 = points.filter(p => p.label === 0).length;
    const count1 = points.length - count0;
    const majority = count0 >= count1 ? 0 : 1;

    if (depth >= state.maxDepth || points.length < 2 || count0 === 0 || count1 === 0) {
        node.isLeaf = true;
        node.prediction = majority;
        return node;
    }

    let bestGini = 1.0;
    let bestSplit = null;

    // Random Feature Selection (Subset of features)
    // Since we only have X and Y, we'll just pick one randomly sometimes or check both but weigh them?
    // Standard RF checks sqrt(features). Sqrt(2) is ~1.4, so we check 1 feature?
    // Let's check both for visualization quality, but introduce randomness in split point?
    // Or strictly follow RF: randomly select k features. Here k=1.

    const allFeatures = ['x', 'y'];
    // Randomly select 1 feature to consider for this node? 
    // Or just consider both since 2 is small. Let's consider both for better viz, 
    // but rely on Bootstrap for diversity.

    for (let feature of allFeatures) {
        points.sort((a, b) => a[feature] - b[feature]);
        for (let i = 0; i < points.length - 1; i++) {
            const threshold = (points[i][feature] + points[i + 1][feature]) / 2;
            const left = points.filter(p => p[feature] <= threshold);
            const right = points.filter(p => p[feature] > threshold);
            if (left.length === 0 || right.length === 0) continue;

            const giniLeft = 1 - ((left.filter(p => p.label === 0).length / left.length) ** 2 + (left.filter(p => p.label === 1).length / left.length) ** 2);
            const giniRight = 1 - ((right.filter(p => p.label === 0).length / right.length) ** 2 + (right.filter(p => p.label === 1).length / right.length) ** 2);
            const gini = (left.length / points.length) * giniLeft + (right.length / points.length) * giniRight;

            if (gini < bestGini) {
                bestGini = gini;
                bestSplit = { feature, threshold, left, right };
            }
        }
    }

    if (!bestSplit) {
        node.isLeaf = true;
        node.prediction = majority;
        return node;
    }

    node.feature = bestSplit.feature;
    node.threshold = bestSplit.threshold;
    node.left = buildTree(bestSplit.left, depth + 1);
    node.right = buildTree(bestSplit.right, depth + 1);
    return node;
}

function predictTree(node, p) {
    if (node.isLeaf) return node.prediction;
    if (p[node.feature] <= node.threshold) return predictTree(node.left, p);
    else return predictTree(node.right, p);
}

function train() {
    if (state.points.length === 0) return;
    state.forest = [];

    for (let i = 0; i < state.nTrees; i++) {
        let sample = state.points;
        if (state.bootstrap) {
            // Bootstrap sampling with replacement
            sample = [];
            for (let j = 0; j < state.points.length; j++) {
                sample.push(state.points[Math.floor(Math.random() * state.points.length)]);
            }
        }
        state.forest.push(buildTree(sample, 0));
    }
    updateStats();
}

function predictForest(p) {
    let votes = 0;
    for (let tree of state.forest) {
        votes += predictTree(tree, p);
    }
    return votes / state.forest.length; // Returns prob of class 1
}

function drawDecisionBoundary() {
    if (state.forest.length === 0) return;
    const w = canvas.width, h = canvas.height;
    const imgData = ctx.createImageData(w, h);
    const data = imgData.data;

    const step = 4;
    for (let py = 0; py < h; py += step) {
        for (let px = 0; px < w; px += step) {
            const { x, y } = Utils.fromCanvasCoords(px, py, canvas);
            const prob = predictForest({ x, y });

            let r, g, b, a;
            if (prob < 0.5) {
                r = 239; g = 68; b = 68;
                a = 40 + (1 - prob) * 30;
            } else {
                r = 59; g = 130; b = 246;
                a = 40 + prob * 30;
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
}

function updateStats() {
    // Accuracy
    let correct = 0;
    state.points.forEach(p => {
        const prob = predictForest(p);
        const pred = prob >= 0.5 ? 1 : 0;
        if (pred === p.label) correct++;
    });
    accEl.textContent = Math.round((correct / state.points.length) * 100) + '%';

    // Feature Importance (Count splits)
    let countX = 0, countY = 0;
    function traverse(node) {
        if (!node || node.isLeaf) return;
        if (node.feature === 'x') countX++;
        else countY++;
        traverse(node.left);
        traverse(node.right);
    }
    state.forest.forEach(tree => traverse(tree));

    const total = countX + countY || 1;
    fiX.style.width = (countX / total * 100) + '%';
    fiY.style.width = (countY / total * 100) + '%';
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

treesSlider.addEventListener('input', (e) => {
    state.nTrees = parseInt(e.target.value);
    treesDisplay.textContent = state.nTrees;
    train();
    draw();
});

depthSlider.addEventListener('input', (e) => {
    state.maxDepth = parseInt(e.target.value);
    depthDisplay.textContent = state.maxDepth;
    train();
    draw();
});

bootstrapCheck.addEventListener('change', (e) => {
    state.bootstrap = e.target.checked;
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
