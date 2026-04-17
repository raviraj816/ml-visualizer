/**
 * Decision Tree Logic
 */

const canvas = document.getElementById('main-canvas');
const ctx = canvas.getContext('2d');

// State
const state = {
    points: [],
    currentClass: 0,
    maxDepth: 5,
    minSamplesSplit: 2,
    noise: 0.1,
    draggingIndex: -1,
    tree: null
};

// Elements
const depthSlider = document.getElementById('depth-slider');
const depthDisplay = document.getElementById('depth-display');
const splitSlider = document.getElementById('split-slider');
const splitDisplay = document.getElementById('split-display');
const noiseSlider = document.getElementById('noise-slider');
const noiseDisplay = document.getElementById('noise-display');
const regenBtn = document.getElementById('regen-data-btn');
const classBtns = document.querySelectorAll('.class-btn');
const accEl = document.getElementById('acc-display');
const depthStatEl = document.getElementById('depth-stat');
const nodesStatEl = document.getElementById('nodes-stat');
const treeVizEl = document.getElementById('tree-viz');

function init() {
    resizeCanvas();
    generateData();
    train();
    draw();
}

function generateData() {
    state.points = Utils.generateClassificationData(40, state.noise, 0.3);
}

// Decision Tree Implementation
class Node {
    constructor(depth) {
        this.depth = depth;
        this.feature = null; // 'x' or 'y'
        this.threshold = null;
        this.left = null;
        this.right = null;
        this.prediction = null; // 0 or 1 (leaf)
        this.isLeaf = false;
    }
}

function buildTree(points, depth) {
    const node = new Node(depth);

    const count0 = points.filter(p => p.label === 0).length;
    const count1 = points.length - count0;
    const majority = count0 >= count1 ? 0 : 1;

    // Stopping criteria
    if (depth >= state.maxDepth || points.length < state.minSamplesSplit || count0 === 0 || count1 === 0) {
        node.isLeaf = true;
        node.prediction = majority;
        return node;
    }

    // Find best split (Gini Impurity)
    let bestGini = 1.0;
    let bestSplit = null;

    const features = ['x', 'y'];
    for (let feature of features) {
        // Sort points by feature
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

function predict(node, p) {
    if (node.isLeaf) return node.prediction;
    if (p[node.feature] <= node.threshold) return predict(node.left, p);
    else return predict(node.right, p);
}

function train() {
    if (state.points.length === 0) return;
    state.tree = buildTree([...state.points], 0);
    updateStats();
    renderTreeViz();
}

function drawDecisionBoundary() {
    if (!state.tree) return;
    const w = canvas.width, h = canvas.height;
    const imgData = ctx.createImageData(w, h);
    const data = imgData.data;

    const step = 4;
    for (let py = 0; py < h; py += step) {
        for (let px = 0; px < w; px += step) {
            const { x, y } = Utils.fromCanvasCoords(px, py, canvas);
            const pred = predict(state.tree, { x, y });

            let r, g, b, a;
            if (pred === 0) {
                r = 239; g = 68; b = 68; a = 60;
            } else {
                r = 59; g = 130; b = 246; a = 60;
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
    if (!state.tree) return;

    // Accuracy
    let correct = 0;
    state.points.forEach(p => {
        if (predict(state.tree, p) === p.label) correct++;
    });
    accEl.textContent = Math.round((correct / state.points.length) * 100) + '%';

    // Tree Stats
    let maxDepth = 0;
    let nodes = 0;
    function traverse(node) {
        if (!node) return;
        nodes++;
        if (node.depth > maxDepth) maxDepth = node.depth;
        traverse(node.left);
        traverse(node.right);
    }
    traverse(state.tree);
    depthStatEl.textContent = maxDepth;
    nodesStatEl.textContent = nodes;
}

function renderTreeViz() {
    if (!state.tree) return;
    let output = "";

    function printNode(node, prefix, isLeft) {
        if (!node) return;

        output += prefix;
        output += (isLeft ? "├── " : "└── ");

        if (node.isLeaf) {
            output += `Leaf: ${node.prediction === 0 ? "Red" : "Blue"}\n`;
        } else {
            output += `${node.feature.toUpperCase()} <= ${node.threshold.toFixed(2)}\n`;
            printNode(node.left, prefix + (isLeft ? "│   " : "    "), true);
            printNode(node.right, prefix + (isLeft ? "│   " : "    "), false);
        }
    }

    // Root
    if (state.tree.isLeaf) {
        output += `Leaf: ${state.tree.prediction === 0 ? "Red" : "Blue"}\n`;
    } else {
        output += `${state.tree.feature.toUpperCase()} <= ${state.tree.threshold.toFixed(2)}\n`;
        printNode(state.tree.left, "", true);
        printNode(state.tree.right, "", false);
    }

    treeVizEl.textContent = output;
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

depthSlider.addEventListener('input', (e) => {
    state.maxDepth = parseInt(e.target.value);
    depthDisplay.textContent = state.maxDepth;
    train();
    draw();
});

splitSlider.addEventListener('input', (e) => {
    state.minSamplesSplit = parseInt(e.target.value);
    splitDisplay.textContent = state.minSamplesSplit;
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
