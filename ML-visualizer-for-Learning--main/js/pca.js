/**
 * PCA Logic
 */

const canvas = document.getElementById('main-canvas');
const ctx = canvas.getContext('2d');

// State
const state = {
    points: [], // {x, y, z}
    mode: '2d', // '2d' or '3d'
    n: 50,
    spread: 0.8,
    rotation: { x: 0, y: 0 },
    components: [], // eigenvectors
    variance: [], // eigenvalues
    mean: { x: 0, y: 0, z: 0 }
};

// Elements
const dimSelect = document.getElementById('dim-select');
const nSlider = document.getElementById('n-slider');
const nDisplay = document.getElementById('n-display');
const spreadSlider = document.getElementById('spread-slider');
const spreadDisplay = document.getElementById('spread-display');
const regenBtn = document.getElementById('regen-data-btn');
const varPc1 = document.getElementById('var-pc1');
const varPc2 = document.getElementById('var-pc2');
const barPc1 = document.getElementById('bar-pc1');
const barPc2 = document.getElementById('bar-pc2');

function init() {
    resizeCanvas();
    generateData();
    calculatePCA();
    draw();
}

function generateData() {
    state.points = [];
    // Generate correlated data
    // x = random
    // y = x * spread + noise
    // z = x * spread2 + noise

    for (let i = 0; i < state.n; i++) {
        const x = (Math.random() - 0.5) * 2; // -1 to 1
        const noise1 = (Math.random() - 0.5) * (1 - state.spread);
        const noise2 = (Math.random() - 0.5) * (1 - state.spread);

        const y = x * state.spread + noise1;
        let z = 0;

        if (state.mode === '3d') {
            z = x * (state.spread * 0.5) + noise2 + (Math.random() - 0.5) * 0.5;
        }

        state.points.push({ x, y, z });
    }
}

function calculatePCA() {
    // 1. Center data
    let sumX = 0, sumY = 0, sumZ = 0;
    state.points.forEach(p => { sumX += p.x; sumY += p.y; sumZ += p.z; });
    state.mean = { x: sumX / state.n, y: sumY / state.n, z: sumZ / state.n };

    const centered = state.points.map(p => ({
        x: p.x - state.mean.x,
        y: p.y - state.mean.y,
        z: p.z - state.mean.z
    }));

    // 2. Covariance Matrix
    // For 2D: 2x2 matrix
    // For 3D: 3x3 matrix

    let cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    const dim = state.mode === '2d' ? 2 : 3;

    for (let p of centered) {
        const vec = [p.x, p.y, p.z];
        for (let i = 0; i < dim; i++) {
            for (let j = 0; j < dim; j++) {
                cov[i][j] += vec[i] * vec[j];
            }
        }
    }
    for (let i = 0; i < dim; i++) for (let j = 0; j < dim; j++) cov[i][j] /= (state.n - 1);

    // 3. Eigen Decomposition (Power Iteration for dominant eigenvector)
    // Simplified: Just find PC1 and PC2 roughly or use numeric library.
    // Implementing a simple Jacobi algorithm for symmetric matrix diagonalization would be best but verbose.
    // Let's use a simplified 2D analytic solution and 3D approximation or Power Iteration.

    if (state.mode === '2d') {
        // Analytic 2x2
        const a = cov[0][0], b = cov[0][1], d = cov[1][1];
        const trace = a + d;
        const det = a * d - b * b;
        const l1 = trace / 2 + Math.sqrt((trace / 2) ** 2 - det);
        const l2 = trace / 2 - Math.sqrt((trace / 2) ** 2 - det);

        state.variance = [l1, l2];

        // Eigenvectors
        // (A - lI)v = 0 => (a-l)x + by = 0 => y = -(a-l)/b * x
        let theta1 = Math.atan2(-(a - l1), b);
        if (Math.abs(b) < 1e-6) theta1 = 0; // Axis aligned

        state.components = [
            { x: Math.cos(theta1), y: Math.sin(theta1), z: 0 },
            { x: -Math.sin(theta1), y: Math.cos(theta1), z: 0 }
        ];
    } else {
        // 3D: Use Power Iteration for PC1, then deflate for PC2
        // PC1
        let v1 = [Math.random(), Math.random(), Math.random()];
        for (let iter = 0; iter < 20; iter++) {
            // Multiply Cov * v
            const nextV = [0, 0, 0];
            for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) nextV[i] += cov[i][j] * v1[j];
            // Normalize
            const mag = Math.sqrt(nextV[0] ** 2 + nextV[1] ** 2 + nextV[2] ** 2);
            v1 = nextV.map(x => x / mag);
        }
        // Eigenvalue 1 = v1^T * Cov * v1
        let l1 = 0;
        for (let i = 0; i < 3; i++) {
            let rowSum = 0;
            for (let j = 0; j < 3; j++) rowSum += cov[i][j] * v1[j];
            l1 += v1[i] * rowSum;
        }

        // Deflate: Cov' = Cov - l1 * v1 * v1^T
        const cov2 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
        for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) cov2[i][j] = cov[i][j] - l1 * v1[i] * v1[j];

        // PC2
        let v2 = [Math.random(), Math.random(), Math.random()];
        for (let iter = 0; iter < 20; iter++) {
            const nextV = [0, 0, 0];
            for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) nextV[i] += cov2[i][j] * v2[j];
            const mag = Math.sqrt(nextV[0] ** 2 + nextV[1] ** 2 + nextV[2] ** 2);
            v2 = nextV.map(x => x / mag);
        }
        let l2 = 0;
        for (let i = 0; i < 3; i++) {
            let rowSum = 0;
            for (let j = 0; j < 3; j++) rowSum += cov2[i][j] * v2[j];
            l2 += v2[i] * rowSum;
        }

        state.variance = [l1, l2];
        state.components = [
            { x: v1[0], y: v1[1], z: v1[2] },
            { x: v2[0], y: v2[1], z: v2[2] }
        ];
    }

    updateStats();
}

function project3D(x, y, z) {
    // Simple perspective projection
    // Rotate around Y then X
    const cosY = Math.cos(state.rotation.y), sinY = Math.sin(state.rotation.y);
    const cosX = Math.cos(state.rotation.x), sinX = Math.sin(state.rotation.x);

    // Rotate Y
    let x1 = x * cosY - z * sinY;
    let z1 = x * sinY + z * cosY;

    // Rotate X
    let y1 = y * cosX - z1 * sinX;
    let z2 = y * sinX + z1 * cosX;

    // Perspective
    const fov = 300;
    const scale = fov / (fov + z2 * 200 + 400); // Zoom out a bit

    return {
        x: x1 * scale,
        y: y1 * scale,
        scale: scale
    };
}

function draw() {
    Utils.clearCanvas(ctx, canvas);
    const w = canvas.width, h = canvas.height;
    const cx = w / 2, cy = h / 2;

    // Draw Axes
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 1;
    if (state.mode === '2d') {
        Utils.drawGrid(ctx, canvas);
    } else {
        // Draw 3D axes
        const origin = project3D(0, 0, 0);
        const xAxis = project3D(1, 0, 0);
        const yAxis = project3D(0, 1, 0);
        const zAxis = project3D(0, 0, 1);

        ctx.beginPath(); ctx.moveTo(cx + origin.x * 200, cy - origin.y * 200); ctx.lineTo(cx + xAxis.x * 200, cy - xAxis.y * 200); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(cx + origin.x * 200, cy - origin.y * 200); ctx.lineTo(cx + yAxis.x * 200, cy - yAxis.y * 200); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(cx + origin.x * 200, cy - origin.y * 200); ctx.lineTo(cx + zAxis.x * 200, cy - zAxis.y * 200); ctx.stroke();
    }

    // Draw Points
    state.points.forEach(p => {
        let px, py;
        if (state.mode === '2d') {
            // Map -1..1 to canvas 0..1 roughly
            const pos = Utils.toCanvasCoords(0.5 + p.x * 0.3, 0.5 + p.y * 0.3, canvas);
            px = pos.cx; py = pos.cy;
        } else {
            const proj = project3D(p.x, p.y, p.z);
            px = cx + proj.x * 200;
            py = cy - proj.y * 200;
        }

        ctx.beginPath();
        ctx.arc(px, py, 4, 0, Math.PI * 2);
        ctx.fillStyle = '#a855f7';
        ctx.fill();
    });

    // Draw Principal Components
    const mean = state.mean;
    const pc1 = state.components[0];
    const pc2 = state.components[1];
    const l1 = Math.sqrt(state.variance[0]);
    const l2 = Math.sqrt(state.variance[1]);

    // Draw PC1
    ctx.strokeStyle = '#6366f1'; // Indigo
    ctx.lineWidth = 3;
    drawVector(mean, pc1, l1);

    // Draw PC2
    ctx.strokeStyle = '#22c55e'; // Green
    ctx.lineWidth = 3;
    drawVector(mean, pc2, l2);
}

function drawVector(origin, vec, length) {
    const w = canvas.width, h = canvas.height;
    const cx = w / 2, cy = h / 2;

    let x1, y1, x2, y2;

    if (state.mode === '2d') {
        const p1 = Utils.toCanvasCoords(0.5 + origin.x * 0.3, 0.5 + origin.y * 0.3, canvas);
        const p2 = Utils.toCanvasCoords(0.5 + (origin.x + vec.x * length) * 0.3, 0.5 + (origin.y + vec.y * length) * 0.3, canvas);
        x1 = p1.cx; y1 = p1.cy;
        x2 = p2.cx; y2 = p2.cy;
    } else {
        const p1 = project3D(origin.x, origin.y, origin.z);
        const p2 = project3D(origin.x + vec.x * length, origin.y + vec.y * length, origin.z + vec.z * length);
        x1 = cx + p1.x * 200; y1 = cy - p1.y * 200;
        x2 = cx + p2.x * 200; y2 = cy - p2.y * 200;
    }

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    // Arrowhead
    const angle = Math.atan2(y2 - y1, x2 - x1);
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - 10 * Math.cos(angle - Math.PI / 6), y2 - 10 * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(x2 - 10 * Math.cos(angle + Math.PI / 6), y2 - 10 * Math.sin(angle + Math.PI / 6));
    ctx.fill();
}

function updateStats() {
    const totalVar = state.variance.reduce((a, b) => a + b, 0);
    const v1 = (state.variance[0] / totalVar * 100) || 0;
    const v2 = (state.variance[1] / totalVar * 100) || 0;

    varPc1.textContent = v1.toFixed(1) + '%';
    varPc2.textContent = v2.toFixed(1) + '%';

    barPc1.style.width = v1 + '%';
    barPc2.style.width = v2 + '%';
}

function resizeCanvas() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
    draw();
}

// Interactions
let isDragging = false;
let lastX = 0, lastY = 0;

canvas.addEventListener('mousedown', (e) => {
    if (state.mode === '3d') {
        isDragging = true;
        lastX = e.clientX;
        lastY = e.clientY;
    }
});

window.addEventListener('mousemove', (e) => {
    if (isDragging && state.mode === '3d') {
        const dx = e.clientX - lastX;
        const dy = e.clientY - lastY;
        state.rotation.y += dx * 0.01;
        state.rotation.x += dy * 0.01;
        lastX = e.clientX;
        lastY = e.clientY;
        draw();
    }
});

window.addEventListener('mouseup', () => isDragging = false);
window.addEventListener('resize', resizeCanvas);

dimSelect.addEventListener('change', (e) => {
    state.mode = e.target.value;
    generateData();
    calculatePCA();
    draw();
});

nSlider.addEventListener('input', (e) => {
    state.n = parseInt(e.target.value);
    nDisplay.textContent = state.n;
    generateData();
    calculatePCA();
    draw();
});

spreadSlider.addEventListener('input', (e) => {
    state.spread = parseFloat(e.target.value);
    spreadDisplay.textContent = state.spread;
    generateData();
    calculatePCA();
    draw();
});

regenBtn.addEventListener('click', () => {
    generateData();
    calculatePCA();
    draw();
});

init();
