/**
 * ML Visualizer - Shared Utilities
 * Math, Matrix, and Plotting helpers.
 */

// --- Math & Matrix ---

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
                for (let k = 0; k < this.cols; k++) {
                    sum += this.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        return result;
    }

    transpose() {
        const result = new Matrix(this.cols, this.rows);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.set(j, i, this.get(i, j));
            }
        }
        return result;
    }

    static solve(A, b) {
        // Gaussian elimination
        const n = A.rows;
        const aug = new Matrix(n, n + 1);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) aug.set(i, j, A.get(i, j));
            aug.set(i, n, b.get(i, 0));
        }

        for (let i = 0; i < n; i++) {
            let pivot = aug.get(i, i);
            if (Math.abs(pivot) < 1e-10) {
                // Pivot
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
                    for (let j = i; j <= n; j++) {
                        aug.set(k, j, aug.get(k, j) - factor * aug.get(i, j));
                    }
                }
            }
        }
        const x = new Matrix(n, 1);
        for (let i = 0; i < n; i++) x.set(i, 0, aug.get(i, n));
        return x;
    }
}

const Utils = {
    Matrix,

    // --- Random Data Generation ---
    generateRegressionData: (n, noiseLevel, type = 'linear') => {
        const points = [];
        for (let i = 0; i < n; i++) {
            const x = Math.random();
            let y = 0;
            if (type === 'linear') {
                y = 0.2 + 0.6 * x;
            } else if (type === 'quadratic') {
                y = 4 * (x - 0.5) * (x - 0.5) + 0.2;
            } else if (type === 'sine') {
                y = 0.5 + 0.4 * Math.sin(x * Math.PI * 2);
            }
            y += (Math.random() - 0.5) * noiseLevel;
            y = Math.max(0, Math.min(1, y));
            points.push({ x, y });
        }
        return points;
    },

    generateClassificationData: (n, noise, separation) => {
        const points = [];
        // Class 0: Bottom Left
        for (let i = 0; i < n / 2; i++) {
            points.push({
                x: 0.25 + (Math.random() - 0.5) * (0.3 + noise),
                y: 0.25 + (Math.random() - 0.5) * (0.3 + noise),
                label: 0
            });
        }
        // Class 1: Top Right
        for (let i = 0; i < n / 2; i++) {
            points.push({
                x: 0.75 + (Math.random() - 0.5) * (0.3 + noise),
                y: 0.75 + (Math.random() - 0.5) * (0.3 + noise),
                label: 1
            });
        }
        return points;
    },

    // --- Metrics ---
    calculateRegressionMetrics: (points, predictFn) => {
        if (points.length === 0) return { mse: 0, mae: 0, rmse: 0 };
        let sumSq = 0, sumAbs = 0;
        for (let p of points) {
            const err = p.y - predictFn(p.x);
            sumSq += err * err;
            sumAbs += Math.abs(err);
        }
        const mse = sumSq / points.length;
        return {
            mse: mse,
            mae: sumAbs / points.length,
            rmse: Math.sqrt(mse)
        };
    },

    calculateClassificationMetrics: (points, predictFn) => {
        if (points.length === 0) return { accuracy: 0, precision: 0, recall: 0, confusion: [0, 0, 0, 0] };
        let tp = 0, fp = 0, tn = 0, fn = 0;
        for (let p of points) {
            const pred = predictFn(p.x, p.y) >= 0.5 ? 1 : 0;
            if (pred === 1 && p.label === 1) tp++;
            else if (pred === 1 && p.label === 0) fp++;
            else if (pred === 0 && p.label === 0) tn++;
            else if (pred === 0 && p.label === 1) fn++;
        }
        const acc = (tp + tn) / points.length;
        const prec = (tp + fp) > 0 ? tp / (tp + fp) : 0;
        const rec = (tp + fn) > 0 ? tp / (tp + fn) : 0;
        return { accuracy: acc, precision: prec, recall: rec, confusion: [tn, fp, fn, tp] };
    },

    // --- Canvas Helpers ---
    clearCanvas: (ctx, canvas) => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    },

    drawGrid: (ctx, canvas) => {
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
    },

    toCanvasCoords: (x, y, canvas) => {
        const w = canvas.width, h = canvas.height, p = 40;
        return {
            cx: p + x * (w - p * 2),
            cy: h - p - y * (h - p * 2)
        };
    },

    fromCanvasCoords: (cx, cy, canvas) => {
        const w = canvas.width, h = canvas.height, p = 40;
        let x = (cx - p) / (w - p * 2);
        let y = (h - p - cy) / (h - p * 2);
        return {
            x: Math.max(0, Math.min(1, x)),
            y: Math.max(0, Math.min(1, y))
        };
    }
};
