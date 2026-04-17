/**
 * Neural Network Logic (Simple MLP from scratch)
 */

const canvas = document.getElementById('main-canvas');
const ctx = canvas.getContext('2d');
const lossCanvas = document.getElementById('loss-chart');
const lossCtx = lossCanvas.getContext('2d');

// State
const state = {
    points: [],
    shape: 'circles',
    hiddenLayers: 1,
    neuronsPerLayer: 4,
    learningRate: 0.03,
    isTraining: false,
    epoch: 0,
    lossHistory: [],
    network: null,
    animationId: null
};

// Elements
const layersSlider = document.getElementById('layers-slider');
const layersDisplay = document.getElementById('layers-display');
const neuronsSlider = document.getElementById('neurons-slider');
const neuronsDisplay = document.getElementById('neurons-display');
const lrSlider = document.getElementById('lr-slider');
const lrDisplay = document.getElementById('lr-display');
const shapeSelect = document.getElementById('shape-select');
const trainBtn = document.getElementById('train-btn');
const regenBtn = document.getElementById('regen-data-btn');
const epochDisplay = document.getElementById('epoch-display');
const lossDisplay = document.getElementById('loss-display');

// --- Neural Net Implementation ---

class Layer {
    constructor(inSize, outSize) {
        this.weights = [];
        this.biases = [];
        // Xavier Initialization
        const scale = Math.sqrt(2 / (inSize + outSize));
        for (let i = 0; i < outSize; i++) {
            const wRow = [];
            for (let j = 0; j < inSize; j++) {
                wRow.push((Math.random() - 0.5) * 2 * scale);
            }
            this.weights.push(wRow);
            this.biases.push(0);
        }
        this.inputs = null;
        this.z = null;
        this.a = null;
    }

    forward(inputs) {
        this.inputs = inputs; // Store for backprop
        this.z = [];
        this.a = [];
        for (let i = 0; i < this.weights.length; i++) {
            let sum = this.biases[i];
            for (let j = 0; j < this.weights[i].length; j++) {
                sum += this.weights[i][j] * inputs[j];
            }
            this.z.push(sum);
            this.a.push(this.activation(sum));
        }
        return this.a;
    }

    activation(z) {
        // Tanh for hidden, Sigmoid for output? Or ReLU?
        // Let's use Tanh for hidden and Sigmoid for output
        return Math.tanh(z);
    }

    activationPrime(z) {
        return 1 - Math.tanh(z) ** 2;
    }
}

class Network {
    constructor(layerSizes) {
        this.layers = [];
        for (let i = 0; i < layerSizes.length - 1; i++) {
            this.layers.push(new Layer(layerSizes[i], layerSizes[i + 1]));
        }
    }

    forward(inputs) {
        let current = inputs;
        for (let layer of this.layers) {
            current = layer.forward(current);
        }
        return current; // Output layer activation
    }

    train(data, lr) {
        let totalLoss = 0;

        for (let { x, y, label } of data) {
            // Forward
            const inputs = [x, y];
            const output = this.forward(inputs)[0]; // Single output neuron

            // Loss (MSE)
            const error = output - label;
            totalLoss += error * error;

            // Backward
            // Output Layer Gradient
            // dLoss/dOut = 2 * error
            // dOut/dZ = activationPrime(Z)
            // dZ/dW = Input

            let grad = 2 * error; // Initial gradient from loss

            for (let i = this.layers.length - 1; i >= 0; i--) {
                const layer = this.layers[i];
                const nextGrad = new Array(layer.inputs.length).fill(0);

                for (let j = 0; j < layer.weights.length; j++) {
                    // Local gradient for neuron j
                    // If output layer, activation is Tanh (simplified, should be Sigmoid for 0-1 but Tanh is -1 to 1. Let's assume labels are -1, 1 or map output)
                    // Let's stick to Tanh everywhere and labels -1, 1 for simplicity?
                    // Or use Sigmoid for last layer.
                    // To keep it simple, let's use Tanh everywhere and map labels 0->-1, 1->1

                    const dAct = layer.activationPrime(layer.z[j]);
                    const localGrad = grad * dAct; // If multiple outputs, grad is array. Here 1 output for last layer.

                    // Update weights and biases
                    for (let k = 0; k < layer.weights[j].length; k++) {
                        // Accumulate gradient for previous layer
                        nextGrad[k] += localGrad * layer.weights[j][k];

                        // Update weight
                        layer.weights[j][k] -= lr * localGrad * layer.inputs[k];
                    }
                    layer.biases[j] -= lr * localGrad;
                }

                // Pass gradient to previous layer
                // For hidden layers, grad is array
                // Wait, my simple backprop loop above assumes 1 output neuron for backprop start.
                // For hidden layers, 'grad' needs to be the 'nextGrad' calculated.
                // But 'grad' is scalar for last layer. For hidden it is vector.
                // Let's fix:
                // This simple implementation is tricky with mixed scalar/vector.
                // Let's just do it properly.
            }
        }
        return totalLoss / data.length;
    }

    // Better Backprop
    trainStep(data, lr) {
        let totalLoss = 0;

        for (let { x, y, label } of data) {
            // Target: 0 -> -0.8, 1 -> 0.8 (Keep within Tanh range)
            const target = label === 0 ? -0.8 : 0.8;

            // Forward
            const inputs = [x, y];
            let acts = [inputs];
            let zs = [];

            let curr = inputs;
            for (let layer of this.layers) {
                curr = layer.forward(curr);
                zs.push(layer.z);
                acts.push(curr);
            }

            const output = curr[0];
            const error = output - target;
            totalLoss += error * error;

            // Backward
            let delta = [2 * error * (1 - output * output)]; // Tanh prime: 1 - out^2

            for (let i = this.layers.length - 1; i >= 0; i--) {
                const layer = this.layers[i];
                const input = acts[i];
                const nextDelta = new Array(input.length).fill(0);

                for (let j = 0; j < layer.weights.length; j++) {
                    const d = delta[j];
                    for (let k = 0; k < layer.weights[j].length; k++) {
                        nextDelta[k] += d * layer.weights[j][k];
                        layer.weights[j][k] -= lr * d * input[k];
                    }
                    layer.biases[j] -= lr * d;
                }

                // Prepare delta for next iteration (previous layer)
                if (i > 0) {
                    // Multiply by activation prime of previous layer
                    for (let k = 0; k < nextDelta.length; k++) {
                        nextDelta[k] *= (1 - input[k] * input[k]); // Tanh prime
                    }
                    delta = nextDelta;
                }
            }
        }
        return totalLoss / data.length;
    }
}

// --- App Logic ---

function init() {
    resizeCanvas();
    generateData();
    initNetwork();
    draw();
}

function generateData() {
    state.points = [];
    const n = 60;

    if (state.shape === 'circles') {
        for (let i = 0; i < n; i++) {
            const angle = Math.random() * Math.PI * 2;
            const r = i < n / 2 ? 0.3 : 0.6;
            state.points.push({
                x: 0.5 + Math.cos(angle) * r * 0.5 + (Math.random() - 0.5) * 0.1,
                y: 0.5 + Math.sin(angle) * r * 0.5 + (Math.random() - 0.5) * 0.1,
                label: i < n / 2 ? 0 : 1
            });
        }
    } else if (state.shape === 'xor') {
        for (let i = 0; i < n; i++) {
            const x = Math.random();
            const y = Math.random();
            const label = (x > 0.5 && y > 0.5) || (x < 0.5 && y < 0.5) ? 0 : 1;
            state.points.push({ x, y, label });
        }
    } else if (state.shape === 'moons') {
        // Simplified moons
        for (let i = 0; i < n; i++) {
            let x, y, label;
            if (i < n / 2) {
                const a = Math.PI * (i / (n / 2));
                x = 0.3 + 0.3 * Math.cos(a);
                y = 0.3 + 0.3 * Math.sin(a);
                label = 0;
            } else {
                const a = Math.PI * ((i - n / 2) / (n / 2));
                x = 0.7 - 0.3 * Math.cos(a);
                y = 0.7 - 0.3 * Math.sin(a);
                label = 1;
            }
            state.points.push({
                x: x + (Math.random() - 0.5) * 0.1,
                y: y + (Math.random() - 0.5) * 0.1,
                label
            });
        }
    } else {
        // Spiral
        for (let i = 0; i < n; i++) {
            const r = i / n * 0.4;
            const t = 1.75 * i / n * 2 * Math.PI;
            const x = 0.5 + r * Math.sin(t);
            const y = 0.5 + r * Math.cos(t);
            const label = i % 2; // Not a real spiral class, just spiral points. 
            // Let's do 2 arms
            const arm = i % 2;
            const t2 = t + arm * Math.PI;
            state.points.push({
                x: 0.5 + r * Math.sin(t2) + (Math.random() - 0.5) * 0.05,
                y: 0.5 + r * Math.cos(t2) + (Math.random() - 0.5) * 0.05,
                label: arm
            });
        }
    }
}

function initNetwork() {
    const layers = [2]; // Input x,y
    for (let i = 0; i < state.hiddenLayers; i++) layers.push(state.neuronsPerLayer);
    layers.push(1); // Output

    state.network = new Network(layers);
    state.epoch = 0;
    state.lossHistory = [];
    state.isTraining = false;
    trainBtn.textContent = "Start Training";
}

function trainLoop() {
    if (!state.isTraining) return;

    const stepsPerFrame = 10;
    let loss = 0;
    for (let i = 0; i < stepsPerFrame; i++) {
        loss = state.network.trainStep(state.points, state.learningRate);
        state.epoch++;
    }

    state.lossHistory.push(loss);
    if (state.lossHistory.length > 100) state.lossHistory.shift();

    epochDisplay.textContent = state.epoch;
    lossDisplay.textContent = loss.toFixed(4);

    draw();
    drawLossChart();

    state.animationId = requestAnimationFrame(trainLoop);
}

function drawDecisionBoundary() {
    const w = canvas.width, h = canvas.height;
    const imgData = ctx.createImageData(w, h);
    const data = imgData.data;

    const step = 5; // Lower res for speed
    for (let py = 0; py < h; py += step) {
        for (let px = 0; px < w; px += step) {
            const { x, y } = Utils.fromCanvasCoords(px, py, canvas);
            const out = state.network.forward([x, y])[0];

            // Out is -1 to 1 roughly (Tanh)
            // Map to color
            let r, g, b, a;
            if (out < 0) {
                r = 239; g = 68; b = 68;
                a = Math.min(150, Math.abs(out) * 100 + 20);
            } else {
                r = 59; g = 130; b = 246;
                a = Math.min(150, Math.abs(out) * 100 + 20);
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
    drawDecisionBoundary();
    Utils.drawGrid(ctx, canvas);

    state.points.forEach(p => {
        const pos = Utils.toCanvasCoords(p.x, p.y, canvas);
        ctx.beginPath();
        ctx.arc(pos.cx, pos.cy, 6, 0, Math.PI * 2);
        ctx.fillStyle = p.label === 0 ? '#ef4444' : '#3b82f6';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
    });
}

function drawLossChart() {
    lossCtx.clearRect(0, 0, lossCanvas.width, lossCanvas.height);
    if (state.lossHistory.length < 2) return;

    lossCtx.beginPath();
    lossCtx.strokeStyle = '#22c55e';
    lossCtx.lineWidth = 2;

    const maxLoss = Math.max(...state.lossHistory, 0.5);
    const w = lossCanvas.width;
    const h = lossCanvas.height;

    state.lossHistory.forEach((l, i) => {
        const x = (i / (state.lossHistory.length - 1)) * w;
        const y = h - (l / maxLoss) * h;
        if (i === 0) lossCtx.moveTo(x, y);
        else lossCtx.lineTo(x, y);
    });
    lossCtx.stroke();
}

function resizeCanvas() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
    lossCanvas.width = lossCanvas.parentElement.clientWidth;
    lossCanvas.height = lossCanvas.parentElement.clientHeight;
    draw();
}

// Interactions
canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const { x, y } = Utils.fromCanvasCoords(cx, cy, canvas);
    // Add point with label based on region? Or toggle?
    // Let's just add random label or based on nearest neighbor?
    // Simple: Add Class 0
    state.points.push({ x, y, label: Math.random() > 0.5 ? 1 : 0 });
    draw();
});

window.addEventListener('resize', resizeCanvas);

layersSlider.addEventListener('input', (e) => {
    state.hiddenLayers = parseInt(e.target.value);
    layersDisplay.textContent = state.hiddenLayers;
    initNetwork();
    draw();
});

neuronsSlider.addEventListener('input', (e) => {
    state.neuronsPerLayer = parseInt(e.target.value);
    neuronsDisplay.textContent = state.neuronsPerLayer;
    initNetwork();
    draw();
});

lrSlider.addEventListener('input', (e) => {
    state.learningRate = parseFloat(e.target.value);
    lrDisplay.textContent = state.learningRate;
});

shapeSelect.addEventListener('change', (e) => {
    state.shape = e.target.value;
    generateData();
    initNetwork();
    draw();
});

regenBtn.addEventListener('click', () => {
    generateData();
    initNetwork();
    draw();
});

trainBtn.addEventListener('click', () => {
    state.isTraining = !state.isTraining;
    trainBtn.textContent = state.isTraining ? "Pause Training" : "Resume Training";
    if (state.isTraining) trainLoop();
    else cancelAnimationFrame(state.animationId);
});

init();
