# ML Visualizer

A comprehensive, interactive Machine Learning visualization tool built with vanilla JavaScript. This project aims to help users understand the intuition behind various ML algorithms through real-time visual feedback.

## Features

*   **10 Interactive Algorithms**:
    *   **Regression**: Linear & Polynomial Regression.
    *   **Classification**: Logistic Regression, KNN, SVM, Decision Trees, Random Forest.
    *   **Unsupervised**: K-Means Clustering, PCA.
    *   **Deep Learning**: Multi-Layer Perceptron (Neural Network).
*   **Real-time Visualization**: See decision boundaries, loss curves, and tree structures evolve as you train.
*   **Interactive Data**: Click to add points, drag to move them, and see how the model reacts instantly.
*   **Pure JavaScript**: No external ML libraries (like TensorFlow.js or Scikit-learn) used. All math and logic are implemented from scratch for educational value.

## How to Run

Since this is a static site, you can run it directly:

1.  **Open `index.html`** in any modern web browser.
2.  **Or serve locally** (recommended for best performance):
    ```bash
    # Python 3
    python -m http.server
    
    # Node.js (http-server)
    npx http-server
    ```

## Deployment

This project is ready for deployment on any static site host (GitHub Pages, Netlify, Vercel).
See [Deployment Guide](deployment_guide.md) for details.

## Technologies

*   **HTML5**: Semantic structure.
*   **CSS3**: Modern styling with Flexbox/Grid and Glassmorphism effects.
*   **JavaScript (ES6+)**: Core logic, math, and canvas rendering.
