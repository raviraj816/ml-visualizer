# Deployment Guide

Since this is a static website (HTML, CSS, and JavaScript only), you have several free and easy options to deploy it to the web.

## Option 1: Netlify Drop (Easiest & Fastest)

This method requires no command line tools and takes about 30 seconds.

1.  **Go to Netlify Drop**: Open [https://app.netlify.com/drop](https://app.netlify.com/drop) in your browser.
2.  **Prepare your folder**: Locate your project folder on your computer:
    `C:\Users\swaro\.gemini\antigravity\playground\prismic-hypernova`
3.  **Drag and Drop**: Drag the entire `prismic-hypernova` folder onto the target area on the Netlify page.
4.  **Wait for Upload**: Netlify will upload your files and give you a random URL (e.g., `silly-name-12345.netlify.app`).
5.  **Done!**: Your site is live. You can claim the site to change the name if you sign up for a free account.

## Option 2: GitHub Pages (Best for Portfolio)

If you want to host the code on GitHub and have a URL like `yourusername.github.io/project-name`.

### Prerequisites
*   You must have a GitHub account.
*   You must have Git installed on your computer.

### Steps
1.  **Create a Repository**: Go to [GitHub.com/new](https://github.com/new) and create a new repository (e.g., named `ml-visualizer`).
2.  **Initialize Git**: Open your terminal in the project folder and run:
    ```bash
    git init
    git add .
    git commit -m "Initial commit"
    ```
3.  **Push to GitHub**: Follow the instructions shown on your new GitHub repository page to push your code. It will look something like:
    ```bash
    git branch -M main
    git remote add origin https://github.com/YOUR_USERNAME/ml-visualizer.git
    git push -u origin main
    ```
4.  **Enable Pages**:
    *   Go to your repository **Settings**.
    *   Click on **Pages** in the left sidebar.
    *   Under **Source**, select `main` branch and `/ (root)` folder.
    *   Click **Save**.
5.  **Done!**: Your site will be available at `https://YOUR_USERNAME.github.io/ml-visualizer/` in a few minutes.

## Option 3: Vercel

Similar to Netlify, Vercel offers great performance for static sites.

1.  Go to [Vercel.com](https://vercel.com) and sign up.
2.  Install Vercel CLI: `npm i -g vercel` (if you have Node.js).
3.  Run `vercel` in your project folder and follow the prompts.
