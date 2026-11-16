# 🚀 Running Chest X-ray Intelligence Studio Backend on Google Colab

This directory provides the complete, self-contained Google Colab notebook to run the **FastAPI Backend Server on a cloud GPU (NVIDIA T4 / V100 / A100)** with an instant, secure public HTTPS URL via Cloudflare Tunnel.

---

## ⚡ Quickstart: 1-Click Open in Google Colab

Click the badge below to open the notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VEDANTH0517/chest-xray-intelligence-studio/blob/main/colab/Run_Backend_Google_Colab.ipynb)

---

## 📋 Step-by-Step Instructions

### Step 1: Open in Colab
1. Go to [Google Colab](https://colab.research.google.com).
2. Choose **File -> Open Notebook**.
3. Select the **GitHub** tab, enter https://github.com/VEDANTH0517/chest-xray-intelligence-studio, and select colab/Run_Backend_Google_Colab.ipynb.
   *(Or click the 'Upload' tab and upload colab/Run_Backend_Google_Colab.ipynb from your computer).*

### Step 2: Enable GPU
In Colab:
- Go to **Runtime -> Change runtime type**.
- Select **T4 GPU** (or any GPU).
- Click **Save**.

### Step 3: Run the Cells in Order
1. **Cell 1**: Hardware Check (verifies CUDA and GPU VRAM).
2. **Cell 2**: Clones the latest code from GitHub.
3. **Cell 3**: Installs all required dependencies and the Cloudflare Tunnel client.
4. **Cell 4**: Verifies model checkpoints.
5. **Cell 5**: Launches the FastAPI server on the GPU and outputs your **live public HTTPS URL**:
   `
   ======================================================================
   🚀 CHEST X-RAY INTELLIGENCE STUDIO IS LIVE ON GPU!
   ======================================================================
   👉 Studio Dashboard URL : https://xxxxxxxx.trycloudflare.com
   📖 Swagger API Docs    : https://xxxxxxxx.trycloudflare.com/docs
   🩺 Health Status API    : https://xxxxxxxx.trycloudflare.com/api/health
   ======================================================================
   `
6. **Click the Studio Dashboard URL** to access the application from any computer, phone, or tablet!
