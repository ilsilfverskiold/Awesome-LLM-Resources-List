# Modal.com Deployment Guide

## Overview

**Free Tier**: $30 per month

Modal.com provides an easy-to-use platform for deploying applications directly from your terminal without requiring a dedicated GPU rental. You can configure resources (GPU, memory, CPU) and deploy with ease. While there’s a bit of a learning curve around model caching, additional guides and tutorials would help simplify the process.

### Key Benefits
- **No Overhead Costs**: No apparent charges for idle storage.
- **Caching**: Simple to cache models without secrets; caching large models (e.g., 7B-8B) requires substantial memory, like an A100 GPU, which can be costly.
- **ETL Pipelines**: Great for ETL processes on CPU, with easy deployment and debugging.

## Quick Start Guide

### Running Locally

1. Set up your account in Modal.com

2. **Set Up Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate

3. **Install Modal**:
   ```bash
   pip install modal

4. **Authenticate with Modal**:
   After installing Modal, authenticate your account by running:
   ```bash
   python3 -m modal setup

   Follow the on-screen prompts to complete the authentication process.

5. **Deploy Your Script**:
   Once authentication is set up, you're ready to deploy your script. For example, if your script is named `text.py`, deploy it with:
   ```bash
   modal deploy text.py

Replace text.py with the specific script or function you want to deploy.

**Example Configurations:** This folder includes sample configurations for deploying both small (e.g., 400M) and large (e.g., 7B) models from Hugging Face, using the vLLM library. Refer to these for guidance on optimizing deployments based on model size and resource requirements.

## Performance Insights (Averages)

### 400M Model (CPU-based)
- **Configuration**: 1 CPU core + 3GB or 5GB memory
- **Cost**: $0.002 - $0.005 per call
- **Average Times**:
  - **Cold Calls**: ~15s startup, ~15s execution
  - **Warm Calls**: ~0ms startup, ~12s execution

### 400M Model (T4 GPU-based)
- **Configuration**: 1 T4 GPU + 3GB memory + 0.5 cores
- **Cost**: $0.01 per call
- **Average Times**:
  - **Cold Calls**: ~13s startup, ~14s execution
  - **Warm Calls**: ~0ms startup, ~14s execution

### 7B Model (A100 GPU-based)
- **Using Gemma 7B with vLLM**, which requires an A100 GPU (80GB)
- **Cost**: ~$0.05 per call
- **Average Times**:
  - **Cold Calls**: ~8s startup, ~40s execution
  - **Warm Calls**: ~0ms startup, ~4s execution
