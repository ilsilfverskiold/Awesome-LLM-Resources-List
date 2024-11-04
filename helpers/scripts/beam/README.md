# Beam Deployment Guide

## Overview

Beam provides a simple and flexible platform for deploying applications with support for various model sizes and configurations. This guide will help you set up and deploy your scripts on Beam, configure endpoints, and understand basic performance expectations.

## Quick Start Guide

### 1. Set Up Environment

1. **Create Virtual Environment**:

   ```bash
   python3 -m venv .venv && source .venv/bin/activate

3. **Install Beam and Configure Token:**
   
    ```bash
    pip install beam-client
    beam configure default --token "your_token_here"

3. **Set Up Deployment Script:** Create an app.py file in the root directory. This file defines your deployment structure and endpoints.
  Refer to the example script (hf_private_models.py) in the folder and rename it app.py.

4. **Deploy Your Script** To deploy your function, use:
   
    ```bash
    beam deploy app.py:function_name

Replace function_name with the specific function in app.py that you want to deploy.

After deployment, Beam will provide you with the endpoint URL, allowing you to test or integrate your deployed function.

## Performance Insights (Averages)

### 400M Model (CPU-based) - texts batch by 30
- **Configuration**: 1 CPU core + 3GB memory
- **Cost**: ~$0.0041 per call
- **Average Times**:
  - **Cold Calls**:
    - Call #1: 21.14 seconds
    - Call #2: 14.56 seconds
  - **Warm Calls**:
    - Call #1: 7.15 seconds
    - Call #2: 3.15 seconds

### 8B Model (A100-40GB GPU)
- **Model**: Llama 3.1
- **Cost**: Based on usage
- **Average Times**:
  - **Cold Start**:
    - Initial Cold Start: 57.91 seconds
    - Second Cold Start: 2.11 minutes
  - **Warm Calls**:
    - Call #1: 7.83 seconds
    - Call #2: 6 seconds

## Tips for Optimizing Beam Deployments

- **Handle Cold Starts**: For larger models, anticipate longer initial cold start times. If frequent calls are expected, consider keeping the model active to reduce startup delays.
- **Queue Management**: Monitor queue times, especially during peak usage, as this can impact response times.
- **Optimize Resource Allocation**: Adjust resources based on model size and usage frequency to control costs and improve performance.
