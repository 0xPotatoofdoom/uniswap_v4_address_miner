a quick a dirty python script to mine addressess/salts for uniswap v4 deployment address.

https://blog.uniswap.org/uniswap-v4-address-mining-challenge

The Uniswap v4 CREATE2 address mining challenge runs from Nov 10-Dec 1, 2024. This guide will help you set up GPU-based mining using DigitalOcean's GPU Droplets.
Hardware Requirements & Costs:

DigitalOcean GPU Droplet: NVIDIA H100x8 ($23.92/hour)
Specifications:

8x GPUs
640GB VRAM
160 vCPUs
1920GB RAM
2TB NVMe Boot Disk
40TB NVMe Scratch Disk

Setup Steps:

Create a GPU Droplet

Select "GPU Droplets" from the left menu
Choose New York (NYC2) datacenter
Select the "AI/ML Ready" image with GPU drivers
Choose Ubuntu 24.10 x64 as the base OS
Enable SSH key authentication (required)
Name your droplet (suggestion: ml-ai-ubuntu-gpu-h100x8-640gb-nyc2)

System Setup

SSH into your new droplet
Install required packages (Python, CUDA toolkit, etc.)
Set up a Python virtual environment
Install required Python packages: numpy, cupy, numba, eth_utils

Running the Miner

Upload the mining script
Run the script using Python
Monitor output for successful finds
Results will be saved to good_salts.txt

Important Notes:

Costs can add up quickly at $23.92/hour
Remember to DELETE the droplet when not actively mining - those things are expensive fr.
Consider setting up monitoring/alerts for your DigitalOcean spending
Make sure to back up any found solutions immediately
The script automatically saves good results to good_salts.txt
