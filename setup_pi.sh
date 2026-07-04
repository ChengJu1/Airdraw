#!/bin/bash
# Raspberry Pi first-time environment setup (Skywriter + web_capture)
set -e

echo "==> System packages (lgpio via apt, venv reuses system packages)..."
sudo apt update
sudo apt install -y python3-venv python3-pip python3-lgpio i2c-tools

echo "==> Virtualenv ~/sky (with system site-packages, needed to import lgpio)..."
rm -rf ~/sky
python3 -m venv --system-site-packages ~/sky
source ~/sky/bin/activate

echo "==> Python dependencies..."
pip install --upgrade pip
pip install flask smbus2 google-genai pillow numpy

mkdir -p ~/captures

echo ""
echo "==> Checking I2C (should show 42)..."
i2cdetect -y 1 || true

echo ""
echo "Done! Usage:"
echo "  source ~/sky/bin/activate"
echo "  python3 ~/web_capture.py"
echo "  browser: http://127.0.0.1:5000"
