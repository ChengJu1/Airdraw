#!/bin/bash
# 树莓派首次环境安装（Skywriter + web_capture）
set -e

echo "==> 系统包 (lgpio 用 apt 装，venv 复用系统包)..."
sudo apt update
sudo apt install -y python3-venv python3-pip python3-lgpio i2c-tools

echo "==> 虚拟环境 ~/sky (含系统 site-packages，才能 import lgpio)..."
rm -rf ~/sky
python3 -m venv --system-site-packages ~/sky
source ~/sky/bin/activate

echo "==> Python 依赖..."
pip install --upgrade pip
pip install flask smbus2 google-genai pillow numpy

mkdir -p ~/captures

echo ""
echo "==> 检查 I2C (应看到 42)..."
i2cdetect -y 1 || true

echo ""
echo "完成! 用法:"
echo "  source ~/sky/bin/activate"
echo "  python3 ~/web_capture.py"
echo "  浏览器: http://127.0.0.1:5000"
