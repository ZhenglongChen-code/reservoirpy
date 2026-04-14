#!/bin/bash
set -e

echo "=== ReservoirPy 部署脚本 ==="
echo ""
echo "用法: bash deploy.sh <secrets_file>"
echo "  secrets_file: 包含 OSS 配置的 .secrets.yaml 文件路径"
echo ""

export PATH="$HOME/.local/bin:$PATH"

cd /opt/reservoirpy

echo "[1/5] 安装依赖..."
uv sync --extra api --extra geostat

echo "[2/5] 配置密钥..."
SECRETS_FILE="${1:-.secrets.yaml}"
if [ -f "$SECRETS_FILE" ]; then
    cp "$SECRETS_FILE" .secrets.yaml
    echo "  已从 $SECRETS_FILE 复制密钥配置"
elif [ -f "$HOME/.secrets.yaml" ]; then
    cp "$HOME/.secrets.yaml" .secrets.yaml
    echo "  已从 ~/.secrets.yaml 复制密钥配置"
else
    echo "  ⚠️  未找到 .secrets.yaml，请手动创建:"
    echo "    OSS_ACCESS_KEY_ID: \"your_key_id\""
    echo "    OSS_ACCESS_KEY_SECRET: \"your_key_secret\""
    echo "    OSS_BUCKET_NAME: \"your_bucket\""
    echo "    OSS_ENDPOINT: \"oss-cn-beijing.aliyuncs.com\""
    echo "    OSS_PREFIX: \"reservoirpy/\""
fi

echo "[3/5] 创建 systemd 服务..."
cat > /etc/systemd/system/reservoirpy.service << 'SERVICE'
[Unit]
Description=ReservoirPy API Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/reservoirpy
Environment=PATH=/root/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/root/.local/bin/uv run uvicorn reservoirpy.api:app --host 0.0.0.0 --port 12375 --workers 1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE

echo "[4/5] 启动服务..."
systemctl daemon-reload
systemctl enable reservoirpy
systemctl restart reservoirpy

echo "[5/5] 等待服务启动..."
sleep 3
systemctl status reservoirpy --no-pager

echo ""
echo "=== 部署完成 ==="
echo "API 地址: http://$(hostname -I | awk '{print $1}'):12375"
echo "API 文档: http://$(hostname -I | awk '{print $1}'):12375/docs"
curl -s http://localhost:12375/ 2>/dev/null || echo "服务启动中，请稍等..."
