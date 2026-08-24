# export UV_INDEX_URL=https://pypi.org/simple/
# 设置uv走国内源
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/
# export UV_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple/
# export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
# 设置huggingface走国内镜像站
export HF_ENDPOINT=https://hf-mirror.com
# 设置环境变量跳过 LFS smudge（避免lerobot下载大文件）
export GIT_LFS_SKIP_SMUDGE=1

# 设置 WandB API Key
# export WANDB_API_KEY=<your_wandb_api_key>

# 设置uv缓存python包的路径为本地路径，而非默认路径（/home/${user}/.cache/uv）
export UV_CACHE_DIR=$(pwd)/.cache/uv
# 设置huggingface lerobot数据集存储在本地路径，而非系统默认路径（/home/${user}/.cache/huggingface/lerobot）
export HF_LEROBOT_HOME=$(pwd)/.cache/huggingface/lerobot
# 设置openpi权重文件存储在本地路径，而非系统默认路径（/home/${user}/.cache/openpi）
export OPENPI_DATA_HOME=$(pwd)/.cache/openpi

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi
