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

# 始终以脚本所在目录作为仓库根目录，避免 source 时受当前工作目录影响。
export PISTAR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 设置uv缓存python包的路径为本地路径，而非默认路径（/home/${user}/.cache/uv）
export UV_CACHE_DIR="$PISTAR_ROOT/.cache/uv"
# 设置huggingface lerobot数据集存储在本地路径，而非系统默认路径（/home/${user}/.cache/huggingface/lerobot）
export HF_LEROBOT_HOME="$PISTAR_ROOT/.cache/huggingface/lerobot"
# 设置openpi权重文件存储在本地路径，而非系统默认路径（/home/${user}/.cache/openpi）
export OPENPI_DATA_HOME="$PISTAR_ROOT/.cache/openpi"

# PiperX 采集不使用 ROS Python；避免 ROS 2 Humble 的 Python 3.10 包污染 Python 3.11 环境。
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PISTAR_ROOT/control_your_robot:$PISTAR_ROOT/control_your_robot/src"

if [ -d "$PISTAR_ROOT/.venv" ]; then
    source "$PISTAR_ROOT/.venv/bin/activate"
fi
