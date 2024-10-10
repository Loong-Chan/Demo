#!/bin/bash
usage() {
  echo "Usage: ${0} --dataset <dataset> --device <device> [other options]" 1>&2
  exit 1 
}

DATASET=""
DEVICE=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  key=${1}
  case ${key} in
    --dataset)
      DATASET=${2}
      shift 2
      ;;
    --device)
      DEVICE=${2}
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("${1}")  # 收集任意其他参数
      shift
      ;;
  esac
done

if [[ -z "${DATASET}" || -z "${DEVICE}" ]]; then
  usage
fi

# 检查并创建 Result 文件夹
if [[ ! -d "Result" ]]; then
  mkdir -p "Result"
fi

# 将所有参数传递给 Python 脚本
nohup python Optuna.py --dataset "${DATASET}" --device "${DEVICE}" "${EXTRA_ARGS[@]}" > "Result/${DATASET}.txt" 2>&1 &
