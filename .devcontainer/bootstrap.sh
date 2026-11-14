#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  ca-certificates curl gnupg lsb-release unzip vim make tree jq \
  python3-pip python3-venv libicu-dev qemu-user-binfmt python3.14-dev

python3 -m pip install --break-system-packages --ignore-installed -q \
  pytest==9.0.3 pre-commit==4.2.0 huggingface_hub==2.4.0

[[ ! -f .pre-commit-config.yaml ]] || pre-commit install --install-hooks

echo "==> bootstrap.sh complete."
