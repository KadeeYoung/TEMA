#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export ENABLE_AUDIO_OUTPUT=0
export SWIFT_AUDIO_LOAD_BACKEND=soundfile_pyav
export OMP_NUM_THREADS=${TEMA_OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${TEMA_MKL_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${TEMA_OPENBLAS_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
exec "${PYTHON:-python}" -m tema_chat.infer "$@"
