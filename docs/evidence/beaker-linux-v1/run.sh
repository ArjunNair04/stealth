#!/usr/bin/env bash
set -e
umask 077
export PATH="$HOME/stealth/toolchains/gcc11/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/stealth/toolchains/gcc11/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$HOME/stealth/toolchains/sanitizer-link:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$HOME/stealth/toolchains/sanitizer-root/usr/lib64:$LD_LIBRARY_PATH"
export CC="$HOME/stealth/toolchains/gcc11/bin/gcc"
export CXX="$HOME/stealth/toolchains/gcc11/bin/g++"
source /share/apps/source_files/python/python-3.11.9.source
export PATH="/share/apps/cmake-3.26.4/bin:/share/apps/ninja-1.10.0/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
cd "$HOME/stealth/runs/linux-v4"
exec "$HOME/vrl_hpc/po_venv/bin/python" -u run_validation.py
