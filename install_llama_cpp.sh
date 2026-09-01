#!/usr/bin/env bash
# Build and install llama.cpp with CUDA for this project.
#
#   ./install_llama_cpp.sh              # build/update, install to ~/.local
#   PREFIX=/opt/llama ./install_llama_cpp.sh
#   SRC=~/src/llama.cpp ./install_llama_cpp.sh
#
# Requires llama.cpp >= b9173: earlier builds load the ASR model and encode
# audio but transcribe everything as empty output (llama.cpp issue #22357).

set -euo pipefail

SRC="${SRC:-$HOME/workspace/llama.cpp}"
PREFIX="${PREFIX:-$HOME/.local}"
REPO="${REPO:-https://github.com/ggml-org/llama.cpp}"
JOBS="${JOBS:-$(nproc)}"
MIN_BUILD=9173

command -v cmake >/dev/null || { echo "cmake not found"; exit 1; }
command -v git   >/dev/null || { echo "git not found";   exit 1; }

# ── CUDA architecture ────────────────────────────────────────────────────────
# Building for the exact compute capability of the installed GPUs keeps the
# build small and avoids shipping kernels this machine cannot run. Quadro
# RTX 8000 / RTX 2080 Ti are 7.5; Ampere is 8.6; Ada 8.9.
if [ -z "${CUDA_ARCH:-}" ]; then
    if command -v nvidia-smi >/dev/null; then
        CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
                    | tr -d '. ' | sort -u | paste -sd';')
    fi
    : "${CUDA_ARCH:?could not detect GPU compute capability; set CUDA_ARCH=75}"
fi
echo "==> CUDA architectures: $CUDA_ARCH"

# ── Source ───────────────────────────────────────────────────────────────────
if [ -d "$SRC/.git" ]; then
    echo "==> Updating $SRC"
    git -C "$SRC" pull --ff-only
else
    echo "==> Cloning into $SRC"
    git clone "$REPO" "$SRC"
fi

# ── Build ────────────────────────────────────────────────────────────────────
# INSTALL_RPATH matters: without it the installed binaries resolve
# libllama-server-impl.so only when LD_LIBRARY_PATH happens to include
# $PREFIX/lib, so they work from an interactive shell but fail under systemd,
# cron, or any other launcher with a clean environment.
echo "==> Configuring"
cmake -B "$SRC/build" -S "$SRC" \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
      -DCMAKE_INSTALL_PREFIX="$PREFIX" \
      -DCMAKE_INSTALL_RPATH="$PREFIX/lib" \
      -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON

echo "==> Building with $JOBS jobs"
cmake --build "$SRC/build" --config Release -j "$JOBS"

echo "==> Installing to $PREFIX"
cmake --install "$SRC/build" --prefix "$PREFIX"

# ── Verify ───────────────────────────────────────────────────────────────────
BIN="$PREFIX/bin/llama-server"
echo "==> Verifying (with a deliberately clean environment)"
if ! VER=$(env -u LD_LIBRARY_PATH "$BIN" --version 2>&1); then
    echo "FAILED: $BIN does not run without LD_LIBRARY_PATH:"
    echo "$VER"
    exit 1
fi
echo "$VER" | grep -iE "^version|^build" || true

# "version: 0.1.0-dev (build 10438, commit ...)" — take the build number, not
# the leading 0 of the version string.
BUILD_NUM=$(echo "$VER" | grep -oE 'build[: ]+[0-9]+' | grep -oE '[0-9]+' | head -1)
if [ -n "$BUILD_NUM" ] && [ "$BUILD_NUM" -lt "$MIN_BUILD" ]; then
    echo "WARNING: build $BUILD_NUM is older than b$MIN_BUILD."
    echo "         Qwen3-ASR transcribes to empty output on such builds."
    exit 1
fi

# --version does not mention CUDA; --list-devices is what enumerates GPUs.
if env -u LD_LIBRARY_PATH "$BIN" --list-devices 2>&1 | grep -q "CUDA"; then
    echo "==> CUDA devices visible:"
    env -u LD_LIBRARY_PATH "$BIN" --list-devices 2>&1 | grep "CUDA" | sed 's/^/    /'
else
    echo "WARNING: no CUDA devices listed — the build may be CPU-only."
    echo "         ASR will still run but far slower."
fi

case ":$PATH:" in
    *":$PREFIX/bin:"*) ;;
    *) echo "NOTE: $PREFIX/bin is not on PATH; add it or set LLAMA_SERVER_BIN=$BIN" ;;
esac

echo "==> Done. Build $BUILD_NUM installed to $PREFIX"
