#!/usr/bin/env bash
#
# Source this file after `conda activate sandygs`.
# It configures the CUDA 11.8 build toolchain without hard-coding Conda's
# installation prefix.

if [ "${CONDA_DEFAULT_ENV:-}" != "sandygs" ] || [ -z "${CONDA_PREFIX:-}" ]; then
    echo "Activate the sandygs Conda environment before sourcing this script." >&2
    return 1 2>/dev/null || exit 1
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CC="${CC:-/usr/bin/gcc-11}"
export CXX="${CXX:-/usr/bin/g++-11}"
export MAX_JOBS="${MAX_JOBS:-2}"
export PYTHONDONTWRITEBYTECODE=1

# Respect an explicit architecture override. Otherwise derive all visible
# compute capabilities from the PyTorch environment (for example "8.9" for an
# RTX 4060 or "8.0" for an A100). An empty result is left unset so a headless
# login node does not silently compile for the wrong server GPU.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    _SANDYGS_DETECTED_ARCH_LIST="$(
        python -c 'import torch
caps = []
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(index)
        value = f"{major}.{minor}"
        if value not in caps:
            caps.append(value)
print(" ".join(caps))' 2>/dev/null
    )"
    if [ -n "${_SANDYGS_DETECTED_ARCH_LIST}" ]; then
        export TORCH_CUDA_ARCH_LIST="${_SANDYGS_DETECTED_ARCH_LIST}"
    else
        echo "Warning: no CUDA GPU is visible; TORCH_CUDA_ARCH_LIST remains unset." >&2
        echo "Set it explicitly before compiling on a headless/login node." >&2
    fi
    unset _SANDYGS_DETECTED_ARCH_LIST
fi

case ":${PATH}:" in
    *":${CUDA_HOME}/bin:"*) ;;
    *) export PATH="${CUDA_HOME}/bin:${PATH}" ;;
esac

case ":${LIBRARY_PATH:-}:" in
    *":${CUDA_HOME}/lib64:"*) ;;
    *) export LIBRARY_PATH="${CUDA_HOME}/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}" ;;
esac
case ":${LIBRARY_PATH:-}:" in
    *":${CONDA_PREFIX}/lib:"*) ;;
    *) export LIBRARY_PATH="${CONDA_PREFIX}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}" ;;
esac

case ":${LD_LIBRARY_PATH:-}:" in
    *":${CUDA_HOME}/lib64:"*) ;;
    *) export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac
if [ -d /usr/lib/wsl/lib ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *":/usr/lib/wsl/lib:"*) ;;
        *) export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
    esac
fi
case ":${LD_LIBRARY_PATH:-}:" in
    *":${CONDA_PREFIX}/lib:"*) ;;
    *) export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac

# WSL exposes libcuda.so.1 but often omits the unversioned linker name required
# by NVDIFFREC's `-lcuda`. Keep the compatibility link local to this Conda env.
if [ -e /usr/lib/wsl/lib/libcuda.so.1 ] && [ ! -e "${CONDA_PREFIX}/lib/libcuda.so" ]; then
    ln -s /usr/lib/wsl/lib/libcuda.so.1 "${CONDA_PREFIX}/lib/libcuda.so"
fi

echo "SandyGS environment configured: ${CONDA_PREFIX}"
echo "CUDA architectures: ${TORCH_CUDA_ARCH_LIST:-PyTorch default (not explicitly set)}"
