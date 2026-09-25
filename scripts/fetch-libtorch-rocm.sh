#!/usr/bin/env bash
# Downloads the official PyTorch libtorch build for ROCm into the cache that
# -p:TorchBackend=rocm-linux and --torch-lib-dir use. The archive is about
# 4.9 GB; the shared libraries plus the rocBLAS and hipBLASLt kernel databases
# are kept, which is about 9.4 GB.
set -euo pipefail

version="2.10.0+rocm7.0"
sha256="d8561904e2cee6af1be8083ede61cef82fdeb2586697bcff66488bea1539e0e1"
url="https://download.pytorch.org/libtorch/rocm7.0/libtorch-shared-with-deps-2.10.0%2Brocm7.0.zip"
destination="${1:-${XDG_CACHE_HOME:-$HOME/.cache}/llavon-lora/libtorch-rocm7.0-2.10.0}"
library="${destination}/libtorch/lib/libtorch_hip.so"

for tool in curl unzip sha256sum; do
    command -v "${tool}" >/dev/null || { echo "missing required tool: ${tool}" >&2; exit 1; }
done

if [[ -f "${library}" ]]; then
    echo "libtorch ${version} is already installed at ${destination}/libtorch/lib" >&2
    exit 0
fi

mkdir -p "${destination}"
archive="${destination}/libtorch-${version}.zip.partial"
trap 'rm -f "${archive}"' EXIT

echo "downloading libtorch ${version} (about 4.9 GB)" >&2
curl --fail --location --retry 3 --output "${archive}" "${url}"
echo "${sha256}  ${archive}" | sha256sum --check --status || {
    echo "checksum mismatch for ${archive}; remove it and retry" >&2
    exit 1
}

unzip -q -o "${archive}" \
    'libtorch/lib/*.so*' \
    'libtorch/lib/rocblas/library/*' \
    'libtorch/lib/hipblaslt/library/*' \
    'libtorch/lib/hipsparselt/library/*' \
    'libtorch/build-version' \
    -d "${destination}"
rm -f "${archive}"
echo "libtorch ${version} is ready at ${destination}/libtorch/lib" >&2
