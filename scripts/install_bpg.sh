#!/usr/bin/env bash
# Linux x86-64; install OS build dependencies listed in docs/codecs.md first.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
install_dir="${1:-${repo_dir}/.local}"
source_dir="${repo_dir}/third_party/libbpg-0.9.8"
mkdir -p "${repo_dir}/third_party" "${install_dir}/bin"
if [[ ! -f "${source_dir}/Makefile" ]]; then
  curl --fail --location --retry 3 https://bellard.org/bpg/libbpg-0.9.8.tar.gz \
    --output "${repo_dir}/third_party/libbpg-0.9.8.tar.gz"
  tar -xzf "${repo_dir}/third_party/libbpg-0.9.8.tar.gz" -C "${repo_dir}/third_party"
fi
# Generate headers before dependency files from a previous build are read.
make -C "${source_dir}" CMAKE_OPTS=-DENABLE_LIBNUMA=OFF x265.out
for depth in 8 10 12; do
  cmake -S "${source_dir}/x265/source" -B "${source_dir}/x265.out/${depth}bit" -DENABLE_LIBNUMA=OFF
done
make -C "${source_dir}" -j "${JOBS:-2}" USE_BPGVIEW= CMAKE_OPTS=-DENABLE_LIBNUMA=OFF bpgenc bpgdec
install -m 755 "${source_dir}/bpgenc" "${source_dir}/bpgdec" "${install_dir}/bin/"
printf 'BPG installed in %s/bin; add it to PATH.\n' "${install_dir}"
