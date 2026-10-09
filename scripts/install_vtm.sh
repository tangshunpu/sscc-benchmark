#!/usr/bin/env bash
# Linux x86-64; VTM is reference software and can encode slowly.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
install_dir="${1:-${repo_dir}/.local}"
source_dir="${repo_dir}/third_party/VVCSoftware_VTM"
vtm_ref="${VTM_REF:-VTM-23.14}"
mkdir -p "${repo_dir}/third_party" "${install_dir}/bin"
if [[ ! -d "${source_dir}" ]]; then
  git clone --depth 1 --branch "${vtm_ref}" \
    https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git "${source_dir}"
fi
cmake -S "${source_dir}" -B "${source_dir}/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "${source_dir}/build" --config Release -j "${JOBS:-2}"
for app in EncoderApp DecoderApp; do
  binary="$(find "${source_dir}/bin" -type f \( -name "${app}" -o -name "${app}Static" \) -executable -print -quit)"
  if [[ -z "${binary}" ]]; then
    printf 'Could not locate %s in %s/bin\n' "${app}" "${source_dir}" >&2
    exit 1
  fi
  install -m 755 "${binary}" "${install_dir}/bin/${app}"
done
git -C "${source_dir}" rev-parse HEAD > "${install_dir}/vtm-commit.txt"
printf 'VTM installed in %s/bin; config: %s/cfg/encoder_intra_vtm.cfg\n' "${install_dir}" "${source_dir}"
