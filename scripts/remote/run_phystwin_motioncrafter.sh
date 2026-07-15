#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CASE [CASE ...]" >&2
  exit 2
fi

revision="1d6a8947ec6ebabbcf4fc1e0f6d06828fcf6f257"
prob4d_revision="81499fc0e76c87444266bc96f90b01db1beccc67"
repo_root="${BPT_REPO_ROOT:-/home/florianpfaff/Bayesian-PhysTwin}"
data_root="${BPT_DATA_ROOT:-/home/florianpfaff/bayesian-phystwin-confirmatory-data}"
raw_root="${BPT_RAW_ROOT:-/mnt/corsair/florianpfaff/phystwin-data/extracted/data/different_types}"
output_root="${BPT_MOTIONCRAFTER_OUTPUT_ROOT:-/mnt/corsair/florianpfaff/motioncrafter-phystwin-v1}"
motioncrafter_root="${MOTIONCRAFTER_ROOT:-/home/florianpfaff/MotionCrafter}"
cache_root="${MOTIONCRAFTER_CACHE_ROOT:-/mnt/corsair/florianpfaff/motioncrafter-cache}"
motioncrafter_python="${MOTIONCRAFTER_PYTHON:-/home/florianpfaff/.venvs/motioncrafter-v1/bin/python}"
bpt_python="${BPT_PYTHON:-/home/florianpfaff/.venvs/bpt-gpu/bin/python}"
camera_csv="${BPT_MOTIONCRAFTER_CAMERAS:-0}"
input_mode="${BPT_MOTIONCRAFTER_INPUT_MODE:-disjoint}"
prob4d_root="${PROB4D_ROOT:-/home/florianpfaff/Prob4D}"
prob4d_overlap="${PROB4D_MOTIONCRAFTER_OVERLAP:-8}"

if [[ "${input_mode}" != "disjoint" && "${input_mode}" != "prob4d_uniform" ]]; then
  echo "BPT_MOTIONCRAFTER_INPUT_MODE must be disjoint or prob4d_uniform" >&2
  exit 2
fi

actual_revision="$(git -C "${motioncrafter_root}" rev-parse HEAD)"
if [[ "${actual_revision}" != "${revision}" ]]; then
  echo "MotionCrafter revision ${actual_revision} does not match ${revision}" >&2
  exit 1
fi
if [[ "${input_mode}" == "prob4d_uniform" ]]; then
  actual_prob4d_revision="$(git -C "${prob4d_root}" rev-parse HEAD)"
  if [[ "${actual_prob4d_revision}" != "${prob4d_revision}" ]]; then
    echo "Prob4D revision ${actual_prob4d_revision} does not match ${prob4d_revision}" >&2
    exit 1
  fi
fi

IFS=',' read -r -a cameras <<< "${camera_csv}"
for case_name in "$@"; do
  case_dir="${data_root}/${case_name}"
  raw_case_dir="${raw_root}/${case_name}"
  frame_len="$(${bpt_python} -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["frame_len"])' \
    "${case_dir}/split.json")"
  for camera in "${cameras[@]}"; do
    if [[ "${input_mode}" == "prob4d_uniform" ]]; then
      view_dir="${output_root}/${case_name}/camera${camera}_prob4d_uniform_o${prob4d_overlap}"
      bundle_dir="${view_dir}/bundle"
      prediction="${view_dir}/prob4d_uniform.npz"
      association_dir="${view_dir}/association_current"
      if [[ ! -s "${bundle_dir}/predictions.json" ]]; then
        echo "Prob4D MotionCrafter ${case_name} camera ${camera}: ${frame_len} frames"
        PYTHONPATH="${prob4d_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
          "${motioncrafter_python}" -m prob4d.motioncrafter \
            "${raw_case_dir}/color/${camera}.mp4" \
            --upstream-root "${motioncrafter_root}" \
            --output-dir "${bundle_dir}" \
            --cache-dir "${cache_root}" \
            --height 320 \
            --width 576 \
            --window-size 25 \
            --overlap "${prob4d_overlap}" \
            --decode-chunk-size 4 \
            --frame-stop "${frame_len}" \
            --low-memory-usage
      fi
      if [[ ! -s "${prediction}" ]]; then
        PYTHONPATH="${prob4d_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
          "${motioncrafter_python}" \
            "${repo_root}/scripts/remote/export_prob4d_uniform.py" \
            "${bundle_dir}/predictions.json" \
            "${prediction}" \
            --prob4d-root "${prob4d_root}" \
            --expected-prob4d-revision "${prob4d_revision}"
      fi
    else
      view_dir="${output_root}/${case_name}/camera${camera}_native"
      prediction="${view_dir}/${camera}.npz"
      association_dir="${view_dir}/association_frozen"
      if [[ ! -s "${prediction}" ]]; then
        echo "MotionCrafter ${case_name} camera ${camera}: ${frame_len} frames"
        "${motioncrafter_python}" "${motioncrafter_root}/run.py" \
          --video_path="${raw_case_dir}/color/${camera}.mp4" \
          --num_frames="${frame_len}" \
          --save_folder="${view_dir}" \
          --cache_dir="${cache_root}" \
          --height=320 \
          --width=576 \
          --adjust_resolution=True \
          --window_size=25 \
          --decode_chunk_size=4 \
          --process_stride=1 \
          --model_type=determ \
          --low_memory_usage=True
      fi
    fi
    if [[ -s "${association_dir}/summary.json" ]]; then
      echo "skip ${case_name} camera ${camera}: frozen association exists"
      continue
    fi
    PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      "${bpt_python}" -m \
        bayesian_phystwin.cli.phystwin_motioncrafter_association \
        "${case_dir}" \
        "${raw_case_dir}" \
        "${prediction}" \
        "${association_dir}" \
        --camera-index "${camera}" \
        --process-stride 1 \
        --seed-stride-pixels 4 \
        --transport-candidate-count 1 \
        --candidate-count 8 \
        --motion-strength 0
  done
done
