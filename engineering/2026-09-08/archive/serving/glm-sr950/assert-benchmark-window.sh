#!/usr/bin/env bash
# Refuse a GLM benchmark when an unapproved llama.cpp model process is present.
set -euo pipefail

allowed_ports=",${GLM_BENCH_ALLOWED_LLAMA_PORTS:-18091,18081,5811,5804},"
offenders=()

for cmdline_file in /proc/[0-9]*/cmdline; do
  [[ -r "$cmdline_file" ]] || continue
  argv=()
  mapfile -d '' -t argv <"$cmdline_file" 2>/dev/null || true
  (( ${#argv[@]} > 0 )) || continue

  executable="${argv[0]##*/}"
  case "$executable" in
    llama-server|llama-cli|llama-bench)
      ;;
    *)
      continue
      ;;
  esac

  port=""
  for ((index = 1; index + 1 < ${#argv[@]}; index++)); do
    if [[ "${argv[index]}" == "--port" ]]; then
      port="${argv[index + 1]}"
      break
    fi
  done

  if [[ -n "$port" && "$allowed_ports" == *",$port,"* ]]; then
    continue
  fi

  pid="${cmdline_file#/proc/}"
  pid="${pid%/cmdline}"
  offenders+=("pid=$pid executable=$executable port=${port:-none} model=${argv[*]}")
done

if (( ${#offenders[@]} > 0 )); then
  echo "Benchmark window is not single-model; refusing contaminated run." >&2
  printf '  %s\n' "${offenders[@]}" >&2
  echo "Allowed llama-server ports: ${allowed_ports#,}" >&2
  echo "Override GLM_BENCH_ALLOWED_LLAMA_PORTS only for an announced window." >&2
  exit 75
fi
