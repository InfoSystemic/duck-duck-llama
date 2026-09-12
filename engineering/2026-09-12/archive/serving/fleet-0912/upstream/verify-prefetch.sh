#!/usr/bin/env bash
set -euo pipefail
report_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(cd -- "$report_dir/../../../engines/llama.cpp-deepseek41-jigsaw-0912" && pwd)
build_dir=$(mktemp -d "$report_dir/prefetch-check.XXXXXX")
trap 'rm -rf -- "$build_dir"' EXIT
cd -- "$source_dir"
flags=(-std=c++17 -O2 -ffunction-sections -fdata-sections -Iinclude -Isrc -Iggml/include)
link_flags=(-Wl,--wrap=posix_madvise -Wl,--gc-sections)
taskset -c 15,31,47,63 c++ "${flags[@]}" "$report_dir/test_prefetch_ranges.cpp" src/llama-mmap.cpp "${link_flags[@]}" -o "$build_dir/fixed"
"$build_dir/fixed"
git show 3b6fcfe4f7e2c282076f0c159278d3acfa3ad4e5:src/llama-mmap.cpp > "$build_dir/upstream.cpp"
taskset -c 15,31,47,63 c++ "${flags[@]}" "$report_dir/test_prefetch_ranges.cpp" "$build_dir/upstream.cpp" "${link_flags[@]}" -o "$build_dir/upstream"
if "$build_dir/upstream" > "$build_dir/upstream.log" 2>&1; then
    echo "FAIL: pristine upstream unexpectedly passed" >&2
    exit 1
fi
rg -F 'FAIL: real posix_madvise succeeded' "$build_dir/upstream.log"
echo "PASS: regression fails on pristine upstream and passes on patched actual source"
