#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
julia_project="${JULIA_PROJECT:-$repo_dir}"

exec julia --project="$julia_project" "$script_dir/run.jl" "$@"
