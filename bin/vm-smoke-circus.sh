#!/bin/bash
# Shared Circus CLI helpers for bin/vm-smoke*.sh — source only, do not execute.
: "${CIRCUS_PROFILE:=tart-av-dev}"
export CIRCUS_PROFILE

circus_ensure_ready() {
  local timeout="${1:-180}"
  circus ensure-ready --profile "$CIRCUS_PROFILE" --timeout "$timeout"
}

# Run one guest shell command. Pass the command as a single quoted string (paths with
# spaces break if argv is split). Optional --env K=V and --timeout N precede --.
circus_exec() {
  local -a opts=()
  local cmd=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --env|--timeout)
        opts+=("$1" "$2")
        shift 2
        ;;
      --)
        shift
        cmd="$*"
        break
        ;;
      *)
        cmd="$*"
        break
        ;;
    esac
  done

  if [[ -z "$cmd" ]]; then
    echo "circus_exec: missing command" >&2
    return 1
  fi

  circus exec "${opts[@]}" -- "$cmd"
}

circus_put() {
  circus put --host "$1" --guest "$2"
}

circus_get() {
  circus get --guest "$1" --host "$2"
}

circus_wait_file() {
  local guest_path="$1"
  local timeout="${2:-60}"
  local min_bytes="${3:-0}"
  if (( min_bytes > 0 )); then
    circus wait --guest "$guest_path" --min-bytes "$min_bytes" --timeout "$timeout"
  else
    circus wait --guest "$guest_path" --timeout "$timeout"
  fi
}

circus_path_is_dir() {
  circus_exec --timeout 30 -- "test -d $(printf '%q' "$1")"
}
