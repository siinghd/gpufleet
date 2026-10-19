#!/usr/bin/env bash
# Box probe + Blender benchmark ladder. Runs on the rented box; prints KEY=value lines and BENCH= JSON lines.
#   bash probe.sh [blender_version=5.2.2] [frames=48]
set -uo pipefail
BV=${1:-5.2.2}; FR=${2:-48}
W=${PROBE_DIR:-/tmp/probe}; mkdir -p "$W"; cd "$W"
say() { printf '%s=%s\n' "$1" "$2"; }
now() { date +%s.%N; }
since() { python3 -c "import sys,time; print(round(time.time()-float(sys.argv[1]),2))" "$1"; }

say gpu "$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -1)"
say cuda "$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | head -1)"
say cpu "$(nproc) x $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
say ram_gb "$(awk '/MemTotal/{printf "%.0f", $2/1048576}' /proc/meminfo)"
say disk "$(df -h "$W" | awk 'NR==2{print $2" total, "$4" free"}')"
say os "$(. /etc/os-release; echo "$PRETTY_NAME")"
say egl_vendor "$(ls /usr/share/glvnd/egl_vendor.d/ 2>/dev/null | tr '\n' ' ')"
say libEGL_nvidia "$(ldconfig -p | grep -c libEGL_nvidia)"
say libnvidia_encode "$(ldconfig -p | grep -c libnvidia-encode)"
say vulkan_icd "$(ls /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/ 2>/dev/null | tr '\n' ' ')"

# Download speed: Blender tarball itself, 8 parallel ranges when aria2c exists.
URL="https://download.blender.org/release/Blender${BV%.*}/blender-${BV}-linux-x64.tar.xz"
BL=$(command -v blender || true)
if [ -n "$BL" ] && "$BL" -b --factory-startup --version 2>/dev/null | grep -q "Blender $BV"; then
  mkdir -p "$W/blender" && ln -sf "$BL" "$W/blender/blender"
  say blender_preinstalled "$BL"
fi
if [ ! -x "$W/blender/blender" ]; then
  command -v aria2c >/dev/null || (apt-get -qq update && apt-get -qq install -y aria2 >/dev/null 2>&1)
  t=$(now)
  for m in https://download.blender.org/release https://mirrors.ocf.berkeley.edu/blender/release https://ftp.nluug.nl/pub/graphics/blender/release; do
    u="$m/Blender${BV%.*}/blender-${BV}-linux-x64.tar.xz"
    if command -v aria2c >/dev/null; then aria2c -q -x8 -s8 -o b.tar.xz "$u" && break; else curl -fsSLo b.tar.xz "$u" && break; fi
  done
  dt=$(since "$t"); sz=$(stat -c %s b.tar.xz)
  say download_MBps "$(python3 -c "print(round($sz/1048576/$dt,1))")"
  t=$(now); mkdir -p blender && tar -xf b.tar.xz -C blender --strip-components=1; rm -f b.tar.xz
  say extract_s "$(since "$t")"
  apt-get -qq install -y libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 libegl1 libsm6 libxkbcommon0 >/dev/null 2>&1
fi
B="$W/blender/blender -b --factory-startup -noaudio"
SC="$(dirname "$(readlink -f "$0")")/cycles_bench.py"
export OPTIX_CACHE_PATH=$W/optix-cache __GL_SHADER_DISK_CACHE=1 __GL_SHADER_DISK_CACHE_PATH=$W/gl-cache
mkdir -p "$OPTIX_CACHE_PATH" "$__GL_SHADER_DISK_CACHE_PATH"

t=$(now); $W/blender/blender -b --factory-startup -noaudio --python-expr 'import bpy' >/dev/null 2>&1
say blender_start_s "$(since "$t")"

run() { local tag=$1; shift; $B -P "$SC" -- --frames "$FR" --out "$W/out-$tag" --tag "$tag" "$@" 2>&1 | grep -E '^BENCH|Error|error:' | tail -3; }
run cold-optix --device OPTIX
run warm-optix --device OPTIX
run no-persist --device OPTIX --no-persistent
run cuda --device CUDA
# Two lanes on one GPU: wall time of both vs one, with SM utilisation sampled meanwhile.
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -lms 500 > "$W/smi2.csv" &
SMI=$!
t=$(now)
run lane-a --device OPTIX & A=$!
run lane-b --device OPTIX & Bp=$!
wait $A $Bp
say two_lanes_wall_s "$(since "$t")"
kill $SMI 2>/dev/null
say two_lanes_util_avg "$(awk -F, '{u+=$1; m=($2>m?$2:m); n++} END{printf "%.0f%% maxmem %dMiB", u/n, m}' "$W/smi2.csv")"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -lms 500 > "$W/smi1.csv" &
SMI=$!; run one-lane-util --device OPTIX; kill $SMI 2>/dev/null
say one_lane_util_avg "$(awk '{u+=$1; n++} END{printf "%.0f%%", u/n}' "$W/smi1.csv")"
run eevee-cold --engine BLENDER_EEVEE --frames 24
run eevee-warm --engine BLENDER_EEVEE --frames 24

# Encode the 1080x1920 frames: NVENC vs libx264 (the current toon-studio setting).
F="$W/out-warm-optix/f_%04d.png"
# The image's static ffmpeg has no NVENC; Ubuntu's does. Prefer one that lists h264_nvenc.
FF=; for f in /usr/bin/ffmpeg $(command -v ffmpeg); do $f -hide_banner -encoders 2>/dev/null | grep -q h264_nvenc && { FF=$f; break; }; done
say ffmpeg_nvenc "${FF:-none}"
if [ -n "$FF" ]; then
  t=$(now); $FF -loglevel error -y -framerate 30 -i "$F" -c:v h264_nvenc -preset p5 -cq 18 "$W/nv.mp4" \
    && say nvenc_s "$(since "$t")" || say nvenc_s failed
fi
t=$(now); ffmpeg -loglevel error -y -framerate 30 -i "$F" -c:v libx264 -crf 16 -g 1 "$W/x264.mp4" && say x264_s "$(since "$t")"
if false; then :
fi
say done 1
