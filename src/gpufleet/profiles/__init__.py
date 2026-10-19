"""Box software profiles: what a box must have before it takes work.

A profile is the same on every provider: on Docker providers (Vast) most of it is baked into `vast_image` and the
bootstrap only verifies; on Colab (no Docker) the bootstrap installs it. Bootstraps are idempotent shell, run once
per box over the existing SSH connection, and must print `GF_OK` on success.

`caches` are directories whose contents are worth carrying from box to box (compiled OptiX kernels, the NVIDIA GL
shader cache). The daemon keeps one copy per (profile, GPU model, driver) on this machine, restores it on boot and
saves it when the box is destroyed, so only the first box of a kind pays kernel and shader compilation.
"""

from __future__ import annotations

COMMON = r"""
set -u
mkdir -p /work /var/lib/gpufleet /var/cache/gpufleet
command -v rsync >/dev/null && command -v python3 >/dev/null || {
  export DEBIAN_FRONTEND=noninteractive
  (apt-get -qq update && apt-get -qq install -y rsync python3) >/dev/null 2>&1
}
command -v rsync >/dev/null || { echo "GF_FAIL no rsync"; exit 1; }
command -v python3 >/dev/null || { echo "GF_FAIL no python3"; exit 1; }
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 | head -1 | sed 's/^/GF_GPU /'
  nvidia-smi -L >/dev/null 2>&1 || { echo "GF_FAIL nvidia-smi cannot see a GPU"; exit 3; }
fi
# Colab keeps the NVIDIA userspace libraries in /usr/lib64-nvidia but registers no EGL vendor, so headless EGL
# (EEVEE, any OpenGL) silently falls back to Mesa's software renderer. Register the NVIDIA one.
if [ -e /usr/lib64-nvidia/libEGL_nvidia.so.0 ] && [ ! -e /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
  mkdir -p /usr/share/glvnd/egl_vendor.d
  printf '{"file_format_version":"1.0.0","ICD":{"library_path":"/usr/lib64-nvidia/libEGL_nvidia.so.0"}}\n' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
fi
echo "GF_CPU $(nproc)"
"""

BLENDER = r"""
BV=${GF_BLENDER_VERSION:-5.2.2}
BLSHA=${GF_BLENDER_SHA256:-84098912789dc450e95697c4184fb8a90acbe5111c2ba4aede3fecb57806a168}
B=$(readlink -f "$(command -v blender || true)" 2>/dev/null || true)
[ -z "$B" ] && [ -x /opt/blender/blender ] && B=/opt/blender/blender
if [ -n "$B" ] && "$B" -b --factory-startup -noaudio --version 2>/dev/null | grep -q "Blender $BV"; then :; else
  # download.blender.org sometimes truncates or serves HTML: try mirrors in turn, check the hash.
  f=blender-$BV-linux-x64.tar.xz; v=${BV%.*}; ok=
  for m in https://download.blender.org/release https://mirrors.ocf.berkeley.edu/blender/release \
           https://ftp.nluug.nl/pub/graphics/blender/release https://mirror.clarkson.edu/blender/release; do
    curl -fsSL --retry 2 --connect-timeout 15 -o /tmp/$f "$m/Blender$v/$f" || continue
    echo "$BLSHA  /tmp/$f" | sha256sum -c --status && { ok=1; break; }
  done
  [ -n "$ok" ] || { echo "GF_FAIL blender download"; exit 1; }
  rm -rf /opt/blender && mkdir -p /opt/blender && tar -xf /tmp/$f -C /opt/blender --strip-components=1 && rm -f /tmp/$f
  export DEBIAN_FRONTEND=noninteractive
  (apt-get -qq install -y libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 libegl1 libsm6 libxkbcommon0 \
     || (apt-get -qq update && apt-get -qq install -y libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 libegl1 \
         libsm6 libxkbcommon0)) >/dev/null 2>&1
  B=/opt/blender/blender
fi
# Put it on PATH unless it already is (never link a path onto itself).
[ "$(readlink -f /usr/local/bin/blender 2>/dev/null)" = "$B" ] || ln -sf "$B" /usr/local/bin/blender
"$B" -b --factory-startup -noaudio --version 2>/dev/null | head -1 | sed 's/^/GF_BLENDER /'
# GPU self-test before the box takes work (measured 19 Oct 2026: a Vast 4090 host passed nvidia-smi but could not
# load OptiX kernels, and its EEVEE fell back to software at 45 s/frame). 64x64, 1 sample, a few seconds; it also
# warms the OptiX and shader caches for the real renders.
if [ "${GF_SELFTEST:-1}" = 1 ] && command -v nvidia-smi >/dev/null; then
  cat > /tmp/gf_selftest.py <<'PY'
import sys, time
import bpy
s = bpy.context.scene
s.render.resolution_x = s.render.resolution_y = 64
p = bpy.context.preferences.addons["cycles"].preferences
p.compute_device_type = "OPTIX"
p.refresh_devices()
if not [d for d in p.devices if d.type == "OPTIX"]:
    print("GF_SELFTEST_FAIL no OptiX device"); sys.exit(3)
for d in p.devices:
    d.use = d.type == "OPTIX"
s.render.engine = "CYCLES"; s.cycles.device = "GPU"; s.cycles.samples = 1
s.render.filepath = "/tmp/gf_st_cycles.png"
t = time.time(); bpy.ops.render.render(write_still=True); tc = time.time() - t
for name in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
    try:
        s.render.engine = name; break
    except TypeError:
        pass
s.render.filepath = "/tmp/gf_st_eevee.png"
t = time.time(); bpy.ops.render.render(write_still=True); te = time.time() - t
import gpu
r = gpu.platform.renderer_get().lower()
if "llvmpipe" in r or "softpipe" in r or "swrast" in r:
    print("GF_SELFTEST_FAIL EEVEE on a software renderer:", r); sys.exit(3)
print("GF_SELFTEST_OK cycles_s=%.1f eevee_s=%.1f gl=%s" % (tc, te, r))
PY
  st=$(timeout 300 "$B" -b --factory-startup -noaudio -P /tmp/gf_selftest.py 2>&1 </dev/null)
  echo "$st" | grep -E '^GF_SELFTEST' | head -2
  if ! echo "$st" | grep -q '^GF_SELFTEST_OK'; then
    echo "$st" | tail -n 15 | sed 's/^/GF_LOG /'
    echo "GF_FAIL gpu self-test"; exit 3
  fi
  if echo "$st" | grep -qiE 'Failed to load OptiX kernel|OPTIX_ERROR'; then echo "GF_FAIL optix kernel load"; exit 3; fi
fi
true
"""

BUILTIN: dict[str, dict] = {
    "base": {
        "vast_image": "nvidia/cuda:12.8.1-base-ubuntu24.04",
        "disk_gb": 40,
        "bootstrap": COMMON,
        "env": {},
        "caches": [],
    },
    "blender": {
        # Prebaked toon-studio render box: Blender 5.2.2, Node, ffmpeg 7, Chrome. Any image with blender on PATH
        # (or none: the bootstrap installs it) works.
        "vast_image": "ghcr.io/siinghd/numora-render-box:5.2.2-20261008",
        "disk_gb": 40,
        "bootstrap": COMMON + BLENDER,
        "env": {
            "OPTIX_CACHE_PATH": "/var/cache/gpufleet/optix",
            "OPTIX_CACHE_MAXSIZE": "4294967296",
            # The NVIDIA GL shader cache is off by default for root (how boxes run); EEVEE first frames go from
            # ~9 s to ~1.3 s with it warm (measured on an RTX PRO 6000, 2026-10-19).
            "__GL_SHADER_DISK_CACHE": "1",
            "__GL_SHADER_DISK_CACHE_PATH": "/var/cache/gpufleet/glcache",
            "__GL_SHADER_DISK_CACHE_SKIP_CLEANUP": "1",
            "__GL_SHADER_DISK_CACHE_SIZE": "4294967296",
        },
        "caches": ["/var/cache/gpufleet/optix", "/var/cache/gpufleet/glcache"],
    },
}


def get(name: str, overrides: dict | None = None) -> dict:
    base = dict(BUILTIN.get(name, {}))
    extra = (overrides or {}).get(name, {})
    if not base and not extra:
        raise KeyError(f"unknown profile {name!r} (known: {sorted(set(BUILTIN) | set(overrides or {}))})")
    prof = {**base, **extra, "name": name}
    if "bootstrap_extra" in extra:
        prof["bootstrap"] = prof.get("bootstrap", COMMON) + "\n" + extra["bootstrap_extra"]
    prof["env"] = {**base.get("env", {}), **extra.get("env", {})}
    return prof
