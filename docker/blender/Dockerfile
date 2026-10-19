# syntax=docker/dockerfile:1
# gpufleet "blender" profile image: Blender + an NVENC-capable ffmpeg on a small CUDA base, nothing else.
# Cold boots on GPU marketplaces are dominated by the image pull, so every layer here earns its size:
#   - ubuntu base: ~30 MB and already cached on most hosts (only our layers are pulled)
#   - Blender trimmed of what a headless render never touches (static libpython, UI translations)
#   - each big tree in its own layer, so a Blender bump leaves the ffmpeg layer byte-identical
# Built by .github/workflows/blender-image.yml for linux/amd64.

ARG BLV=5.2.2

FROM --platform=$BUILDPLATFORM ubuntu:24.04 AS fetch
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl xz-utils && rm -rf /var/lib/apt/lists/*

FROM fetch AS blender
ARG BLV
ARG BLSHA=84098912789dc450e95697c4184fb8a90acbe5111c2ba4aede3fecb57806a168
ARG MIRRORS="https://download.blender.org/release https://mirrors.ocf.berkeley.edu/blender/release https://ftp.nluug.nl/pub/graphics/blender/release"
RUN set -e; f=blender-$BLV-linux-x64.tar.xz; ok=; \
    for m in $MIRRORS; do \
      curl -fsSL --retry 3 -o /tmp/$f "$m/Blender${BLV%.*}/$f" && echo "$BLSHA  /tmp/$f" | sha256sum -c - && { ok=1; break; }; \
    done; [ -n "$ok" ]; \
    mkdir -p /opt/blender && tar -xf /tmp/$f -C /opt/blender --strip-components=1 && rm /tmp/$f; \
    find /opt/blender -name 'libpython*.a' -delete; \
    # NVIDIA-only boxes: AMD (HIP) kernels and the AMD/Intel denoiser plugins are dlopen()ed only on those GPUs.
    # (libhiprt, libsycl and the oneAPI kernel library stay: the blender binary links them directly.)
    rm -f /opt/blender/5.2/scripts/addons_core/cycles/lib/kernel_gfx*.fatbin.zst \
          /opt/blender/lib/libOpenImageDenoise_device_hip.so* /opt/blender/lib/libOpenImageDenoise_device_sycl.so*; \
    find /opt/blender -path '*/datafiles/locale/*' -mindepth 1 -maxdepth 4 -type d ! -name languages -exec rm -rf {} + 2>/dev/null || true

FROM fetch AS ffmpeg
# BtbN static GPL build: includes h264_nvenc/hevc_nvenc (the distro-independent static builds most images ship do not).
ARG FFURL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz
RUN set -e; curl -fsSL --retry 3 -o /tmp/ff.tar.xz "$FFURL"; mkdir -p /opt/ffmpeg; \
    tar -xf /tmp/ff.tar.xz -C /opt/ffmpeg --strip-components=1; rm /tmp/ff.tar.xz; \
    rm -rf /opt/ffmpeg/doc /opt/ffmpeg/man /opt/ffmpeg/bin/ffplay

# Plain Ubuntu, not nvidia/cuda: Blender ships its own CUDA/OptiX kernels and the NVIDIA container runtime mounts the
# driver libraries (libcuda, libnvoptix, EGL, NVENC) when these variables are set. Ubuntu's base layer is the one
# most likely to be cached on a host already.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive NVIDIA_DRIVER_CAPABILITIES=all NVIDIA_VISIBLE_DEVICES=all
# Runtime libraries Blender needs headless (X/GL client libs are linked even in background mode), EGL via glvnd for
# EEVEE, python3 + numpy for scene scripts and checks, rsync/openssh for gpufleet.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl rsync openssh-server python3 python3-numpy \
      libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libsm6 libgl1 libegl1 libglvnd0 libvulkan1 \
    && rm -rf /var/lib/apt/lists/* \
    # A public image must not ship SSH host keys (every box would share them): delete the ones the package made and
    # wrap sshd so each box generates its own on first start (Vast starts sshd before any onstart hook).
    && dpkg-divert --local --rename --add /usr/sbin/sshd \
    && rm -f /etc/ssh/ssh_host_* \
    && mkdir -p /run/sshd /root/.ssh /work && chmod 700 /root/.ssh \
    && mkdir -p /usr/share/glvnd/egl_vendor.d \
    && printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
       > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY --chmod=755 sshd-wrapper.sh /usr/sbin/sshd
COPY --link --from=ffmpeg /opt/ffmpeg/bin/ /usr/local/bin/
COPY --link --from=blender /opt/blender /opt/blender
ARG BLV
RUN ln -s /opt/blender/blender /usr/local/bin/blender && echo "gpufleet-blender ${BLV}" > /opt/gpufleet-image \
    && [ -z "$(find /etc/ssh -name 'ssh_host_*')" ] && [ -x /usr/sbin/sshd.distrib ]
