#!/bin/sh
# Installed as /usr/sbin/sshd in the image; the real binary is dpkg-diverted to /usr/sbin/sshd.distrib.
# The image is public, so it ships without SSH host keys: each box makes its own the first time anything starts or
# tests sshd. Vast's SSH launcher starts sshd before any onstart script runs and documents no hook before that, so the
# keys are made here, whichever way it starts sshd (/usr/sbin/sshd directly, or "service ssh start", which runs it too).
# ssh-keygen -A only creates the key types that are missing, so later starts cost nothing.
ssh-keygen -A >&2 || exit 1
exec /usr/sbin/sshd.distrib "$@"
