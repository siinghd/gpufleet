"""Content-addressed blob transfer through object storage (Cloudflare R2, any S3 API), for inputs and outputs that are
large or that would otherwise crawl over a slow SSH link (Colab's tunnel measured 5.7 MB/s up).

A directory or file is packed once into a zstd tar, named by the SHA-256 of its content, uploaded once, and fetched by
any number of boxes in parallel straight from the store. Boxes never see credentials: the daemon hands them presigned
URLs for single objects that expire after an hour, so a box can read or write exactly the object it was given and
nothing else in the bucket.

Config (~/.config/gpufleet/config.toml):
    [blobs]
    env_file = "~/.config/gpufleet/r2.env"   # R2_ENDPOINT, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
    prefix = "gpufleet/"
The "local" backend (a directory and file:// URLs) exists for tests.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

ZSTD = shutil.which("zstd") is not None


def _env_file(path: str) -> dict:
    out = {}
    for line in Path(path).expanduser().read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip("'\"")
    return out


def fingerprint(path: Path) -> str:
    """Cheap change detector (paths, sizes, mtimes) so an unchanged tree is not re-packed."""
    h = hashlib.sha256()
    if path.is_file():
        st = path.stat()
        h.update(f"{path.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    else:
        for p in sorted(path.rglob("*")):
            if p.is_file():
                st = p.stat()
                h.update(f"{p.relative_to(path)}:{st.st_size}:{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


class Blobs:
    def __init__(self, cfg: dict, cache_dir: Path):
        self.cfg = cfg
        self.prefix = cfg.get("prefix", "gpufleet/")
        self.cache = cache_dir
        self.cache.mkdir(parents=True, exist_ok=True)
        self.backend = cfg.get("backend", "s3")
        self._s3 = None
        self._packed: dict[str, tuple[str, str]] = {}  # fingerprint -> (digest, packed file)
        if self.backend == "local":
            self.root = Path(cfg["root"])
            self.root.mkdir(parents=True, exist_ok=True)
        else:
            env = _env_file(cfg.get("env_file", "~/.config/gpufleet/r2.env"))
            self.bucket = env.get("R2_BUCKET") or cfg.get("bucket")
            self._creds = env

    def _client(self):
        if self._s3 is None:
            import boto3
            from botocore.config import Config
            e = self._creds
            self._s3 = boto3.client("s3", endpoint_url=e["R2_ENDPOINT"], aws_access_key_id=e["R2_ACCESS_KEY_ID"],
                                    aws_secret_access_key=e["R2_SECRET_ACCESS_KEY"], region_name="auto",
                                    config=Config(signature_version="s3v4", retries={"max_attempts": 5}))
        return self._s3

    # -- packing -------------------------------------------------------------------------------------------------
    def _pack(self, src: Path, exclude: tuple[str, ...]) -> tuple[str, str]:
        fp = fingerprint(src) + "|" + ",".join(exclude)
        if fp in self._packed and os.path.exists(self._packed[fp][1]):
            return self._packed[fp]
        fd, tmp = tempfile.mkstemp(dir=self.cache, suffix=".tar.zst" if ZSTD else ".tar.gz")
        os.close(fd)
        argv = ["tar", "-C", str(src.parent if src.is_file() else src), "--sort=name", "--mtime=@0", "--owner=0",
                "--group=0", "--numeric-owner", *sum((["--exclude", x] for x in exclude), []),
                "-I", "zstd -T0 -3" if ZSTD else "gzip -1", "-cf", tmp, src.name if src.is_file() else "."]
        subprocess.run(argv, check=True)
        h = hashlib.sha256()
        with open(tmp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
        final = self.cache / (digest + (".tar.zst" if ZSTD else ".tar.gz"))
        os.replace(tmp, final)
        self._packed[fp] = (digest, str(final))
        return digest, str(final)

    # -- store ---------------------------------------------------------------------------------------------------
    def _key(self, digest: str, ext: str) -> str:
        return f"{self.prefix}blobs/{digest}{ext}"

    def _exists(self, key: str) -> bool:
        if self.backend == "local":
            return (self.root / key).exists()
        try:
            self._client().head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def _upload(self, key: str, path: str) -> None:
        if self.backend == "local":
            dst = self.root / key
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dst)
            return
        from boto3.s3.transfer import TransferConfig
        self._client().upload_file(path, self.bucket, key, Config=TransferConfig(
            multipart_threshold=16 << 20, multipart_chunksize=16 << 20, max_concurrency=8))

    def url_get(self, key: str, ttl: int = 3600) -> str:
        if self.backend == "local":
            return (self.root / key).as_uri()
        return self._client().generate_presigned_url("get_object", Params={"Bucket": self.bucket, "Key": key},
                                                     ExpiresIn=ttl)

    def url_put(self, key: str, ttl: int = 3600) -> str:
        if self.backend == "local":
            (self.root / key).parent.mkdir(parents=True, exist_ok=True)
            return (self.root / key).as_uri()
        return self._client().generate_presigned_url("put_object", Params={"Bucket": self.bucket, "Key": key},
                                                     ExpiresIn=ttl)

    def download(self, key: str, dst: str) -> None:
        if self.backend == "local":
            shutil.copyfile(self.root / key, dst)
            return
        self._client().download_file(self.bucket, key, dst)

    def delete(self, key: str) -> None:
        if self.backend == "local":
            (self.root / key).unlink(missing_ok=True)
            return
        self._client().delete_object(Bucket=self.bucket, Key=key)

    # -- high level ----------------------------------------------------------------------------------------------
    async def stage(self, src: str, exclude: tuple[str, ...] = ()) -> tuple[str, str]:
        """Pack and upload (once) a local file or tree. Returns (digest, presigned GET URL)."""
        def work():
            p = Path(os.path.expanduser(src)).resolve()
            digest, packed = self._pack(p, exclude)
            key = self._key(digest, Path(packed).suffix if not packed.endswith(".tar.zst") else ".tar.zst")
            if not self._exists(key):
                self._upload(key, packed)
            return digest, self.url_get(key)
        return await asyncio.to_thread(work)

    @staticmethod
    def fetch_cmd(url: str, digest: str, dst: str) -> str:
        """Shell for the box: download and unpack into dst unless this exact blob is already there."""
        z = "zstd -d" if url.split("?")[0].endswith(".zst") else "gzip -d"
        q = lambda s: "'" + s.replace("'", "'\\''") + "'"
        mark = f"{dst.rstrip('/')}/.gf-blob-{digest[:16]}"
        return (f"[ -e {q(mark)} ] || {{ mkdir -p {q(dst)} && curl -fsSL --retry 4 --retry-all-errors {q(url)} "
                f"| {z} | tar -x -C {q(dst)} && rm -f {q(dst.rstrip('/'))}/.gf-blob-* && touch {q(mark)}; }}")

    def output_slot(self, name: str) -> tuple[str, str]:
        """(key, presigned PUT URL) for a box to upload one packed output to."""
        key = f"{self.prefix}out/{int(time.time())}-{name}.tar.zst"
        return key, self.url_put(key)

    @staticmethod
    def upload_cmd(url: str, src: str) -> str:
        q = lambda s: "'" + s.replace("'", "'\\''") + "'"
        if url.startswith("file://"):
            dst = url[len("file://"):]
            return f"tar -C {q(src)} -cf - . | zstd -q -T0 -3 > {q(dst)}"
        return f"tar -C {q(src)} -cf - . | zstd -q -T0 -3 | curl -fsS --retry 4 -X PUT --upload-file - {q(url)}"

    async def collect(self, key: str, dst: str) -> None:
        """Download an output blob a box uploaded and unpack it into dst, then delete it from the store."""
        def work():
            Path(dst).mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.cache, suffix=".tar.zst")
            os.close(fd)
            try:
                self.download(key, tmp)
                subprocess.run(["tar", "-I", "zstd -d", "-xf", tmp, "-C", dst], check=True)
            finally:
                os.unlink(tmp)
            self.delete(key)
        await asyncio.to_thread(work)
