"""Lossless shared-storage transport of images already prepared for training."""

import hashlib
import os
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

from PIL import Image

# Identity describes the transport contract. Prepared pixels themselves capture
# upstream resize/configuration changes without relying on source path identity.
_CACHE_VERSION = b"miles-prepared-rgb-png-v1\0"
_VALIDATED = OrderedDict()
_LOCK = threading.Lock()
_MAX_VALIDATED = 4096


def _image_digest(image):
    digest = hashlib.sha256(_CACHE_VERSION)
    digest.update(f"{image.mode}:{image.width}:{image.height}\0".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def _validate(path, expected_digest, retries=3):
    stat = path.stat()
    identity = (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    with _LOCK:
        if _VALIDATED.get(identity) == expected_digest:
            _VALIDATED.move_to_end(identity)
            return
    try:
        with Image.open(path) as cached:
            if cached.format != "PNG" or cached.mode != "RGB" or _image_digest(cached) != expected_digest:
                raise ValueError("prepared image content does not match its cache key")
    except Exception as exc:
        raise ValueError(f"Invalid immutable rollout image cache entry: {path}") from exc
    after = path.stat()
    after_identity = (str(path), after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if after_identity != identity:
        # Removing the publisher's temporary hard link legitimately changes
        # ctime. Revalidate instead of mistaking this race for corruption.
        if retries:
            return _validate(path, expected_digest, retries=retries - 1)
        raise ValueError(f"Rollout image cache entry changed during validation: {path}")
    with _LOCK:
        _VALIDATED[identity] = expected_digest
        _VALIDATED.move_to_end(identity)
        while len(_VALIDATED) > _MAX_VALIDATED:
            _VALIDATED.popitem(last=False)


def cache_prepared_image(image, cache_dir):
    """Publish prepared RGB pixels once; return an absolute, immutable PNG path.

    Every reader checks file identity and validates pixels on first observation
    or after a filesystem change. Corrupt entries fail closed; they are never
    silently replaced while another engine may be reading them. Cache storage
    must support atomic hard links and be visible to every rollout/teacher worker.
    """
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    digest = _image_digest(rgb)
    directory = Path(cache_dir).expanduser().resolve() / "prepared-rgb-v1" / digest[:2]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.png"
    if not path.exists():
        fd, temporary = tempfile.mkstemp(prefix=".image-", suffix=".png", dir=directory)
        try:
            with os.fdopen(fd, "wb") as output:
                rgb.save(output, format="PNG")
                output.flush()
                os.fsync(output.fileno())
            os.chmod(temporary, 0o444)
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass  # A concurrent publisher won; validate its content below.
        finally:
            os.unlink(temporary)
    _validate(path, digest)
    return str(path)


def prepared_image_paths(images, cache_dir):
    """Preserve image order and duplicates in the engine payload."""
    return [cache_prepared_image(image, cache_dir) for image in images]
