"""Content-addressed run records and a portable, exclusive run lock."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from src.common.config import PROJECT_ROOT
from src.common.io_utils import json_safe, read_json, write_json, sha256_file


def digest(value):
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def source_hash(paths=None):
    paths = paths or [PROJECT_ROOT / 'src', PROJECT_ROOT / 'scripts', PROJECT_ROOT / 'vendor']
    files = sorted(p for base in paths for p in base.rglob('*.py'))
    return digest({p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p) for p in files})


def environment():
    import importlib.metadata
    import torch
    packages = {}
    for name in ['torch', 'numpy', 'pandas', 'pyarrow', 'scikit-learn', 'PyYAML', 'matplotlib']:
        packages[name] = importlib.metadata.version(name)
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, capture_output=True, text=True)
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', '.'], cwd=PROJECT_ROOT, capture_output=True, text=True)
    return dict(python_executable=sys.executable, python_version=platform.python_version(),
                os=platform.platform(), machine=platform.machine(), packages=packages,
                torch_version=torch.__version__, cuda_version=torch.version.cuda,
                mps_available=torch.backends.mps.is_available(), cuda_available=torch.cuda.is_available(),
                code_hash=source_hash(), git_commit=git.stdout.strip(), git_dirty=bool(dirty.stdout))


@contextmanager
def run_lock(path):
    """Use kernel locks: automatically released on crash, including on Windows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0:
                handle.write(b'0'); handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f'Run is already locked: {path}') from exc
    try:
        yield
    finally:
        handle.close()


def relative_file(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Artifact path escapes its package: {name}')
    return path


def inventory(root):
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(Path(root).rglob('*'))
            if p.is_file() and not p.name.endswith('.tmp')}
