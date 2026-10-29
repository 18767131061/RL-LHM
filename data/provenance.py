"""Content identities for generated prediction and evaluation artifacts."""
import hashlib
from pathlib import Path
import subprocess


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_revision():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                            text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None
