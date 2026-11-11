import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


files = ["data/raw/iris.csv"]
manifest = {
    "files": [
        {
            "path": str(p),
            "sha256": sha256(Path(p)),
            "size_bytes": Path(p).stat().st_size,
        }
        for p in files
    ]
}

out = Path("data/hash_manifest.json")
out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print(json.dumps(manifest, indent=2, ensure_ascii=False))
