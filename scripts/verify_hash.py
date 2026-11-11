import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


manifest = json.loads(Path("data/hash_manifest.json").read_text())
ok = True
for entry in manifest["files"]:
    actual = sha256(Path(entry["path"]))
    status = "OK" if actual == entry["sha256"] else "MISMATCH"
    if actual != entry["sha256"]:
        ok = False
    print(f"{status}: {entry['path']}")

print("VERDICT:", "совпадает" if ok else "не совпадает")
