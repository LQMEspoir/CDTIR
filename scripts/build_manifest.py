from pathlib import Path
import sys
import os

ROOT=Path(os.environ.get("CDTIR_ROOT",Path.cwd())).expanduser().resolve()
CODE_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE_ROOT/"src"))
from tourism_ie.paper import build_manifest

if __name__=="__main__":
    manifest=build_manifest(ROOT)
    print(f"MANIFEST.json: {len(manifest['files'])} files")
