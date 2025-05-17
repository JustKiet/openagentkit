import subprocess
from pathlib import Path

req_dir = Path("requirements")
req_files = [
    "requirements-core",
    "requirements-openai",
    "requirements-mcp",
    "requirements-milvus",
    "requirements-voyageai",
    "requirements-dev",
    "requirements-all",
]

for name in req_files:
    infile = req_dir / f"{name}.in"
    outfile = req_dir / f"{name}.txt"
    print(f"Compiling {infile} -> {outfile}")
    subprocess.run([
        "pip-compile", str(infile), "--output-file", str(outfile)
    ], check=True)
