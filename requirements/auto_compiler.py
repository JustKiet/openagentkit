import os
import subprocess
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

BASE_DIR = './requirements'
INDEX_URL = "https://pypi.org/simple"
PIP_COMPILE_BASE_CMD = [sys.executable, "-m", "piptools", "compile"]
MAX_WORKERS = 4

def find_requirements_in_files(base_dir: str) -> List[str]:
    """Find all .in files in BASE_DIR."""
    requirements_files: List[str] = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".in"):
                requirements_files.append(os.path.join(root, file))
    return requirements_files

def compile_requirements_core(core_file: str) -> bool:
    output_file = core_file.replace(".in", ".txt")
    cmd = PIP_COMPILE_BASE_CMD + [
        "--output-file", output_file,
        "--index-url", INDEX_URL,
        core_file
    ]
    logger.info(f"Compiling core requirements: {core_file} → {output_file}")
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Successfully compiled {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to compile core requirements: {e}")
        return False

def compile_extension_requirements(extension_file: str, core_txt: str) -> bool:
    output_file = extension_file.replace(".in", ".txt")
    cmd = PIP_COMPILE_BASE_CMD + [
        "--output-file", output_file,
        "--index-url", INDEX_URL,
        core_txt,
        extension_file
    ]
    logger.info(f"Compiling extension: {extension_file} (with {core_txt}) → {output_file}")
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Successfully compiled {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to compile {extension_file}: {e}")
        return False

def main() -> None:
    requirements_files = find_requirements_in_files(BASE_DIR)
    core_file = next((f for f in requirements_files if os.path.basename(f) == "requirements-core.in"), None)

    if not core_file:
        logger.error("requirements-core.in not found. Cannot proceed.")
        return

    if not compile_requirements_core(core_file):
        logger.error("Core requirements failed to compile. Aborting.")
        return

    core_txt = core_file.replace(".in", ".txt")
    extensions = [f for f in requirements_files if f != core_file]

    logger.info(f"Found {len(extensions)} extension requirements.in files.")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {
            executor.submit(compile_extension_requirements, file, core_txt): file for file in extensions
        }

        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result = future.result()
                if not result:
                    logger.warning(f"Compilation failed for {file}")
            except Exception as exc:
                logger.error(f"Unexpected error compiling {file}: {exc}")

if __name__ == "__main__":
    main()
