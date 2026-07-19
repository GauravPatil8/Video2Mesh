from pathlib import Path
import subprocess

def clone_repo(repo_url: str, destination: str | Path):
    destination = Path(destination)
    subprocess.run(
        ["git", "clone", repo_url, str(destination)],
        check=True,
    )