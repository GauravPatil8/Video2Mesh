from pipeline.utils.general import clone_repo
import subprocess

clone_repo("https://github.com/Anttwo/SuGaR.git", "./libs/")

subprocess.call([
    "python",
    "./libs/SuGaR/install.py"
])

subprocess.call([
    "conda",
    "run",
    "-n",
    "sugar",
    "pip",
    "install",
    "-r",
    "requirements.txt"
])
