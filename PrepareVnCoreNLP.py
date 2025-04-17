import os
import subprocess

root_dir = os.path.dirname(os.path.abspath(__file__))
vncorenlp_dir = os.path.join(root_dir, "vnCoreNLP")

vncorenlp_repo = "https://github.com/vncorenlp/VnCoreNLP.git"

# Clone the repo if the folder doesn't already exist
if not os.path.exists(vncorenlp_dir):
    print("Cloning VNCoreNLP repository...")
    subprocess.run(["git", "clone", vncorenlp_repo, vncorenlp_dir])
else:
    print("VNCORENLP folder already exists. Skipping clone.")

try:
    from py_vncorenlp import download_model
    download_model(save_dir=vncorenlp_dir)
except ImportError:
    print("py_vncorenlp not installed, skipping model download.")

print(f"VNCoreNLP is ready.\n Copy below directory into VNCORENLP_DIR in .env\n vnCoreNLP : {vncorenlp_dir}  ")
