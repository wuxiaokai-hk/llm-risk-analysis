"""Clone the authors' data repo (Shah, Paturi & Chava 2023, CC BY-NC 4.0) into data/fomc-hawkish-dovish."""
import os
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dest = os.path.join(HERE, "data", "fomc-hawkish-dovish")
if os.path.isdir(dest):
    print("already there:", dest)
else:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/gtfintechlab/fomc-hawkish-dovish", dest],
                   check=True)
    print("data in", dest)
