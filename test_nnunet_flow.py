import sys
import subprocess

env_name = "nonewnet_v3"
dataset_num = 200
# Build the command using `conda run` (conda>=4.6) or shell activation
cmd = [
    "conda", "run", "-n", env_name,
    "nnUNetv2_plan_and_preprocess",
    "-d", f'{dataset_num}',
    "--verify_dataset_integrity"
]
# Optionally add logging / capture
try:
    cp = subprocess.run(cmd, cwd="stylegan-v", text=True, capture_output=True, check=True)
except subprocess.CalledProcessError as e:
    # Print what you would have seen, then re-raise
    if e.stdout: print(e.stdout, end="")
    if e.stderr: print(e.stderr, end="", file=sys.stderr)
    raise

# Print everything you would have seen
if cp.stdout: print(cp.stdout, end="")
if cp.stderr: print(cp.stderr, end="", file=sys.stderr)