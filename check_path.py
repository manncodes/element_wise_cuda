import torch
import os
print(os.path.dirname(torch.__file__) + "/include")

print('-'*50)

from pathlib import Path

torch_path = Path(torch.__file__).parent
print(f"Torch path: {torch_path}")

# Look for all.h
all_h_files = list(torch_path.glob("**/*all.h"))
print("all.h files found:")
for f in all_h_files:
    print(f)