"""
This is a workaround to avoid auditwheel adding
casm libraries from other wheels... just rename the wheel

Expects dist/<vers>_raw to contain wheels.

Then:

    python label_wheels.py <vers>

Result: dist/<vers> contains renamed wheels
"""

import os
import shutil
import sys

print(sys.argv)
if len(sys.argv) != 2:
    print("Expected: label_wheels.py <vers>")
    exit(1)

vers = sys.argv[1]
raw_dir = f"dist/{vers}_raw"
if not os.path.exists(raw_dir):
    print(f"Does not exist: {raw_dir}")
    exit(1)

processed_dir = f"dist/{vers}"
if os.path.exists(processed_dir):
    print(f"Error, directory already exists: {processed_dir}")
    exit(1)

os.mkdir(processed_dir)
for file in os.listdir(raw_dir):
    processed_file = file.replace("linux", "manylinux_2_28")
    shutil.copyfile(
        os.path.join(raw_dir, file), os.path.join(processed_dir, processed_file)
    )
