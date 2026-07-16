"""
Build the complete Kaggle notebook JSON from our cell definitions.
"""
import json

# Read the cells file
with open('.openclaw/tmp/user-files/kaggle_11h_cells.py', 'r') as f:
    content = f.read()

# Split into cells by the CELL markers
import re
cell_markers = re.findall(r'# ═+\n# CELL \d+[a-z]?:.*?\n# ═+\n', content)

cells = []
# First split the content by cell markers
parts = re.split(r'# ═+\n# CELL \d+[a-z]?:.*?\n# ═+\n', content)

# The first part is the module docstring - make it a markdown cell
if parts[0].strip():
    docstring = parts[0].strip()
    # Remove triple quotes
    docstring = docstring.strip('"""').strip()
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"# Materials Futures Research Sprint\n{docstring}"]
    })

# Process each cell
for i, marker in enumerate(cell_markers):
    # Extract cell info from marker
    cell_match = re.search(r'CELL (\d+[a-z]?):\s*(.*)', marker)
    if not cell_match:
        continue
    cell_id = cell_match.group(1)
    cell_title = cell_match.group(2).strip()

    # Get cell content
    if i + 1 < len(parts):
        cell_content = parts[i + 1].strip()
    else:
        continue

    if not cell_content:
        continue

    # Determine cell type
    # Check if it's a markdown header comment
    lines = cell_content.split('\n')
    if lines[0].startswith('#') and not lines[0].startswith('#!') and len(lines) < 5:
        cell_type = "markdown"
        # Convert Python comments to markdown
        md_lines = []
        for line in lines:
            if line.startswith('# '):
                md_lines.append(line[2:])
            elif line.startswith('#'):
                md_lines.append(line[1:])
            else:
                md_lines.append(line)
        cell_content = '\n'.join(md_lines)
    else:
        cell_type = "code"

    cells.append({
        "cell_type": cell_type,
        "metadata": {},
        "source": cell_content.split('\n') if cell_type == "markdown" else [cell_content]
    })

# Build notebook
notebook = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        },
        "accelerator": "TPU",
        "gpu": {
            "driver_version": "535.161.08",
            "count": 1
        },
        "tpu": {
            "version": "v5e-8"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4,
    "cells": cells
}

# Write notebook
with open('.openclaw/tmp/user-files/materials-futures-sprint.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Created notebook with {len(cells)} cells")
for i, cell in enumerate(cells):
    src = ''.join(cell['source'])[:60].replace('\n', ' ')
    print(f"  [{i}] {cell['cell_type']}: {src}")
