import json
import os
import sys
from pathlib import Path

def fix_notebook(path):
    print(f"Processing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    model_ids = set()

    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            data = output.get("data", {})

            # widget-view format
            if "application/vnd.jupyter.widget-view+json" in data:
                wid = data["application/vnd.jupyter.widget-view+json"].get("model_id")
                if wid:
                    model_ids.add(wid)

            # fallback: any raw model_id
            if "model_id" in output:
                model_ids.add(output["model_id"])

    widget_state = {
        wid: {
            "model_module": "@jupyter-widgets/base",
            "model_name": "WidgetModel",
            "state": {}
        }
        for wid in model_ids
    }

    nb.setdefault("metadata", {})
    nb["metadata"]["widgets"] = {
        "application/vnd.jupyter.widget-state+json": {
            "state": widget_state,
            "version_major": 2,
            "version_minor": 0
        }
    }

    fixed_path = str(Path(path).with_name(Path(path).stem + "_FIXED.ipynb"))
    with open(fixed_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"✔ Found {len(model_ids)} widgets.")
    print(f"✔ Saved: {fixed_path}\n")
    return fixed_path

# Main
if len(sys.argv) > 1:
    # Explicit notebook path
    fix_notebook(sys.argv[1])
else:
    # Fix all .ipynb files in repo
    for nb in Path(".").rglob("*.ipynb"):
        fix_notebook(nb)
