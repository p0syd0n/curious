"""
run_graphs.py
Usage: python run_graphs.py <log_file.json> <graphset_folder>

Generates all graphs from the graphset into output/run-<slug>/
"""

import json
import sys
import os
import importlib.util
import random

# ── slug generation ──────────────────────────────────────────────────────────

ADJECTIVES = [
    "amber", "bold", "calm", "dark", "eager", "faint", "grand", "heavy",
    "idle", "jolly", "kind", "lean", "mild", "noble", "oval", "pale",
    "quick", "rare", "sharp", "tall", "urban", "vast", "warm", "young",
    "zesty", "brisk", "crisp", "dusty", "fresh", "green",
]

NOUNS = [
    "apple", "birch", "cedar", "dune", "ember", "frost", "grove", "haze",
    "inlet", "junco", "knoll", "ledge", "marsh", "night", "orbit", "prism",
    "quill", "ridge", "stone", "thorn", "umbra", "vale", "wave", "xenon",
    "yucca", "zenith", "brook", "cliff", "delta", "flint",
]

def make_slug():
    return f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}-{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}"

# ── main ─────────────────────────────────────────────────────────────────────

def load_data(path):
    with open(path) as f:
        return json.load(f)

def load_graphset(folder):
    """Return list of (name, module) for every .py file in the graphset folder."""
    scripts = sorted(
        f for f in os.listdir(folder)
        if f.endswith(".py") and not f.startswith("_")
    )
    modules = []
    for filename in scripts:
        full_path = os.path.join(folder, filename)
        spec = importlib.util.spec_from_file_location(filename[:-3], full_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        modules.append((filename[:-3], mod))
    return modules

def run(log_path="log.json", graphset_folder="analysis/graphset"):
    data = load_data(log_path)
    slug = make_slug()
    out_dir = os.path.join("output", f"run-{slug}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Log:      {log_path}")
    print(f"Graphset: {graphset_folder}")
    print(f"Output:   {out_dir}")
    print()

    modules = load_graphset(graphset_folder)

    for name, mod in modules:
        save_path = os.path.join(out_dir, f"{name}.png")
        try:
            mod.plot(data, save=save_path)
            print(f"  ✓ {name}.png")
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")

    print(f"\nDone → {out_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_graphs.py <log_file.json> <graphset_folder>")
        print("Running default")
        run()
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])