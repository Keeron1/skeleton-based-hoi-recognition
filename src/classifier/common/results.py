import json
from pathlib import Path

# Per-model results bundle written at the end of each training notebook.
# comparison.ipynb reads these to produce the cross-model figures.

def save(path, model_name, action_names, fold_reports, fold_cms):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Confusion matrices come in as numpy arrays. Serialize as nested lists.
    cms = [cm.tolist() if hasattr(cm, "tolist") else cm for cm in fold_cms]

    payload = {
        "model": model_name,
        "action_names": list(action_names),
        "fold_reports": fold_reports,
        "fold_cms": cms,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

def load(path):
    with open(path) as f:
        return json.load(f)
