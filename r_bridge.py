"""Python -> R bridge.

app.py calls predict(); this module runs R/03_predict.R with Rscript and
returns the predictions. It also reads the small files that 02_train.R saves
in models/ (dropdown choices and evaluation metrics).
"""
import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PREDICT_SCRIPT = ROOT / "R" / "03_predict.R"
MODEL_DIR = ROOT / "models"


def models_ready() -> bool:
    """True once 02_train.R has been run."""
    return (
        (MODEL_DIR / "views_model.rds").exists()
        and (MODEL_DIR / "comments_model.rds").exists()
        and (MODEL_DIR / "choices.json").exists()
    )


def load_choices() -> dict:
    """Dropdown options, e.g. {"category": ["Cooking", "Education", ...]}."""
    return json.loads((MODEL_DIR / "choices.json").read_text(encoding="utf-8"))


def load_metrics():
    """Model evaluation table (MAE, RMSE, R2), or None if not available."""
    path = MODEL_DIR / "metrics.csv"
    return pd.read_csv(path) if path.exists() else None


def predict(user_data: dict) -> dict:
    """Send the inputs to R and return e.g. {"predicted_views": 18500, "predicted_comments": 145}.

    Raises RuntimeError with a readable message if R is missing or the
    prediction fails.
    """
    # Pass the JSON through a temp file: quoting JSON on the command line is
    # unreliable on Windows.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(user_data, f)
        input_path = f.name

    try:
        result = subprocess.run(
            ["Rscript", str(PREDICT_SCRIPT), input_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=ROOT,
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Rscript was not found. Install R and make sure it is on your PATH."
        )
    except subprocess.CalledProcessError as e:
        message = (e.stderr or "").strip() or "The R prediction script failed."
        raise RuntimeError(message)
    finally:
        Path(input_path).unlink(missing_ok=True)

    # R prints a single line of JSON; take the last non-empty line to be safe
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("R returned no output.")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        raise RuntimeError(f"Could not read R's output: {lines[-1]}")