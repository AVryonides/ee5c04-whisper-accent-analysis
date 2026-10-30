
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

CANDIDATES = {
    "japanese": [678, 239, 541, 59, 63],
    "chinese": [455, 394],
}

for accent, indices in CANDIDATES.items():

    baseline = pd.read_csv(
        DATA_DIR / f"{accent}_baseline_predictions.csv",
        keep_default_na=False
    )

    finetuned = pd.read_csv(
        DATA_DIR / f"{accent}_finetuned_decoding_predictions.csv",
        keep_default_na=False
    )

    assert baseline["reference"].equals(
        finetuned["reference"]
    )

    for index in indices:

        before = baseline.iloc[index]
        after = finetuned.iloc[index]

        print("\n" + "=" * 70)
        print(f"ACCENT: {accent.upper()} | INDEX: {index}")
        print("=" * 70)

        print(f"REFERENCE:  {before['reference']}")
        print(f"BASELINE:   {before['hypothesis']}")
        print(f"FINE-TUNED: {after['hypothesis']}")

        print("\nNORMALISED:")
        print(f"REFERENCE:  {before['normalized_reference']}")
        print(f"BASELINE:   {before['normalized_hypothesis']}")
        print(f"FINE-TUNED: {after['normalized_hypothesis']}")
