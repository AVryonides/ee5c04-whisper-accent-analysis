
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

CANDIDATES = {
    "chinese": [154, 394],
    "japanese": [239],
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

        print("\n" + "=" * 65)
        print(f"{accent.upper()} | RECORDING {index}")
        print("=" * 65)

        print("REFERENCE: ", before["reference"])
        print("BASELINE:  ", before["hypothesis"])
        print("FINE-TUNED:", after["hypothesis"])

        print("\nNORMALISED:")
        print("REFERENCE: ", before["normalized_reference"])
        print("BASELINE:  ", before["normalized_hypothesis"])
        print("FINE-TUNED:", after["normalized_hypothesis"])
