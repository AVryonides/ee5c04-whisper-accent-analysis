
from pathlib import Path

import pandas as pd
import jiwer

csv_path = Path("data/canadian_baseline_predictions.csv")

df = pd.read_csv(csv_path, keep_default_na=False)

references = df["reference"].astype(str).tolist()
hypotheses = df["hypothesis"].astype(str).tolist()

normalized_references = (
    df["normalized_reference"].astype(str).tolist()
)
normalized_hypotheses = (
    df["normalized_hypothesis"].astype(str).tolist()
)

raw_wer = jiwer.wer(references, hypotheses) * 100
normalized_wer = jiwer.wer(
    normalized_references,
    normalized_hypotheses
) * 100

print("=" * 45)
print("CANADIAN BASELINE WER VERIFICATION")
print("=" * 45)

print(f"Number of recordings: {len(df)}")
print(f"Raw WER:             {raw_wer:.2f}%")
print(f"Normalized WER:      {normalized_wer:.2f}%")

assert len(df) == 407, "Unexpected test-set size"

print("\nVerification completed.")
