
from pathlib import Path
import pandas as pd
import cmudict

RESULTS_DIR = Path("results")
DATA_DIR = Path("data")
dictionary = cmudict.dict()

examples = pd.read_csv(
    RESULTS_DIR / "cmudict_vowel_examples.csv"
)

selected = examples[
    (examples["Accent"] == "Chinese")
    & (examples["Model"] == "Fine-tuned")
    & (examples["Contrast"] == "IH / IY")
]

baseline = pd.read_csv(
    DATA_DIR / "chinese_baseline_predictions.csv",
    keep_default_na=False
)

finetuned = pd.read_csv(
    DATA_DIR / "chinese_finetuned_decoding_predictions.csv",
    keep_default_na=False
)

print("\nCHINESE ENGLISH: IH / IY EXAMPLES")

for _, row in selected.iterrows():
    index = int(row["Index"])
    ref_word = row["Reference Word"]
    hyp_word = row["Hypothesis Word"]

    print("\n" + "=" * 65)
    print(f"RECORDING {index}")
    print(f"REFERENCE WORD: {ref_word}")
    print(f"PREDICTED WORD: {hyp_word}")
    print(f"CMU REF: {dictionary.get(ref_word)}")
    print(f"CMU HYP: {dictionary.get(hyp_word)}")

    print("REFERENCE: ", baseline.iloc[index]["reference"])
    print("BASELINE:  ", baseline.iloc[index]["hypothesis"])
    print("FINE-TUNED:", finetuned.iloc[index]["hypothesis"])

print("\nTotal matching examples:", len(selected))
