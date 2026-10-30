
from pathlib import Path

import pandas as pd
import jiwer

# -----------------------------------------
# Configuration
# -----------------------------------------

DATA_DIR = Path("data")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

ACCENTS = [
    "canadian",
    "indian",
    "chinese",
    "japanese",
    "korean",
]

EXPECTED_COUNTS = {
    "canadian": 407,
    "indian": 650,
    "chinese": 629,
    "japanese": 700,
    "korean": 450,
}

# -----------------------------------------
# Calculate error statistics
# -----------------------------------------

def calculate_error_profile(references, hypotheses):

    alignment = jiwer.process_words(
        references,
        hypotheses
    )

    substitutions = alignment.substitutions
    deletions = alignment.deletions
    insertions = alignment.insertions
    hits = alignment.hits

    total_words = substitutions + deletions + hits
    total_errors = substitutions + deletions + insertions

    calculated_wer = total_errors / total_words

    assert abs(calculated_wer - alignment.wer) < 1e-10

    return {
        "Reference Words": total_words,
        "Correct Words": hits,
        "Substitutions": substitutions,
        "Deletions": deletions,
        "Insertions": insertions,
        "Total Errors": total_errors,
        "WER (%)": round(calculated_wer * 100, 2),
        "Substitution Share (%)": round(
            substitutions / total_errors * 100, 2
        ) if total_errors else 0,
        "Deletion Share (%)": round(
            deletions / total_errors * 100, 2
        ) if total_errors else 0,
        "Insertion Share (%)": round(
            insertions / total_errors * 100, 2
        ) if total_errors else 0,
    }

# -----------------------------------------
# Analyse all five accents
# -----------------------------------------

results = []

for accent in ACCENTS:

    baseline_path = DATA_DIR / f"{accent}_baseline_predictions.csv"
    finetuned_path = (
        DATA_DIR / f"{accent}_finetuned_decoding_predictions.csv"
    )

    baseline_df = pd.read_csv(
        baseline_path, keep_default_na=False
    )
    finetuned_df = pd.read_csv(
        finetuned_path, keep_default_na=False
    )

    assert len(baseline_df) == EXPECTED_COUNTS[accent]
    assert len(finetuned_df) == EXPECTED_COUNTS[accent]

    # Confirm both models were evaluated on
    # exactly the same reference sentences.
    assert baseline_df["normalized_reference"].equals(
        finetuned_df["normalized_reference"]
    ), f"Reference mismatch for {accent}"

    for model_name, df in [
        ("Baseline", baseline_df),
        ("Fine-tuned", finetuned_df),
    ]:

        references = df["normalized_reference"].tolist()
        hypotheses = df["normalized_hypothesis"].tolist()

        profile = calculate_error_profile(
            references, hypotheses
        )

        results.append({
            "Accent": accent.capitalize(),
            "Model": model_name,
            "Recordings": len(df),
            **profile,
        })

# Display and save results
results_df = pd.DataFrame(results)

print("\nERROR PROFILE ANALYSIS\n")

print(
    results_df[
        [
            "Accent",
            "Model",
            "Reference Words",
            "Substitutions",
            "Deletions",
            "Insertions",
            "WER (%)",
        ]
    ].to_string(index=False)
)

output_path = OUTPUT_DIR / "error_profiles.csv"

results_df.to_csv(output_path, index=False)

print(f"\nResults saved to: {output_path}")
