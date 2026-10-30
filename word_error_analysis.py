
from pathlib import Path
from collections import Counter

import pandas as pd
import jiwer

DATA_DIR = Path("data")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

ACCENTS = ["canadian", "indian", "chinese", "japanese", "korean"]

all_results = []

for accent in ACCENTS:
    for model, suffix in [
        ("Baseline", "baseline_predictions"),
        ("Fine-tuned", "finetuned_decoding_predictions"),
    ]:
        path = DATA_DIR / f"{accent}_{suffix}.csv"
        df = pd.read_csv(path, keep_default_na=False)

        substitutions = Counter()
        deletions = Counter()
        insertions = Counter()

        for ref, hyp in zip(
            df["normalized_reference"],
            df["normalized_hypothesis"]
        ):
            ref_words = ref.split()
            hyp_words = hyp.split()

            alignment = jiwer.process_words(ref, hyp)

            for chunk in alignment.alignments[0]:
                r = ref_words[chunk.ref_start_idx:chunk.ref_end_idx]
                h = hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx]

                if chunk.type == "substitute":
                    substitutions[(" ".join(r), " ".join(h))] += 1

                elif chunk.type == "delete":
                    deletions[" ".join(r)] += 1

                elif chunk.type == "insert":
                    insertions[" ".join(h)] += 1

        for (source, target), count in substitutions.most_common(20):
            all_results.append({
                "Accent": accent,
                "Model": model,
                "Error Type": "Substitution",
                "Reference": source,
                "Hypothesis": target,
                "Count": count,
            })

        for word, count in deletions.most_common(20):
            all_results.append({
                "Accent": accent,
                "Model": model,
                "Error Type": "Deletion",
                "Reference": word,
                "Hypothesis": "",
                "Count": count,
            })

        for word, count in insertions.most_common(20):
            all_results.append({
                "Accent": accent,
                "Model": model,
                "Error Type": "Insertion",
                "Reference": "",
                "Hypothesis": word,
                "Count": count,
            })

results_df = pd.DataFrame(all_results)
results_df.to_csv(
    OUTPUT_DIR / "frequent_word_errors.csv",
    index=False
)

for accent in ACCENTS:
    print(f"\n{'=' * 45}\n{accent.upper()}\n{'=' * 45}")

    subset = results_df[
        (results_df["Accent"] == accent)
        & (results_df["Model"] == "Baseline")
        & (results_df["Error Type"] == "Substitution")
    ]

    print(subset.head(10).to_string(index=False))

print("\nSaved to results/frequent_word_errors.csv")


print("\n\nTOP SUBSTITUTIONS: BASELINE VS FINE-TUNED")

for accent in ACCENTS:
    print(f"\n{'=' * 50}")
    print(accent.upper())
    print(f"{'=' * 50}")

    for model in ["Baseline", "Fine-tuned"]:
        subset = results_df[
            (results_df["Accent"] == accent)
            & (results_df["Model"] == model)
            & (results_df["Error Type"] == "Substitution")
        ]

        print(f"\n{model}:")
        print(
            subset[
                ["Reference", "Hypothesis", "Count"]
            ].head(10).to_string(index=False)
        )
