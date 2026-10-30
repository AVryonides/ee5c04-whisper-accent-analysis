
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import jiwer

DATA_DIR = Path("data")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

ACCENTS = ["chinese", "japanese", "korean"]

# Common normalisation/contraction fragments.
# These are flagged for manual review, not deleted
# from our original WER calculations.
FRAGMENTS = {"s", "t", "m", "d", "ve", "re", "ll"}

all_rows = []

for accent in ACCENTS:

    model_counts = {}
    model_examples = {}

    for model_name, suffix in [
        ("Baseline", "baseline_predictions"),
        ("Fine-tuned", "finetuned_decoding_predictions")
    ]:

        df = pd.read_csv(
            DATA_DIR / f"{accent}_{suffix}.csv",
            keep_default_na=False
        )

        counts = Counter()
        examples = defaultdict(list)

        for index, row in df.iterrows():

            reference = row["normalized_reference"]
            hypothesis = row["normalized_hypothesis"]

            alignment = jiwer.process_words(
                reference, hypothesis
            )

            ref_words = reference.split()
            hyp_words = hypothesis.split()

            for chunk in alignment.alignments[0]:

                if chunk.type != "substitute":
                    continue

                ref_span = ref_words[
                    chunk.ref_start_idx:chunk.ref_end_idx
                ]

                hyp_span = hyp_words[
                    chunk.hyp_start_idx:chunk.hyp_end_idx
                ]

                # Restrict this analysis to 1-to-1
                # aligned substitutions.
                if len(ref_span) != 1 or len(hyp_span) != 1:
                    continue

                pair = (ref_span[0], hyp_span[0])

                counts[pair] += 1

                examples[pair].append({
                    "recording_index": index,
                    "reference": row["reference"],
                    "hypothesis": row["hypothesis"]
                })

        model_counts[model_name] = counts
        model_examples[model_name] = examples

    # Include every substitution observed in either model
    all_pairs = (
        set(model_counts["Baseline"])
        | set(model_counts["Fine-tuned"])
    )

    for ref_word, hyp_word in all_pairs:

        baseline_count = model_counts["Baseline"][
            (ref_word, hyp_word)
        ]

        finetuned_count = model_counts["Fine-tuned"][
            (ref_word, hyp_word)
        ]

        # Flag likely normalisation fragments
        needs_review = (
            ref_word in FRAGMENTS
            or hyp_word in FRAGMENTS
        )

        baseline_examples = model_examples["Baseline"].get(
            (ref_word, hyp_word), []
        )

        first_example = (
            baseline_examples[0]
            if baseline_examples else None
        )

        all_rows.append({
            "Accent": accent.capitalize(),
            "Reference": ref_word,
            "Hypothesis": hyp_word,
            "Baseline Count": baseline_count,
            "Fine-tuned Count": finetuned_count,
            "Reduction": baseline_count - finetuned_count,
            "Review Normalisation": needs_review,
            "Example Index": (
                first_example["recording_index"]
                if first_example else ""
            ),
            "Example Reference": (
                first_example["reference"]
                if first_example else ""
            ),
            "Example Hypothesis": (
                first_example["hypothesis"]
                if first_example else ""
            )
        })

results = pd.DataFrame(all_rows)

results = results.sort_values(
    ["Accent", "Baseline Count"],
    ascending=[True, False]
)

output_path = OUTPUT_DIR / "systematic_substitutions.csv"
results.to_csv(output_path, index=False)

# Display the top candidates, excluding
# obvious contraction fragments from this
# exploratory ranking only.
for accent in ACCENTS:

    subset = results[
        (results["Accent"] == accent.capitalize())
        & (~results["Review Normalisation"])
        & (results["Baseline Count"] >= 2)
    ]

    print(f"\n{'=' * 55}")
    print(f"{accent.upper()} — RECURRING SUBSTITUTIONS")
    print(f"{'=' * 55}")

    print(
        subset[
            [
                "Reference",
                "Hypothesis",
                "Baseline Count",
                "Fine-tuned Count",
                "Example Index"
            ]
        ].head(20).to_string(index=False)
    )

print(f"\nSaved: {output_path}")
