
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import jiwer

DATA_DIR = Path("data")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

ACCENTS = [
    "canadian", "indian", "chinese",
    "japanese", "korean"
]

MODELS = {
    "Baseline": "baseline_predictions",
    "Fine-tuned": "finetuned_decoding_predictions"
}

# English /ɪ/ versus /iː/ candidate minimal pairs.
# Each pair is bidirectional for error detection.
MINIMAL_PAIRS = [
    ("ship", "sheep"),
    ("bit", "beat"),
    ("sit", "seat"),
    ("fill", "feel"),
    ("fit", "feet"),
    ("hit", "heat"),
    ("slip", "sleep"),
    ("chip", "cheap"),
    ("lick", "leak"),
    ("still", "steel"),
    ("pick", "peak"),
    ("sick", "seek"),
]

pair_lookup = {}

for word_a, word_b in MINIMAL_PAIRS:
    pair_lookup[(word_a, word_b)] = (word_a, word_b)
    pair_lookup[(word_b, word_a)] = (word_a, word_b)

rows = []
summary = []

for accent in ACCENTS:

    for model, suffix in MODELS.items():

        df = pd.read_csv(
            DATA_DIR / f"{accent}_{suffix}.csv",
            keep_default_na=False
        )

        counts = Counter()

        for index, row in df.iterrows():

            ref = row["normalized_reference"]
            hyp = row["normalized_hypothesis"]

            ref_words = ref.split()
            hyp_words = hyp.split()

            alignment = jiwer.process_words(ref, hyp)

            for chunk in alignment.alignments[0]:

                if chunk.type != "substitute":
                    continue

                source = ref_words[
                    chunk.ref_start_idx:chunk.ref_end_idx
                ]

                target = hyp_words[
                    chunk.hyp_start_idx:chunk.hyp_end_idx
                ]

                if len(source) != 1 or len(target) != 1:
                    continue

                reference_word = source[0]
                predicted_word = target[0]

                key = (reference_word, predicted_word)

                if key not in pair_lookup:
                    continue

                pair = pair_lookup[key]
                counts[(reference_word, predicted_word)] += 1

                rows.append({
                    "Accent": accent.capitalize(),
                    "Model": model,
                    "Index": index,
                    "Vowel Pair": f"{pair[0]} / {pair[1]}",
                    "Reference Word": reference_word,
                    "Predicted Word": predicted_word,
                    "Reference Sentence": row["reference"],
                    "Predicted Sentence": row["hypothesis"],
                })

        summary.append({
            "Accent": accent.capitalize(),
            "Model": model,
            "Matched Substitutions": sum(counts.values()),
            "Distinct Confusions": len(counts),
        })

# Save every matching example
columns = [
    "Accent", "Model", "Index", "Vowel Pair",
    "Reference Word", "Predicted Word",
    "Reference Sentence", "Predicted Sentence"
]

examples_df = pd.DataFrame(rows, columns=columns)

examples_df.to_csv(
    OUTPUT_DIR / "vowel_contrast_examples.csv",
    index=False
)

summary_df = pd.DataFrame(summary)
summary_df.to_csv(
    OUTPUT_DIR / "vowel_contrast_summary.csv",
    index=False
)

print("\nVOWEL CONTRAST: /ɪ/ VS /iː/\n")
print(summary_df.to_string(index=False))

print("\nMATCHED EXAMPLES:\n")

if examples_df.empty:
    print(
        "No exact minimal-pair substitutions found. "
        "This does not establish that the vowel "
        "contrast is absent from the recordings."
    )
else:
    print(
        examples_df[
            [
                "Accent", "Model", "Index",
                "Reference Word", "Predicted Word"
            ]
        ].to_string(index=False)
    )

print("\nSaved:")
print("results/vowel_contrast_examples.csv")
print("results/vowel_contrast_summary.csv")
