
from pathlib import Path
from collections import Counter
import re

import pandas as pd
import jiwer
import cmudict

DATA_DIR = Path("data")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

ACCENTS = ["chinese", "japanese", "korean"]

MODELS = {
    "Baseline": "baseline_predictions",
    "Fine-tuned": "finetuned_decoding_predictions"
}

PRONUNCIATIONS = cmudict.dict()

VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY", "IH", "IY",
    "OW", "OY", "UH", "UW"
}


def remove_stress(phone):
    """Remove stress digits from CMU vowel symbols."""
    return re.sub(r"[012]$", "", phone)


def get_vowel_contrasts(word1, word2):
    """
    Find vowel contrasts where two dictionary
    pronunciations have identical phoneme
    sequences except for one vowel.
    """

    pronunciations1 = PRONUNCIATIONS.get(word1, [])
    pronunciations2 = PRONUNCIATIONS.get(word2, [])

    contrasts = set()

    for pronunciation1 in pronunciations1:
        p1 = [remove_stress(p) for p in pronunciation1]

        for pronunciation2 in pronunciations2:
            p2 = [remove_stress(p) for p in pronunciation2]

            if len(p1) != len(p2):
                continue

            differences = [
                (a, b)
                for a, b in zip(p1, p2)
                if a != b
            ]

            if len(differences) != 1:
                continue

            vowel1, vowel2 = differences[0]

            if vowel1 in VOWELS and vowel2 in VOWELS:
                contrasts.add((vowel1, vowel2))

    return contrasts


all_examples = []
summary_counter = Counter()
pair_counter = Counter()

for accent in ACCENTS:

    for model, suffix in MODELS.items():

        path = DATA_DIR / f"{accent}_{suffix}.csv"

        df = pd.read_csv(
            path, keep_default_na=False
        )

        for index, row in df.iterrows():

            reference = row["normalized_reference"]
            hypothesis = row["normalized_hypothesis"]

            ref_words = reference.split()
            hyp_words = hypothesis.split()

            alignment = jiwer.process_words(
                reference, hypothesis
            )

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

                ref_word = source[0]
                hyp_word = target[0]

                contrasts = get_vowel_contrasts(
                    ref_word, hyp_word
                )

                # Exclude ambiguous dictionary matches:
                # more than one possible vowel contrast.
                if len(contrasts) != 1:
                    continue

                vowel1, vowel2 = next(iter(contrasts))

                # Group IH->IY and IY->IH into
                # the same undirected contrast.
                contrast_name = " / ".join(
                    sorted([vowel1, vowel2])
                )

                summary_counter[
                    (accent, model, contrast_name)
                ] += 1

                pair_counter[
                    (
                        accent,
                        model,
                        contrast_name,
                        ref_word,
                        hyp_word
                    )
                ] += 1

                all_examples.append({
                    "Accent": accent.capitalize(),
                    "Model": model,
                    "Index": index,
                    "Contrast": contrast_name,
                    "Reference Vowel": vowel1,
                    "Hypothesis Vowel": vowel2,
                    "Reference Word": ref_word,
                    "Hypothesis Word": hyp_word,
                    "Reference Sentence": row["reference"],
                    "Hypothesis Sentence": row["hypothesis"],
                })

# -----------------------------------------
# Save detailed examples
# -----------------------------------------

examples_df = pd.DataFrame(
    all_examples,
    columns=[
        "Accent", "Model", "Index", "Contrast",
        "Reference Vowel", "Hypothesis Vowel",
        "Reference Word", "Hypothesis Word",
        "Reference Sentence", "Hypothesis Sentence"
    ]
)

examples_df.to_csv(
    OUTPUT_DIR / "cmudict_vowel_examples.csv",
    index=False
)

# -----------------------------------------
# Save frequency summary
# -----------------------------------------

summary_rows = []

for (accent, model, contrast), count in summary_counter.items():
    summary_rows.append({
        "Accent": accent.capitalize(),
        "Model": model,
        "Vowel Contrast": contrast,
        "Count": count
    })

summary_df = pd.DataFrame(
    summary_rows,
    columns=[
        "Accent", "Model",
        "Vowel Contrast", "Count"
    ]
)

summary_df = summary_df.sort_values(
    "Count", ascending=False
)

summary_df.to_csv(
    OUTPUT_DIR / "cmudict_vowel_summary.csv",
    index=False
)

# -----------------------------------------
# Print results
# -----------------------------------------

print("\nCMU DICTIONARY VOWEL ERROR ANALYSIS")

for accent in ACCENTS:

    print("\n" + "=" * 55)
    print(accent.upper())
    print("=" * 55)

    for model in MODELS:

        print(f"\n{model} — MOST FREQUENT CONTRASTS")

        subset = summary_df[
            (summary_df["Accent"] == accent.capitalize())
            & (summary_df["Model"] == model)
        ]

        if subset.empty:
            print("No matching substitutions found.")
            continue

        print(
            subset[
                ["Vowel Contrast", "Count"]
            ].head(10).to_string(index=False)
        )

        # Show the word pairs responsible
        # for the most frequent contrast.
        top_contrast = subset.iloc[0]["Vowel Contrast"]

        candidates = [
            (ref, hyp, count)
            for (a, m, c, ref, hyp), count
            in pair_counter.items()
            if a == accent
            and m == model
            and c == top_contrast
        ]

        candidates.sort(
            key=lambda x: x[2],
            reverse=True
        )

        print(f"\nExamples for {top_contrast}:")
        for ref, hyp, count in candidates[:10]:
            print(f"  {ref} -> {hyp}: {count}")

print("\nSaved results:")
print("results/cmudict_vowel_examples.csv")
print("results/cmudict_vowel_summary.csv")
