
from pathlib import Path

import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio

DATA_DIR = Path("data")
OUTPUT_DIR = Path("results/vowel_audio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = {
    "chinese": [154, 394],
    "japanese": [239],
}

for accent, indices in CANDIDATES.items():

    dataset = load_dataset(
        f"naharte/{accent}_english",
        split="test"
    )

    dataset = dataset.cast_column(
        "audio",
        Audio(sampling_rate=16000, num_channels=1)
    )

    predictions = pd.read_csv(
        DATA_DIR / f"{accent}_baseline_predictions.csv",
        keep_default_na=False
    )

    for index in indices:

        sample = dataset[index]

        dataset_reference = (
            sample["transcription"].strip().lower()
        )

        csv_reference = (
            predictions.iloc[index]["reference"]
            .strip().lower()
        )

        if dataset_reference != csv_reference:
            raise ValueError(
                f"Reference mismatch: {accent} {index}"
            )

        audio = sample["audio"].get_all_samples()

        waveform = audio.data.squeeze().cpu().numpy()
        sample_rate = audio.sample_rate

        output_file = (
            OUTPUT_DIR / f"{accent}_{index}.wav"
        )

        sf.write(output_file, waveform, sample_rate)

        print(f"\n{accent.upper()} | Index {index}")
        print(f"Reference: {csv_reference}")
        print(f"Saved: {output_file}")

print("\nAll selected recordings exported.")
