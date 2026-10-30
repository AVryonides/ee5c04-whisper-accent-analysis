
from pathlib import Path

import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio

DATA_DIR = Path("data")
OUTPUT_DIR = Path("results/audio_examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INDICES = [678, 239, 541, 59, 63]

# Load the same dataset and test split
# used during the Whisper experiments.
dataset = load_dataset(
    "naharte/japanese_english",
    split="test"
)

dataset = dataset.cast_column(
    "audio",
    Audio(sampling_rate=16000, num_channels=1)
)

predictions = pd.read_csv(
    DATA_DIR / "japanese_baseline_predictions.csv",
    keep_default_na=False
)

for index in INDICES:
    sample = dataset[index]

    dataset_reference = (
        sample["transcription"].strip().lower()
    )
    csv_reference = (
        predictions.iloc[index]["reference"]
        .strip().lower()
    )

    # Confirm the audio still corresponds
    # to the correct CSV recording.
    if dataset_reference != csv_reference:
        raise ValueError(
            f"Reference mismatch at index {index}:\n"
            f"Dataset: {dataset_reference}\n"
            f"CSV: {csv_reference}"
        )

    audio = sample["audio"].get_all_samples()

    waveform = audio.data.squeeze().cpu().numpy()
    sample_rate = audio.sample_rate

    output_file = (
        OUTPUT_DIR / f"japanese_{index}.wav"
    )

    sf.write(
        output_file,
        waveform,
        sample_rate
    )

    print(f"\nIndex: {index}")
    print(f"Reference: {csv_reference}")
    print(f"Audio saved: {output_file}")

print("\nFinished exporting selected recordings.")
