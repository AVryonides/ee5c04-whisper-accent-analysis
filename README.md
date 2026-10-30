# EE5C04 — Whisper Tiny foreign-accented English

Assignment Part 1 supporting materials by Alexandros Vryonides. This repository contains the original modified Colab notebooks (sanitised for publication), 11 original local Python analysis scripts, aggregate result CSVs, and an inventory of the ten original prediction CSVs.

## Publication status

**Per-recording prediction CSVs and audio are not published.** All five upstream dataset cards lacked an explicit redistribution licence when checked on 30 October 2026. The CSVs contain dataset reference transcripts. Their filenames, row counts, column schemas and SHA-256 checksums are recorded in `data/manifest.json`; see `data/README.md`. They can be analysed locally by someone authorised to access them. No model weights, credentials, private Drive directories, teaching handouts, or report PDF are included. This public repository therefore does not independently reproduce all results without the local CSVs.

## Method

Compare pretrained multilingual `openai/whisper-tiny` against separately accent-adapted models for Canadian, Indian, Chinese, Japanese and Korean English. The notebooks cover Hugging Face dataset loading, audio feature extraction, fine-tuning, decoding, normalisation and CSV export. Their saved configuration is Korean; see `notebooks/README.md` before adapting it to other accents.

The saved fine-tuning configuration uses learning rate 1e-5, 2,000 training steps, training batch size 2, gradient accumulation 1, warm-up 500 steps, and evaluation/saving every 1,000 steps. These are the settings visible in the saved notebook, not independently established settings for every historical run. The test split is also the trainer evaluation set used for best-checkpoint selection; this limits interpretation as a completely held-out final test. No models were retrained for this repository.

For analysis, use the saved raw and `BasicTextNormalizer`-normalised reference/hypothesis columns. Corpus WER = (substitutions + deletions + insertions) / reference words; it is not the mean of utterance WERs. `error_analysis.py` checks expected recording counts, identical ordered normalised references between conditions, and agreement between edit counts and JiWER WER. These checks do not establish speaker independence or absence of training/test leakage.

## Results

Normalised corpus WER, recomputed from the original local prediction CSVs during repository preparation:

| Accent | Test recordings | Baseline WER (%) | Fine-tuned WER (%) |
|---|---:|---:|---:|
| Canadian | 407 | 21.03 | 19.04 |
| Indian | 650 | 27.27 | 9.14 |
| Chinese | 629 | 21.81 | 12.63 |
| Japanese | 700 | 45.5 | 20.97 |
| Korean | 450 | 28.1 | 9.54 |

Canadian baseline raw WER was also recomputed: **41.04%** for 407 recordings; normalised WER **21.03%**. The earlier conversational 35.00% value is not used. Aggregate edit counts are in `results/error_profiles.csv`. Fine-tuning reduced WER for each accent in these saved runs. Dataset differences and checkpoint selection prevent attributing every difference exclusively to accent.

## Run the local analysis

From the repository root, with Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Place authorised original prediction CSVs in data/ (see manifest).
python error_analysis.py
python verify_canadian_wer.py
python word_error_analysis.py
python systematic_error_analysis.py
python vowel_contrast_analysis.py
python cmudict_vowel_analysis.py
```

`requirements.txt` provides compatible ranges, not a recovered lockfile for the original training environment. Notebook installations are separately specified in the original notebooks and include unpinned packages. Exact historical training reproduction is not guaranteed.

`inspect_error_analysis.py` and `inspect_vowel_examples.py` inspect selected recordings; `verify_cmudict_examples.py` checks generated dictionary examples. `inspect_japanese_audio.py` and `inspect_vowel_audio.py` download selected upstream audio for local listening and require the optional audio dependencies. Check upstream access and usage terms first. Generated transcript/example CSVs and audio should remain local pending licence clarification.

Word errors, systematic substitutions, manual listening candidates and CMUdict vowel contrasts support exploratory error analysis. Dictionary pronunciations are not measurements of a speaker’s realised pronunciation and cannot alone prove a phonetic cause. Only aggregate vowel summaries are included publicly.

## Verification and provenance

The 11 Python files are unchanged copies of the local analysis scripts. Notebook sanitisation is documented in `notebooks/README.md`. During preparation, `error_analysis.py` and `verify_canadian_wer.py` were executed against temporary copies of all ten local CSVs; the generated error profile matched the saved aggregate file exactly. Notebook training and audio downloads were not rerun. No transcript CSV is included in Git history.

## Sources and acknowledgements

- Radford et al. (2023), *Robust Speech Recognition via Large-Scale Weak Supervision*: https://arxiv.org/abs/2212.04356 . Whisper code: https://github.com/openai/whisper ; checkpoint: https://huggingface.co/openai/whisper-tiny .
- Sanchit Gandhi (2022), *Fine-Tune Whisper For Multilingual ASR with Transformers*: https://huggingface.co/blog/fine-tune-whisper . The supplied teaching notebook and its original attribution are retained in the modified notebook.
- JiWER documentation: https://jitsi.github.io/jiwer/ .
- Carnegie Mellon Pronouncing Dictionary: https://github.com/cmusphinx/cmudict . The dictionary is accessed through the `cmudict` package rather than redistributed here.
- Assignment dataset sources: [Canadian](https://huggingface.co/datasets/naharte/canadian_english), [Indian](https://huggingface.co/datasets/naharte/indian_english), [Chinese](https://huggingface.co/datasets/naharte/chinese_english), [Japanese](https://huggingface.co/datasets/naharte/japanese_english), [Korean](https://huggingface.co/datasets/naharte/korean_english).

No repository-wide open-source licence is asserted over third-party notebooks or datasets. Upstream material retains its original rights and terms.
