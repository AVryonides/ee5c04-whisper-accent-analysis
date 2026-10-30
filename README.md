# EE5C04 — Whisper Tiny Foreign-Accented English

Supporting materials for my EE5C04 Assignment Part 1 report.

- `notebooks/` — modified Colab notebooks for fine-tuning and decoding.
- `data/` — baseline and fine-tuned prediction CSVs for Canadian, Indian, Chinese, Japanese and Korean English.
- Python scripts — local error analysis and verification used in the report.
- `results/` — summary CSVs from the analysis.

The notebooks have saved outputs cleared and private Google Drive paths replaced. The report contains the methodology, results and discussion.

To run the analysis, install the packages in `requirements.txt` and run the scripts from the repository root:

```bash
pip install -r requirements.txt
python error_analysis.py
python verify_canadian_wer.py
```

## Acknowledgements

The notebooks build on the supplied assignment material and [Sanchit Gandhi’s Whisper fine-tuning tutorial](https://huggingface.co/blog/fine-tune-whisper). The analysis uses [Whisper](https://github.com/openai/whisper), [JiWER](https://jitsi.github.io/jiwer/) and [CMUdict](https://github.com/cmusphinx/cmudict).

Dataset sources: [Canadian](https://huggingface.co/datasets/naharte/canadian_english), [Indian](https://huggingface.co/datasets/naharte/indian_english), [Chinese](https://huggingface.co/datasets/naharte/chinese_english), [Japanese](https://huggingface.co/datasets/naharte/japanese_english) and [Korean](https://huggingface.co/datasets/naharte/korean_english). No additional licence is granted for the source data.
