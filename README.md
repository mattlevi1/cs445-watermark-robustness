# How Robust Are Watermarks on AI-Generated Text?

CS 445 (Machine Learning) semester project at James Madison University, by Prince Ezeonyim and Matthew Levi.

## The project

Since August 2026, the EU AI Act has required AI providers to mark AI-generated content in a machine-detectable way. Major providers now embed invisible statistical watermarks in the text their models write. This project asks:

> **How robust are generation-time text watermarks to everyday, meaning-preserving edits, and how does that robustness depend on edit strength, text length, and watermark strength?**

We compare two watermarking schemes:

- **Green-list watermark** (Kirchenbauer et al., 2023). At each step, a secret key and the previous token split the vocabulary into "green" and "red" tokens, and the model is nudged toward green ones. We implement this ourselves in PyTorch.
- **SynthID-Text** (Dathathri et al., 2024). Google DeepMind's production watermark, which uses tournament sampling. We use the open-source implementation in Hugging Face Transformers.

We will generate watermarked text with a small open model (OPT-1.3B), attack it with meaning-preserving edits, and measure how often the watermark is still detected:

- synonym substitution
- paraphrasing
- back-translation
- splicing into human-written text

The main metric is the true-positive rate at a 1% false-positive rate.

## Data

We use the **RealNewsLike** subset of **C4** (Raffel et al., 2020), the same data used by Kirchenbauer et al. (2023). Each news article is split into:

- a **prompt**: the opening of the article, which the model continues, and
- a **human continuation**: the next 200 tokens of the real article, which serve as unwatermarked human-written text.

The pairs are split into a **calibration set** (20%), used to choose settings and set detection thresholds, and a **test set** (80%), used for the final results. No model is trained, so there is no training set.

## How to run

You need Python 3.9 or newer.

```bash
git clone https://github.com/mattlevi1/cs445-watermark-robustness.git
cd cs445-watermark-robustness
pip install -r requirements.txt
python explore_data.py
```

The first run downloads the OPT-1.3B tokenizer (a few MB). It streams 1,000 articles from C4 RealNewsLike, so there is no need to download the full 15 GB dataset. It takes about a minute on a normal internet connection.

### What it does

`explore_data.py`:

1. Streams articles from C4 RealNewsLike.
2. Prints article-length statistics in words and in OPT tokens, plus the most common news domains.
3. Counts how many articles are long enough to give a prompt plus a 200-token continuation.
4. Builds prompt/continuation pairs and writes the calibration and test splits to `data/calibration.jsonl` and `data/test.jsonl`.

### Useful options

| Option | Default | Meaning |
|---|---|---|
| `--n` | 1000 | Number of articles to load |
| `--continuation-tokens` | 200 | Length of the human continuation |
| `--min-prompt-tokens` | 50 | Minimum prompt length for an article to be used |
| `--tokenizer` | `facebook/opt-1.3b` | Tokenizer for counting tokens; `none` counts words instead |
| `--calibration-fraction` | 0.2 | Fraction of pairs held out for calibration |
| `--seed` | 445 | Random seed for the split |
| `--local-file` | (none) | Read articles from a local JSONL file instead of downloading |

For a quick test that needs no model download:

```bash
python explore_data.py --n 200 --tokenizer none
```

### Running on Google Colab

```python
!git clone https://github.com/mattlevi1/cs445-watermark-robustness.git
%cd cs445-watermark-robustness
!pip install -q -r requirements.txt
!python explore_data.py
```

## Status

- [x] Data loading, summary statistics, and calibration/test split
- [ ] Green-list watermark (generation and z-score detection)
- [ ] SynthID-Text generation and detection
- [ ] Attacks: synonym substitution, paraphrasing, back-translation, splicing
- [ ] Evaluation: detection vs. attack strength, text length, and watermark strength

## References

Dathathri, S., See, A., Ghaisas, S., Huang, P.-S., . . . Kohli, P. (2024). Scalable watermarking for identifying large language model outputs. *Nature, 634*(8035), 818–823. https://doi.org/10.1038/s41586-024-08025-4

Kirchenbauer, J., Geiping, J., Wen, Y., Katz, J., Miers, I., & Goldstein, T. (2023). A watermark for large language models. In *Proceedings of the 40th International Conference on Machine Learning* (Vol. 202, pp. 17061–17084). PMLR.

Kirchenbauer, J., Geiping, J., Wen, Y., Shu, M., Saifullah, K., Kong, K., Fernando, K., Saha, A., Goldblum, M., & Goldstein, T. (2024). On the reliability of watermarks for large language models. In *The Twelfth International Conference on Learning Representations (ICLR 2024)*.

Krishna, K., Song, Y., Karpinska, M., Wieting, J., & Iyyer, M. (2023). Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense. In *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*.

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research, 21*(140), 1–67.

Zhao, X., Ananth, P., Li, L., & Wang, Y.-X. (2024). Provable robust watermarking for AI-generated text. In *The Twelfth International Conference on Learning Representations (ICLR 2024)*.
