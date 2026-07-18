# Automated Layer Selection for Efficient Fine-Tuning of Medical Image Segmentation Models

> This repository began as a fork of **CSWin-UNet**, "Transformer UNet with cross-shaped windows for medical image segmentation" (Liu et al., *Information Fusion*, 2025) — see [Citation](#citation). Everything beyond the base architecture (`networks/`, `config.py`) was built out for this project: a robustness study, a from-scratch dataset-repair pipeline, and an automated, gradient-driven fine-tuning framework for continual learning across multiple segmentation tasks.

## Project summary

Fine-tuning large pre-trained segmentation models is expensive, and manually choosing which layers to update rarely generalizes. This project builds and evaluates an **automated layer-selection framework** that combines two complementary ideas from the literature:

- **Surgical Fine-Tuning** (Lee et al., 2022) — every epoch, a Relative Gradient Norm (RGN) pass measures how much each layer "wants" to move for the new task and turns that into a per-layer learning rate, so task-relevant layers adapt fast while the rest are effectively frozen.
- **TPGM — Trainable Projected Gradient Method** (Tian et al., 2023) — each layer is constrained to stay within a *learnable* radius of its pre-trained weights; the radius itself is optimized (bi-level optimization) against a held-out split, so the model learns automatically how much drift each layer can tolerate.

These are evaluated, individually and combined ("Hybrid"), against standard full fine-tuning on a **three-stage continual-learning protocol** using CSWin-UNet as the backbone:

```
Task 1: Synapse (8 abdominal organs)  →  Task 2: KiTS23 (kidney + tumor + cyst)  →  Task 3: LiTS17 (liver + tumor)
```

The central question: after sequentially fine-tuning on Task 2 and Task 3, how much of the Task 1 (and Task 2) performance survives? This is the classic **catastrophic forgetting** problem, and it is severe under naive full fine-tuning (see [Results](#results)).

A secondary thread of the project (the earlier phase of the work, chronicled in [`start.txt`](start.txt)) is a **robustness stress-test**: how much does CSWin-UNet's segmentation accuracy degrade under a simple Gaussian-blur domain shift, and can fine-tuning recover it?

## Background: CSWin-UNet

CSWin-UNet tackles the standard Transformer-for-segmentation trade-off: capture long-range context like a Transformer, without the quadratic cost of full self-attention. It does this with **cross-shaped window attention** (horizontal + vertical stripe attention computed in parallel, `LePEAttention`/`CSWinBlock` in [`networks/cswin_unet.py`](networks/cswin_unet.py)) inside a UNet-style encoder-decoder with skip connections, and uses **CARAFE** (content-aware feature reassembly) instead of plain transposed-conv/bilinear upsampling for sharper decoder outputs. The original paper reports 81.12% Dice on Synapse with only 23.57M parameters. See [References](#references) for the paper and its own upstream references (Swin-Unet, CSWin-Transformer).

## Repository structure

```
CSWIN/
├── networks/
│   ├── cswin_unet.py           # CSWinTransformer backbone+decoder, CARAFE upsampling
│   └── vision_transformer.py   # CSwinUnet wrapper (builds model from a yacs config, checkpoint loading)
├── config.py                   # yacs CfgNode schema for model/training hyperparameters
├── configs/                    # per-run yaml configs (all use the same tiny backbone; differ only in which checkpoint to warm-start from)
├── datasets/
│   ├── dataset_synapse.py      # shared PyTorch Dataset (2D .npz slices for train, .h5 volumes for test)
│   ├── create_lists.py         # regenerates train/test split files from files on disk
│   └── Abdomen/convert.py      # raw BTCV NIfTI → Synapse npz converter, 13→8 organ label remap (see "Dataset repair" below)
├── lists/                      # train/val/test split files per dataset (lists_Synapse, lists_Synapse_blurred, kits23)
├── train.py / trainer.py       # training entry point + SGD/poly-LR loop (CE + Dice loss)
├── test.py / utils.py          # volume-level inference, Dice + HD95 metrics, FLOPs/param profiling
├── finetune.py                 # plain full fine-tuning baseline
├── finetune_basic.py           # early surgical fine-tuning prototype (per-batch RGN, (lr, wd) grid search)
├── finetune_surgical_2.py      # refined Surgical Fine-Tuning (per-epoch RGN / eb-criterion layer weighting)
├── tpgm.py / tpgm_simple.py    # TPGM projection module + trainer (two implementations, see below)
├── finetune_tpgm.py            # TPGM fine-tuning, single task
├── finetune_tpgm_continual.py  # TPGM for continual learning (retains full output head across tasks, see below)
├── apply_blur_train.py / apply_blur_test.py       # Gaussian-blur domain-shift generator (Synapse → Synapse-Blurred)
├── visualize_blurs_train.py / visualize_blurs_test.py  # interactive before/after slice viewers
├── check_lables.py             # .npz inspection/debug tool (used to diagnose the label bug below)
├── Report.pdf                  # full thesis write-up (methodology, related work, complete results)
├── start.txt                   # running experiment log / narrative of what was tried and why
└── log.txt                     # raw per-run train/test console logs
```

Everything else at the top level (`pretrain*/`, `finetune*/`, `output_surgical/`, `test_log/`, `test_visualization/`, `segments/`, `pretrained_ckpt/`, `preprocessing_visualization_blurred/`) is **generated output** — checkpoints, TensorBoard logs, and qualitative visualizations from individual runs, not source code. `cswin/` and `env/` are local Python virtual environments, not vendored source.

## Setup

```bash
pip install -r requirements.txt
# a few imports used by the custom scripts aren't pinned in requirements.txt:
pip install loguru matplotlib thop nibabel
```

Pretrained ImageNet CSWin-Tiny weights go in `pretrained_ckpt/` (as in the original upstream repo).

## Dataset repair (Synapse)

The Synapse data distributed by the TransUnet/CSWin-UNet authors turned out to be broken — most provided `.npz` files had missing or degenerate labels, silently producing near-zero Dice on every fine-tuning attempt. `check_lables.py` was written to inspect raw `.npz` contents (keys, shapes, label histograms) and confirm the problem. The fix, [`datasets/Abdomen/convert.py`](datasets/Abdomen/convert.py), rebuilds the dataset directly from the raw BTCV/"Multi-Atlas Labeling Beyond the Cranial Vault" NIfTI files: it clips/normalizes CT intensities, and remaps the original 13 organ labels down to the 8 used throughout this project (spleen, right kidney, left kidney, gallbladder, liver, stomach, aorta, pancreas), dropping the rest to background.

## Blur robustness study

To probe how brittle the segmentation model is to a simple image-quality domain shift, `apply_blur_train.py`/`apply_blur_test.py` apply a Gaussian blur (σ≈1.0–1.5) slice-by-slice to produce **Synapse-Blurred**, keeping labels unchanged. A model pre-trained on clean Synapse (~0.795 mean Dice) collapsed to roughly 0.3–0.4 mean Dice when evaluated directly on the blurred volumes — motivating the fine-tuning/adaptation work that follows. After fine-tuning (once the label bug above was fixed), performance recovered to ~0.77–0.79 mean Dice on both clean and blurred data. Full per-class numbers are in [`start.txt`](start.txt) and [`log.txt`](log.txt).

## Fine-tuning methods

| Script | Method |
|---|---|
| `finetune.py` | Standard full fine-tuning (AdamW + cosine annealing), optional first-N-layer freezing |
| `finetune_surgical_2.py` | **Surgical FT**: every epoch, a few gradient probe batches compute either a Relative Gradient Norm (`‖grad‖ / ‖param‖`, normalized per epoch) or an "eb-criterion" score per parameter tensor, which sets that tensor's own AdamW learning rate — layers the new task needs most adapt fastest, the rest are effectively frozen |
| `tpgm.py` / `tpgm_simple.py` | **TPGM**: one learnable scalar constraint per weight tensor bounds how far it may drift (L2 or "mars"/per-row L1 norm) from its pre-trained anchor; constraints are themselves optimized (Adam) against a held-out split, then the final weights are projected back inside their learned radius |
| `finetune_tpgm.py` | TPGM applied to a single fine-tuning task |
| `finetune_tpgm_continual.py` | TPGM for **continual learning**: the model keeps its full previous-task output head (`--model_num_classes`), only the new task's channels (`--num_classes`) receive supervision on new data, while TPGM's projection toward the frozen prior-task anchor protects the unsupervised channels from drifting |

## Results

Dice similarity coefficient per organ/class, after the full **Synapse → KiTS23 → LiTS17** sequential fine-tuning run (HD95 and complete tables are in `Report.pdf`, Tables 1–4):

| Dataset | Class | Standard FT | Surgical FT | TPGM | Hybrid (Surgical+TPGM) |
|---|---|---|---|---|---|
| Synapse | Aorta | 0.745 | 0.844 | **0.856** | 0.842 |
| | Gallbladder | 0.000 | 0.607 | **0.633** | 0.603 |
| | Left Kidney | 0.610 | 0.786 | **0.807** | 0.777 |
| | Right Kidney | 0.385 | 0.713 | **0.718** | 0.697 |
| | Liver | 0.632 | **0.912** | 0.903 | 0.906 |
| | Pancreas | 0.000 | 0.607 | **0.664** | 0.584 |
| | Spleen | 0.135 | **0.912** | 0.908 | 0.898 |
| | Stomach | 0.340 | **0.817** | 0.815 | 0.812 |
| KiTS23 | Kidney | 0.614 | 0.620 | 0.701 | **0.719** |
| | Tumor | 0.551 | 0.549 | 0.555 | **0.586** |
| | Cyst | 0.351 | 0.403 | 0.401 | **0.410** |
| LiTS17 | Liver | 0.753 | 0.693 | **0.782** | 0.683 |
| | Tumor | 0.550 | 0.565 | **0.579** | 0.569 |

Independent (non-continual) upper-bound baselines, i.e. training from scratch on each dataset alone: KiTS23 kidney/tumor/cyst = 0.915 / 0.759 / 0.463 Dice; LiTS17 liver/tumor = 0.735 / 0.527 Dice.

**Takeaways:**
- **Standard full fine-tuning suffers severe catastrophic forgetting** — gallbladder and pancreas collapse entirely (0.000 Dice) after two rounds of sequential fine-tuning.
- **Surgical Fine-Tuning** almost completely recovers the forgotten classes, at a small cost to final-task (LiTS17) performance.
- **TPGM** gives the best overall balance: it recovers forgotten classes as well as or better than Surgical FT *and* achieves the best final-task scores, at the cost of a heavier per-epoch training loop (bi-level optimization).
- **Hybrid** (Surgical + TPGM together) consistently beats standard fine-tuning and is far cheaper (~1 hour per fine-tuning stage vs. ~8 hours for training from scratch), but doesn't beat either individual method outright — the two regularization strategies partly compete with each other.

## Experimental setup

CT volumes were windowed/normalized, resampled to isotropic 1×1×1mm spacing, and 2D axial slices resized to 224×224. AdamW (lr 0.001, weight decay 0.01), batch size 32, 30 epochs per stage with cosine annealing, trained on an RTX 4060 Ti. Evaluated with Dice Similarity Coefficient and 95th-percentile Hausdorff Distance (HD95).

## Known quirks (for anyone continuing this work)

- Instantiating `CSwinUnet` always dumps a fresh, *untrained* `cswin_unet.pth` to the working directory as a side effect — the ~90MB file at the repo root is not a trained checkpoint.
- `finetune_basic.py`'s internal header still reads `# finetune_surgical.py` — despite its name, it's an earlier/messier surgical fine-tuning prototype (per-batch RGN, (lr, wd) grid search), not a plain baseline. The plain baseline is `finetune.py`.
- `train.py`'s default `list_dir` for the `Synapse` dataset key points at `lists_Synapse_blurred`, not `lists_Synapse` — pass `--list_dir` explicitly if you want clean-data training.
- No `.git` history ships with this copy (it was carried over as a plain file copy rather than a clone), so there's no upstream commit history or remote to diff against.

## Citation

If referencing the base architecture:

```bibtex
@article{liu2025cswin,
  title={CSWin-UNet: Transformer UNet with cross-shaped windows for medical image segmentation},
  author={Liu, Xiao and Gao, Peng and Yu, Tao and Wang, Fei and Yuan, Ru-Yue},
  journal={Information Fusion},
  volume={113},
  pages={102634},
  year={2025},
  publisher={Elsevier}
}
```

## References

* [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
* [CSWin-Transformer](https://github.com/microsoft/CSWin-Transformer)
* Lee, Y. et al. (2022) 'Surgical fine-tuning improves adaptation to distribution shifts', arXiv:2210.11466.
* Tian, J. et al. (2023) 'Trainable Projected Gradient Method for Robust Fine-Tuning', CVPR 2023, pp. 7836-7845.
* Landman, B. et al. (2015) Multi-atlas labeling beyond the cranial vault (Synapse).
* Heller, N. et al., eds. (2024) Kidney and Kidney Tumor Segmentation: MICCAI 2023 Challenge, KiTS 2023.
* Bilic, P. et al. (2023) 'The Liver Tumor Segmentation Benchmark (LiTS)', Medical Image Analysis, 84, 102680.

Full reference list, literature review, and complete methodology/results are in [`Report.pdf`](Report.pdf).
