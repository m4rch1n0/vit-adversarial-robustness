# Adversarial Decay Profile of Vision Transformers

This project measures how a pretrained ViT-Base/16, fine-tuned on Imagenette,
loses accuracy under white-box evasion attacks as the perturbation budget grows,
and compares it with ResNet-50 and ResNet-18 trained under the same conditions.
Besides accuracy, it tracks confidence and calibration, attention shift,
per-class vulnerability and transferability between architectures.

<p align="center">
  <img src="figures/vit_vs_resnet50/comparative/comparative_accuracy_summary.png" width="700" alt="ViT vs ResNet-50 accuracy decay under four attacks"/>
</p>

## Main results

1. **FGSM.** At ε=16/255 the ViT keeps 58% accuracy, ResNet-50 40% and
   ResNet-18 29%. The ViT curve decreases monotonically. Both CNN curves are
   non-monotonic (accuracy goes up again at intermediate ε), which is usually a
   sign of gradient masking.

2. **PGD-10.** All three models end up at 0%, but at different budgets. Both
   ResNets are already at 0% at ε=1/255, the smallest budget tested. The ViT
   is at 41% at ε=1/255 and reaches 0% only at ε=16/255.

3. **Attention.** Attention rollout shows that PGD moves the CLS attention from
   the object to the background. The cosine similarity between clean and
   adversarial rollout drops to about 0.49 at ε=16/255. At equal ε, FGSM
   shifts attention more than PGD.

4. **Transferability.** Adversarial examples crafted on ResNet-50 fool the ViT
   in 1.3% of cases (effective transfer, PGD ε=8/255). From ResNet-50 to
   ResNet-18 the rate is 26%.

5. **Per-class differences.** Under FGSM ε=4/255, *chain saw* loses 60
   percentage points and *English springer* only 15.

## Results

### Accuracy vs ε

Accuracy as a function of ε for four L∞ attacks. All models are linear probes
on Imagenette trained with the same settings.

| Attack | ViT-B/16 AUDC | ResNet-50 AUDC | ResNet-18 AUDC |
|--------|---------------|----------------|----------------|
| FGSM | 0.635 | 0.530 | 0.317 |
| PGD-10 | 0.030 | 0.000 | 0.000 |
| Patch-PGD (centre) | 0.652 | 0.442 | 0.327 |
| Salient-patch PGD | 0.505 | 0.242 | 0.107 |

AUDC is the area under the accuracy-vs-ε curve, computed with the trapezoidal
rule and normalised to [0, 1]. Higher means more robust.

### Attention shift

Attention rollout heatmaps for one image per class, clean and under FGSM and
PGD at ε=4/255:

<p align="center">
  <img src="figures/attention/attention_rollout_grid_eps4.png" width="550" alt="Attention rollout: clean vs FGSM vs PGD"/>
</p>

The shift is measured as cosine similarity and KL divergence between the clean
and adversarial rollout vectors:

<p align="center">
  <img src="figures/attention/attention_shift_curves.png" width="700" alt="Attention shift and KL divergence vs epsilon"/>
</p>

FGSM and PGD, which perturb the whole image, cause large shifts. The two patch
attacks cause small ones.

### Transferability

Effective transfer rate: among the samples that fool the source model, the
fraction that also fools the target model. Rows are sources, columns are
targets.

PGD, ε=8/255:

| Source \ Target | ViT-B/16 | ResNet-50 | ResNet-18 |
|---|---|---|---|
| ViT-B/16 | - | 6.4% | 9.7% |
| ResNet-50 | 1.3% | - | 25.9% |
| ResNet-18 | 1.5% | 26.5% | - |

FGSM, ε=8/255:

| Source \ Target | ViT-B/16 | ResNet-50 | ResNet-18 |
|---|---|---|---|
| ViT-B/16 | - | 22.2% | 30.3% |
| ResNet-50 | 4.7% | - | 43.1% |
| ResNet-18 | 3.1% | 25.0% | - |

Between the two CNNs the transfer rate is 25-43%, from a CNN to the ViT it is
1-5%. FGSM transfers better than PGD for every pair.

### Per-class vulnerability

Accuracy drop per class under FGSM ε=4/255:

<p align="center">
  <img src="figures/per_class/vulnerability_ranking_fgsm_eps4.png" width="650" alt="Per-class vulnerability ranking"/>
</p>

Confusion matrix under PGD ε=4/255. The errors concentrate on a few class
pairs:

<p align="center">
  <img src="figures/per_class/confusion_matrix_pgd_eps4.png" width="550" alt="Confusion matrix under PGD"/>
</p>

### Examples

Clean images and their FGSM, PGD and Patch-PGD versions at ε=8/255:

<p align="center">
  <img src="figures/qualitative_grid.png" width="550" alt="Clean vs adversarial examples"/>
</p>

## Setup

### Threat model

| | |
|---|---|
| Knowledge | White-box (architecture, weights, gradients) |
| Stage | Test time only, training data and weights are not modified |
| Goal | Untargeted misclassification |
| Norm | L∞, ε ∈ {1, 2, 4, 8, 16}/255 |
| Domain | Pixel space [0, 1], clipped before ImageNet normalisation |

### Models

| Model | Params | Pretraining | Training on Imagenette | Clean acc |
|---|---|---|---|---|
| ViT-Base/16 | 86M | ImageNet-21k, fine-tuned on 1k | Linear probe | 98.9% |
| ResNet-50 | 25M | ImageNet-1k | Linear probe | 98.9% |
| ResNet-18 | 11M | ImageNet-1k | Linear probe | 96.9% |

Training settings for all models: AdamW, lr 1e-3, cosine schedule, 10 epochs,
batch size 64, seed 42.

### Attacks

| Name | Description |
|---|---|
| FGSM | Single step (Goodfellow et al., 2014) |
| PGD-10 | 10 steps, α=ε/4, random start (Madry et al., 2017) |
| Patch-PGD | PGD restricted to a 64×64 centre patch (about 8% of the pixels) |
| Salient-patch PGD | PGD restricted to the 20 ViT patches with the largest gradient norm (about 10% of the pixels) |

### Metrics

- Top-1 accuracy
- Mean confidence on correct and wrong predictions
- ECE (15 bins) and NLL
- Attention shift (cosine similarity and KL divergence of rollout vectors)
- Effective transfer rate
- Per-class accuracy and confusion matrices

## Reproducing the results

```bash
pip install -r requirements.txt
```

Tested on Linux with ROCm 7.2 on an AMD RX 6900 XT (16 GB).

```bash
# 1. Imagenette
mkdir -p data && cd data
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
tar xzf imagenette2-160.tgz && cd ..

# 2. Training (about 10 minutes in total)
python scripts/train.py --config configs/default.yaml
python scripts/train.py --config configs/resnet50.yaml
python scripts/train.py --config configs/resnet18.yaml

# 3. Evaluation under attack
#    checkpoints are saved as experiments/train/<model>/<timestamp>/model.pt
python scripts/evaluate.py --config configs/default.yaml --checkpoint <ckpt>
python scripts/evaluate.py --config configs/resnet50.yaml --checkpoint <ckpt>
python scripts/evaluate.py --config configs/resnet18.yaml --checkpoint <ckpt>

# 4. Analyses
python scripts/attention_analysis.py --config configs/default.yaml --checkpoint <ckpt>
python scripts/per_class_analysis.py --config configs/default.yaml --checkpoint <ckpt>
python scripts/transferability.py --config configs/default.yaml \
    --models "vit:configs/default.yaml:<vit_ckpt>" \
             "resnet50:configs/resnet50.yaml:<r50_ckpt>" \
             "resnet18:configs/resnet18.yaml:<r18_ckpt>"

# 5. Figures
python scripts/visualize.py --metrics <vit_metrics> --metrics-2 <r50_metrics> \
    --label-1 "ViT-Base/16" --label-2 "ResNet-50" --out-dir figures/vit_vs_resnet50
```

`notebooks/results.ipynb` goes through all the results.

## Repository layout

```
configs/      one YAML per model (default.yaml is the ViT)
src/          models, attacks, data loading, evaluation, attention rollout
scripts/      entry points for training, evaluation, analyses and plots
notebooks/    results notebook
experiments/  run outputs (CSV and metadata), one folder per run
figures/      generated plots
```

## Limitations

- One dataset (Imagenette, 10 classes).
- One ViT variant (ViT-Base/16).
- Linear probe only: the backbone is frozen, so the results describe the
  pretrained features.
- No defences are evaluated.
- Untargeted attacks only.
- PGD uses 10 steps. Stronger attacks such as CW or AutoAttack would give
  tighter robustness estimates.

## License

[MIT](LICENSE)
