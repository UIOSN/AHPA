# AHPA: Adaptive Hierarchical Prior Alignment for Diffusion Transformers

This repo contains the Implementation of AHPA.

**✨NeurIPS 2026 Poster** · [Paper](https://arxiv.org/pdf/2605.03317)

## Setup

```bash
conda create -n ahpa python=3.10 -y
conda activate ahpa
pip install -r environment/requirements.txt

hf download REPA-E/iREPA-collections --include "data/**" --local-dir .
hf download REPA-E/iREPA-collections --include "pretrained_models/**" --local-dir .
```

## VAE Feature Mode

AHPA automatically switches between online VAE feature extraction and preprocessed VAE features according to `cached_vae_feature_dir`:

- If `cached_vae_feature_dir` is empty or unset, the frozen VAE extracts the hierarchical target features online during training. This is the default mode.
- If `cached_vae_feature_dir` points to a preprocessed feature directory, training loads the cached features directly.

To use cached features, preprocess them first:

```bash
cd ldm
torchrun --nproc_per_node=8 extract_vae_features.py \
  --data-dir ../data \
  --output-dir ../data/vae_layer_features \
  --vae-ckpt ../pretrained_models/sdvae-ft-mse-f8d4.pt \
  --batch-size 64
```

Then add the cache directory to `configs/ahpa.yaml` before running the normal training command:

```yaml
cached_vae_feature_dir: ../data/vae_layer_features
```

## Train

Run AHPA:

```bash
cd ldm
MODE=ahpa bash run_ahpa.sh
```

Run SRA2:

```bash
cd ldm
MODE=sra2 bash run_ahpa.sh
```

Set `NGPU`, `MODEL`, `DATA_DIR`, `EXP_NAME`, or `MAX_TRAIN_STEPS` before the command to override the defaults. In AHPA, the anchor term assists router training; it is not the main method contribution.

## Generate

```bash
cd ldm
torchrun --nproc_per_node=8 generate.py \
  --model SiT-XL/2 \
  --ckpt exps/ahpa-sit-xl/checkpoints/0400000.pt \
  --sample-dir samples \
  --num-samples 50000 \
  --encoder-depth 7
```

Generated images are saved as PNG files under `samples/`.

## Evaluation

For ImageNet FID/IS evaluation, use the ADM evaluation suite from OpenAI's [`guided-diffusion`](https://github.com/openai/guided-diffusion/tree/main/evaluations) repository and follow its instructions to prepare the reference statistics and sample archive.
