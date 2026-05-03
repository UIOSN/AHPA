# AHPA: Adaptive Hierarchical Representation Alignment

This repository is the official implementation of the AHPA (Adaptive Hierarchical Representation Alignment) method, based on SiT and REPA. It demonstrates how dynamic routing over intermediate representations can enhance representation alignment.

## Dependency Setup

```bash
cd AHPA

# Create environment (Python 3.10+)
conda create -n ahpa python=3.10 -y && conda activate ahpa

# Install dependencies (same as iREPA)
pip install -r requirements.txt
```

## Data & Pretrained Weights 

You will need the pre-cached ImageNet latents and VAE/Encoder pretrained models.

Download the full pre-processed dataset and weights from the iREPA/REPA-E collections to the current directory:
```bash
hf download REPA-E/iREPA-collections --include "data/**" --local-dir "."
hf download REPA-E/iREPA-collections --include "pretrained_models/**" --local-dir "."
hf download REPA-E/iREPA-collections --include "VIRTUAL_imagenet256_labeled.npz" --local-dir "."
```


### Folder Structure
Ensure your workspace looks like this:
```text
AHPA/
├── data/
│   ├── imagenet-latents-images/
│   └── imagenet-latents-sdvae-ft-mse-f8d4/
├── pretrained_models/
│   ├── sdvae-ft-mse-f8d4.pt
│   └── sdvae-ft-mse-f8d4-latents-stats.pt
├── VIRTUAL_imagenet256_labeled.npz
├── ldm
└── guided-diffusion/
    └── evaluations/evaluator_batch.py
```

### Pre-extract VAE Features (Optional but Recommended)
To significantly speed up training, we recommend pre-extracting the necessary VAE features. We provide a script to do this in distributed fashion.

```bash
cd ldm
torchrun --nproc_per_node=8 extract_vae_features.py \
  --data-dir ../data \
  --output-dir ../data/vae_layer_features \
  --vae-ckpt ../pretrained_models/sdvae-ft-mse-f8d4.pt \
  --batch-size 64
```
Once extracted, you can append `--cached-vae-feature-dir ../data/vae_layer_features` to your training commands to skip online extraction.

## Running the Pipeline

This repository provides instructions for Latent Diffusion (SiT) experiments. All code lies within the `ldm` directory.


### One-Click Reproduction Pipeline

We provide a convenient bash script (`run_ahpa.sh`) that automates the entire pipeline: Training -> Generation -> Evaluation chronologically.

```bash
cd ldm
bash run_ahpa.sh
```
You can configure your desired experiments and models within the `run_ahpa.sh`.

### 1. Training

To train the models with different loss modes (AHPA, REPA, SRA2):

**AHPA**
```bash
cd ldm
accelerate launch train.py --config configs/ahpa.yaml \
  --model "SiT-XL/2" \
  --encoder-depth 7 \
  --data-dir ../data \
  --exp-name "ahpa-sit-xl" \
  --max-train-steps 400000 \
  --learning-rate 2e-4 \
  --max-grad-norm 2.0 \
  --batch-size 256
```

**baseline REPA**
```bash
cd ldm
accelerate launch train.py --config configs/repa.yaml \
  --model "SiT-XL/2" \
  --encoder-depth 7 \
  --data-dir ../data \
  --exp-name "repa-sit-xl" \
  --max-train-steps 400000 \
  --learning-rate 2e-4 \
  --max-grad-norm 2.0 \
  --batch-size 256
```

**SRA2**
```bash
cd ldm
accelerate launch train.py --config configs/sra2.yaml \
  --model "SiT-XL/2" \
  --encoder-depth 7 \
  --data-dir ../data \
  --exp-name "sra2-sit-xl" \
  --max-train-steps 400000 \
  --learning-rate 2e-4 \
  --max-grad-norm 2.0 \
  --batch-size 256
```

### 2. Generation

To generate samples using a trained checkpoint:
```bash
cd ldm
python generate_all.py \
  --exp-name "ahpa-sit-xl" \
  --model "SiT-XL/2" \
  --steps 0400000 \
  --sample-dir samples \
  --num-samples 50000 \
  --nproc 8 \
  --encoder-depth 7 \
  --sample-list-out "ahpa-sit-xl_samples.txt"
```

### 3. Evaluation (FID/IS)

We use the ADM evaluation suite to compute ImageNet 256x256 metrics. Make sure `VIRTUAL_imagenet256_labeled.npz` is in the root directory. 

```bash
python ../guided-diffusion/evaluations/evaluator_batch.py \
  --ref_batch ../VIRTUAL_imagenet256_labeled.npz \
  --sample_list ahpa-sit-xl_samples.txt \
  --log eval_results_ahpa-sit-xl.log
```


