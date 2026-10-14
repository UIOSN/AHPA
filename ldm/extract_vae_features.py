import os
import argparse
import glob
import shutil
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.autoencoder import VAE_F8D4
from dataset import HFImgLatentDataset, ImageFolderLatentDataset

SELECTED_LAYERS = [
    'down_2_block_0', 'down_2_block_1','down_2_downsample', 'down_3_block_0',
    'down_3_block_1', 'mid_block_1', 'mid_attn_1', 'mid_block_2'
]

@torch.no_grad()
def extract_all_vae_features(vae, raw_image, target_size=None):
    """
    Extract features from each layer of the VAE Encoder.
    If target_size is specified, adaptive_avg_pool2d downsampling is applied to the output of each layer, greatly saving storage space.
    """
    # Normalize to [-1, 1] for VAE
    if raw_image.dtype == torch.uint8 or raw_image.max() > 1.0:
        x = raw_image.float() / 255.0 * 2.0 - 1.0
    else:
        x = raw_image.float() * 2.0 - 1.0
    x = x.to(vae.encoder.conv_in.weight.device)

    enc = vae.encoder
    temb = None
    features = {}

    def maybe_pool(feat):
        # If target size is set, apply adaptive pooling to reduce dimensions and flatten to [B, T, C]
        if target_size is not None:
            pooled = torch.nn.functional.adaptive_avg_pool2d(feat, (target_size, target_size))
            B, C, H, W = pooled.shape
            return pooled.permute(0, 2, 3, 1).reshape(B, H * W, C)
        # Otherwise keep the original shape, move to CPU [B, C, H, W]
        return feat.cpu().half() # Convert to fp16 to save half the space

    # 1. Conv In
    h = enc.conv_in(x)
    features['conv_in'] = maybe_pool(h)

    # 2. Down blocks
    for i_level in range(enc.num_resolutions):
        for i_block in range(enc.num_res_blocks):
            h = enc.down[i_level].block[i_block](h, temb)
            if len(enc.down[i_level].attn) > 0:
                h = enc.down[i_level].attn[i_block](h)
            features[f'down_{i_level}_block_{i_block}'] = maybe_pool(h)
            
        if i_level != enc.num_resolutions - 1:
            h = enc.down[i_level].downsample(h)
            features[f'down_{i_level}_downsample'] = maybe_pool(h)

    # 3. Mid blocks
    h = enc.mid.block_1(h, temb)
    features['mid_block_1'] = maybe_pool(h)

    h = enc.mid.attn_1(h)
    features['mid_attn_1'] = maybe_pool(h)

    h = enc.mid.block_2(h, temb)  
    features['mid_block_2'] = maybe_pool(h)

    return features

import math
import torch.distributed as dist
from datasets import Dataset, load_from_disk, concatenate_datasets

def main(args):
    # Route HF cache away from home to avoid disk pressure.
    hf_cache_dir = os.path.abspath(args.hf_cache_dir)
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = hf_cache_dir

    # DDP Initialization for multi-gpu feature extraction
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        rank = 0

    output_dir = os.path.abspath(args.output_dir)
    chunks_root = output_dir + "_chunks"
    if rank == 0:
        os.makedirs(chunks_root, exist_ok=True)

    if rank == 0: print("Loading VAE...")
    vae = VAE_F8D4().to(device).eval()
    vae_state_dict = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    vae.load_state_dict(vae_state_dict)

    if rank == 0: print(f"Loading Dataset from {args.data_dir}...")
    try:
        source_dataset = HFImgLatentDataset("sdvae-ft-mse-f8d4", args.data_dir, split="train")
    except Exception as e:
        if rank == 0: print(f"HFImgLatentDataset failed: {e}. Falling back to ImageFolderLatentDataset")
        source_dataset = ImageFolderLatentDataset("sdvae-ft-mse-f8d4", args.data_dir, resolution=args.resolution, split="train")

    # To enable distributed writing of Arrow files, each process handles its portion of raw data
    # and saves it as an individual dataset chunk, which can be merged later via load_from_disk.
    chunk_size = math.ceil(len(source_dataset) / world_size)
    start_idx = rank * chunk_size
    end_idx = min(start_idx + chunk_size, len(source_dataset))
    
    subset_indices = list(range(start_idx, end_idx))
    dataset_subset = torch.utils.data.Subset(source_dataset, subset_indices)

    dataloader = DataLoader(
        dataset_subset,
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    def feature_generator():
        # This generator yields data row by row to avoid out-of-memory errors
        for batch in tqdm(dataloader, desc=f"Rank {rank} Extracting", position=rank):
            raw_image = batch[0] # [B, 3, H, W]
            latents = batch[1]   # [B, 8, h, w] (或者对应你设定的维度)
            labels = batch[2]    # [B]

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                layer_features = extract_all_vae_features(
                    vae, 
                    raw_image, 
                    target_size=args.target_size
                )
            layer_features = {k: v for k, v in layer_features.items() if k in SELECTED_LAYERS}
            missing_layers = [k for k in SELECTED_LAYERS if k not in layer_features]
            if missing_layers:
                raise KeyError(f"Missing expected VAE layers: {missing_layers}")
            
            # Since dataset is organized by row, we need to unpack the batch
            B = raw_image.shape[0]
            for i in range(B):
                # Since we already have latents, we don't save them again to save space
                row = {
                    "labels": labels[i].item(),
                }
                # Save specific layers. Considering file size, it's highly recommended to only save needed layers. If you want to save the entire dict,
                # parse dict into different keys. Here we take putting the entire layer_features slice as an example:
                for k, v in layer_features.items():
                    # The shape of v is [B, 256, C], slice the i-th sample and convert to numpy for saving
                    row[k] = v[i].cpu().numpy().astype("float16")
                
                yield row

    # Specify the target dataset schema (you can keep only mid_block_2 to save space based on your actual needs)
    # If feature schema is not specified, HuggingFace will infer it automatically. Usually numpy arrays map to correct Array formats.
    
    # Save the part handled by this GPU as an independent arrow dataset storage folder
    save_path = os.path.join(chunks_root, f"chunk_{rank:02d}_of_{world_size:02d}")
    if os.path.exists(save_path):
        if rank == 0: print(f"Chunk {save_path} already exists. Skipping chunk generation.")
    else:
        # Use streaming generator to create dataset, preventing OOM (Out Of Memory)
        new_dataset = Dataset.from_generator(feature_generator, cache_dir=hf_cache_dir)
        new_dataset.save_to_disk(save_path)
    
    # Synchronize all processes
    if "LOCAL_RANK" in os.environ:
        dist.barrier()

    if rank == 0:
        chunk_dirs = sorted(glob.glob(os.path.join(chunks_root, "chunk_*_of_*")))
        if not chunk_dirs:
            raise RuntimeError(f"No chunk dirs found in {chunks_root}")

        print(f"Merging {len(chunk_dirs)} chunks into: {output_dir}")
        merged = concatenate_datasets([load_from_disk(d) for d in chunk_dirs])

        if os.path.exists(output_dir):
            if args.overwrite_output:
                shutil.rmtree(output_dir)
            else:
                raise FileExistsError(
                    f"Output dir already exists: {output_dir}. "
                    f"Use --overwrite-output to replace it."
                )

        merged.save_to_disk(output_dir)
        if args.cleanup_chunks:
            shutil.rmtree(chunks_root, ignore_errors=True)
        print(f"Merge completed! Single merged dataset saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract VAE All Layer Features")
    parser.add_argument("--data-dir", type=str, default="../data", help="Dataset directory")
    parser.add_argument("--output-dir", type=str, default="../data/vae_layer_features", help="Output directory to save pre-extracted features")
    parser.add_argument("--vae-ckpt", type=str, default="../pretrained_models/sdvae-ft-mse-f8d4.pt", help="Path to VAE weights")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--target-size", type=int, default=16, 
                        help="Pool all saved layers to (target_size x target_size). Default 16 to align with SiT tokens.")
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default="../data/hf_cache",
        help="HuggingFace datasets cache directory (set this to a large disk, not home).",
    )
    parser.add_argument(
        "--cleanup-chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove temporary distributed chunk folders after merge (default: True).",
    )
    parser.add_argument(
        "--overwrite-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing output-dir merged dataset if it exists.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
