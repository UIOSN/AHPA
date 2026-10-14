import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, DistributedDataParallelKwargs
from models.autoencoder import VAE_F8D4
from models.sit import SiT_models
from loss import SILoss
from dataset import HFImgLatentDataset, ImageFolderLatentDataset, HFImgLatentVaeFeatureDataset
import wandb
import math
from torchvision.utils import make_grid
from utils import SpatialNormalization, ALL_SPNORM_METHODS
from models.sit import ALL_PROJECTION_LAYER_TYPES

logger = get_logger(__name__)
class DynamicRouter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = torch.nn.Linear(1, 128)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 256),
            torch.nn.SiLU(),
        )
        self.alpha_head = torch.nn.Linear(256, 8)
        self.beta_head = torch.nn.Linear(256, 2)
        torch.nn.init.trunc_normal_(self.alpha_head.weight, std=0.01)
        alpha_bias = torch.zeros(8)
        alpha_bias[2] = 4.0 
        alpha_bias[7] = 4.0
        self.alpha_head.bias.data.copy_(alpha_bias)
        torch.nn.init.trunc_normal_(self.beta_head.weight, std=0.01)
        beta_bias = torch.tensor([3.0, 3.0])
        self.beta_head.bias.data.copy_(beta_bias)
    def forward(self, t):
        t = t.view(-1, 1).float()
        x = self.time_embed(t)
        x = self.mlp(x)
        alpha = self.alpha_head(x)
        beta = self.beta_head(x)
        return alpha, beta
def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x
@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z - latents_bias) * latents_scale
    return z 
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger
def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag
def get_raw_model(model):
    """
    Unwrap distributed wrapper (e.g. DDP .module) and torch.compile (_orig_mod) without
    using accelerator.unwrap_model, to avoid KeyError on _orig_mod in some versions.
    """
    m = model.module if hasattr(model, "module") else model
    return getattr(m, "_orig_mod", m)
@torch.no_grad()
def get_vae_mid_features(vae, raw_image, target_size=16):
    """
    Extract VAE encoder features from various layers and downsample to match SiT token grid.
    raw_image: [B, 3, H, W] in [0, 255] or [0, 1]
    Returns: dict of layers mapped to [B, T, D] where T=target_size*target_size
    """
    if raw_image.dtype == torch.uint8 or raw_image.max() > 1.0:
        x = raw_image.float() / 255.0 * 2.0 - 1.0
    else:
        x = raw_image.float() * 2.0 - 1.0
    x = x.to(vae.encoder.conv_in.weight.device)
    enc = vae.encoder
    temb = None
    features = {}
    def format_feature(h):
        h_pool = torch.nn.functional.adaptive_avg_pool2d(h, target_size)
        B, C, H, W = h_pool.shape
        return h_pool.permute(0, 2, 3, 1).reshape(B, H * W, C)
    h = enc.conv_in(x)
    features['conv_in'] = format_feature(h)
    for i_level in range(enc.num_resolutions):
        for i_block in range(enc.num_res_blocks):
            h = enc.down[i_level].block[i_block](h, temb)
            features[f'down_{i_level}_block_{i_block}'] = format_feature(h)
            if len(enc.down[i_level].attn) > 0:
                h = enc.down[i_level].attn[i_block](h)
                features[f'down_{i_level}_attn_{i_block}'] = format_feature(h)
        if i_level != enc.num_resolutions - 1:
            h = enc.down[i_level].downsample(h)
            features[f'down_{i_level}_downsample'] = format_feature(h)
    h = enc.mid.block_1(h, temb)
    features['mid_block_1'] = format_feature(h)
    h = enc.mid.attn_1(h)
    features['mid_attn_1'] = format_feature(h)
    h = enc.mid.block_2(h, temb)
    features['mid_block_2'] = format_feature(h)
    return features
def main(args):
    if args.cached_vae_feature_dir=="":
        print(f"Warning: --cached-vae-feature-dir is an empty string, treating as None (no cache will be used).")
        args.cached_vae_feature_dir = None  
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8
    in_channels = 4
    use_cached_vae_feature = getattr(args, "cached_vae_feature_dir", None) is not None
    if accelerator.is_main_process:
        if use_cached_vae_feature:
            print(f"\n==========================================================================")
            print(f">>> SUCCESS: Training WILL use VAE cache from {args.cached_vae_feature_dir} <<<")
            print(f"==========================================================================\n")
        else:
            print(f"\n==========================================================================")
            print(">>> WARNING: Training will NOT use VAE cache, extracting features online! <<<")
            print(f"==========================================================================\n")
    if args.loss_mode == "ahpa":
        z_dims = [1024]
        if accelerator.is_main_process:
            logger.info("ahpa: concatenated dimension is 1024")
    elif args.loss_mode == "sra2":
        z_dims = [in_channels]
        if accelerator.is_main_process:
            logger.info(f"sra2: in_channels={in_channels} -> z_dims={z_dims}")
    else:
        raise ValueError(f"Unsupported loss mode: {args.loss_mode}")
    block_kwargs = {
        "fused_attn": args.fused_attn, 
        "qk_norm": args.qk_norm,
    }
    model = SiT_models[args.model](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=args.num_classes,
        class_dropout_prob=args.cfg_prob,
        z_dims = z_dims,
        encoder_depth=args.encoder_depth,
        projection_layer_type=args.projection_layer_type,
        proj_kwargs_kernel_size=args.proj_kwargs_kernel_size,
        **block_kwargs
    )
    model = model.to(device)
    ema = deepcopy(model).to(device)  
    requires_grad(ema, False)
    dynamic_router = None
    if args.loss_mode == "ahpa":
        dynamic_router = DynamicRouter().to(device)
    if use_cached_vae_feature and accelerator.is_main_process:
        logger.info(f"Using cached VAE features, selected layer: {args.vae_feature_layer}")
    vae = VAE_F8D4().to(device).eval()
    vae_state_dict = torch.load("../pretrained_models/sdvae-ft-mse-f8d4.pt", map_location=device, weights_only=False)
    vae.load_state_dict(vae_state_dict)
    latents_stats = torch.load("../pretrained_models/sdvae-ft-mse-f8d4-latents-stats.pt", map_location=device, weights_only=False)
    latents_scale = latents_stats['latents_scale'].to(device).view(1, -1, 1, 1)
    latents_bias = latents_stats['latents_bias'].to(device).view(1, -1, 1, 1)
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type,
        weighting=args.weighting,
        proj_coeff=args.proj_coeff,
        loss_mode=args.loss_mode,
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    params = list(model.parameters())
    if dynamic_router is not None:
        params = params + list(dynamic_router.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    if use_cached_vae_feature:
        try:
            train_dataset = HFImgLatentVaeFeatureDataset("sdvae-ft-mse-f8d4", args.data_dir, args.cached_vae_feature_dir, split="train")
        except Exception as e:
            print(f"Error loading HFImgLatentVaeFeatureDataset: {e}")
            raise SystemExit(1)
    else:
        try:
            train_dataset = HFImgLatentDataset("sdvae-ft-mse-f8d4", args.data_dir, split="train")
        except Exception as e:
            print(f"Error loading HFImgLatentDataset: {e}")
            print("Falling back to ImageFolderLatentDataset")
            train_dataset = ImageFolderLatentDataset(
                "sdvae-ft-mse-f8d4",
                args.data_dir,
                resolution=args.resolution,
                split="train",
            )
    print(train_dataset)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
        steps_per_epoch = len(train_dataloader)
        opt_steps_per_epoch = math.ceil(steps_per_epoch / args.gradient_accumulation_steps)
        total_opt_steps = opt_steps_per_epoch * args.epochs
        logger.info(
            f"Steps per epoch (batches): {steps_per_epoch} | "
            f"optimizer steps/epoch: {opt_steps_per_epoch} | "
            f"total optimizer steps for {args.epochs} epochs: {total_opt_steps}"
        )
    update_ema(ema, model, decay=0)  
    model.train()  
    ema.eval()  
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            weights_only=False,
        )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        if dynamic_router is not None:
            dynamic_router.load_state_dict(ckpt['dynamic_router'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']
    if args.compile:
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 512
        model = torch.compile(model, backend="inductor", mode="default")
        @torch.compile(fullgraph=False)
        def optim_step_fn():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    else:
        def optim_step_fn():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    to_prepare = [model, optimizer, train_dataloader]
    if dynamic_router is not None:
        to_prepare.append(dynamic_router)
    prepared = accelerator.prepare(*to_prepare)
    model, optimizer, train_dataloader = prepared[0], prepared[1], prepared[2]
    idx = 3
    if dynamic_router is not None:
        dynamic_router = prepared[idx]
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="AHPA-SRA2", 
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    sample_batch_size = args.n_samples // accelerator.num_processes
    batch = next(iter(train_dataloader))
    if use_cached_vae_feature:
        _, gt_xs, _, gt_labels = batch
    else:
        _, gt_xs, gt_labels = batch
    gt_xs = gt_xs[:sample_batch_size]
    gt_labels = gt_labels[:sample_batch_size]
    gt_xs = sample_posterior(
        gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
    )
    ys = gt_labels.to(device)
    n = ys.size(0)
    xT = torch.randn((n, in_channels, latent_size, latent_size), device=device)
    spnorm = SpatialNormalization(args.spnorm_method)
    for epoch in range(args.epochs):
        model.train()
        for batch in train_dataloader:
            vae_feature = None
            if use_cached_vae_feature:
                raw_image, x, vae_feature, y = batch
            else:
                raw_image, x, y = batch
            raw_image = raw_image.to(device)
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)
            if vae_feature is not None:
                if isinstance(vae_feature, dict):
                    vae_feature = {k: v.to(device) for k, v in vae_feature.items()}
                    for layer in args.vae_feature_layer:
                        if layer not in vae_feature:
                            raise KeyError(
                                f"VAE feature layer '{layer}' not found. "
                                f"Available layers: {sorted(list(vae_feature.keys()))}"
                            )
            labels = y
            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
                zs = []
                with accelerator.autocast():
                    if args.loss_mode == "ahpa":
                        if use_cached_vae_feature:
                            vae_features_dict = vae_feature
                        else:
                            raw_model = get_raw_model(model)
                            grid_size = int(raw_model.x_embedder.num_patches ** 0.5)
                            vae_features_dict = get_vae_mid_features(vae, raw_image, target_size=grid_size)
                        group_configs = [
                            [ 'down_2_block_0', 'down_2_block_1','down_2_downsample', 'down_3_block_0',], 
                            [ 'down_3_block_1', 'mid_block_1', 'mid_attn_1', 'mid_block_2'] 
                        ]
                        normed_features = []
                        for group in group_configs:
                            grp_feats = []
                            for layer_name in group:
                                feat = vae_features_dict[layer_name].to(device)
                                normed = spnorm(feat=feat, zscore_alpha=args.zscore_alpha)
                                grp_feats.append(normed)
                            normed_features.append(grp_feats)
                    elif args.loss_mode == "sra2":
                        pass
            with accelerator.accumulate(model):
                model_kwargs = dict(y=labels)
                time_input = None
                if args.loss_mode == "ahpa":
                    if args.weighting == "uniform":
                        time_input = torch.rand((x.shape[0], 1, 1, 1), device=device)
                    elif args.weighting == "lognormal":
                        rnd_normal = torch.randn((x.shape[0], 1 ,1, 1), device=device)
                        sigma = rnd_normal.exp()
                        if args.path_type == "linear":
                            time_input = sigma / (1 + sigma)
                        elif args.path_type == "cosine":
                            time_input = 2 / torch.pi * torch.atan(sigma)
                    else:
                        time_input = torch.rand((x.shape[0], 1, 1, 1), device=device)
                    t_flat = time_input.view(x.shape[0])
                    alpha_logits, beta_logits = dynamic_router(t_flat)
                    beta_weights = torch.sigmoid(beta_logits) 
                    group_vectors = []
                    alpha_idx = 0
                    for g_idx, grp_feats in enumerate(normed_features):
                        group_size = len(grp_feats)
                        a_logits = alpha_logits[:, alpha_idx : alpha_idx + group_size]
                        a_weights = torch.softmax(a_logits, dim=-1) 
                        alpha_idx += group_size
                        group_normed = []
                        for j, normed in enumerate(grp_feats):
                            w = a_weights[:, j].view(-1, 1, 1)
                            group_normed.append(normed * w)
                        group_aggr = sum(group_normed) 
                        b_w = beta_weights[:, g_idx].view(-1, 1, 1)
                        group_vectors.append(group_aggr * b_w)
                    zs = [torch.cat(group_vectors, dim=-1)]
                elif args.loss_mode == "sra2":
                    raw_model = get_raw_model(model)
                    grid = int(raw_model.x_embedder.num_patches ** 0.5)  
                    lat_small = torch.nn.functional.adaptive_avg_pool2d(x, (grid, grid))  
                    B_, C_, H_, W_ = lat_small.shape
                    z_latent = lat_small.permute(0, 2, 3, 1).reshape(B_, H_ * W_, C_)    
                    zs = [z_latent]
                loss, proj_loss, loss_dict = loss_fn(model, x, model_kwargs, zs=zs, time_input=time_input)
                loss_mean = loss.mean()
                proj_loss_mean = proj_loss.mean()
                loss = loss_mean + proj_loss_mean
                anchor_loss_mean = torch.tensor(0.0, device=device)
                if args.loss_mode == "ahpa":
                    target_g1 = torch.full((alpha_logits.shape[0],), 2, dtype=torch.long, device=device)
                    target_g2 = torch.full((alpha_logits.shape[0],), 3, dtype=torch.long, device=device)
                    anchor_loss_mean = (torch.nn.functional.cross_entropy(alpha_logits[:, 0:4], target_g1) + torch.nn.functional.cross_entropy(alpha_logits[:, 4:8], target_g2)) / 2.0
                    loss = loss + args.dynamic_router_anchor_coeff * anchor_loss_mean
                msg = (f"epoch: {epoch}, global_step: {global_step}, "
                       f"Loss: {loss.item():.4f}, Denoising: {loss_mean.item():.4f}, "
                       f"Proj: {proj_loss_mean.item():.4f}")
                print(msg) 
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = list(model.parameters())
                    if dynamic_router is not None:
                        params_to_clip = params_to_clip + list(dynamic_router.parameters())
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optim_step_fn()
                if accelerator.sync_gradients and global_step % args.ema_update_freq == 0:
                    update_ema(ema, get_raw_model(model), decay=args.ema_decay)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1                
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    original_model = get_raw_model(model)
                    checkpoint = {
                        "model": original_model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    if dynamic_router is not None:
                        checkpoint["dynamic_router"] = accelerator.unwrap_model(
                            dynamic_router
                        ).state_dict()
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            if (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                from samplers import euler_sampler
                with torch.no_grad():
                    samples = euler_sampler(
                        model, 
                        xT, 
                        ys,
                        num_steps=50, 
                        cfg_scale=4.0,
                        guidance_low=0.,
                        guidance_high=1.,
                        path_type=args.path_type,
                        heun=False,
                    ).to(torch.float32)
                    samples = vae.decode(samples / latents_scale + latents_bias).sample
                    gt_samples = vae.decode(gt_xs / latents_scale + latents_bias).sample
                    samples = (samples + 1) / 2.
                    gt_samples = (gt_samples + 1) / 2.
                out_samples = accelerator.gather(samples.to(torch.float32))
                gt_samples = accelerator.gather(gt_samples.to(torch.float32))
                accelerator.log({"samples": wandb.Image(array2grid(out_samples)),
                                 "gt_samples": wandb.Image(array2grid(gt_samples))})
                logging.info("Generating EMA samples done.")
            logs = {
                "denoising_loss": accelerator.gather(loss_mean).mean().detach().item(),
                "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }
            logs.update(loss_dict)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break
    model.eval()  
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=50000)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=str, default="8")
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--projection-layer-type", type=str, default="conv", choices=ALL_PROJECTION_LAYER_TYPES)
    parser.add_argument("--proj-kwargs-kernel-size", type=int, default=1, choices=[1, 3, 5, 7])
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--ema-update-freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) 
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--proj-coeff", type=str, default="1.0")
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--loss-mode", type=str, default="ahpa",
        choices=["ahpa", "sra2"],
        help="Loss mode: 'ahpa' uses the timestep-conditioned dynamic router; "
             "'sra2' aligns SiT with its own patch-embedded latent_z")
    parser.add_argument("--dynamic-router-anchor-coeff", type=float, default=0.5)
    parser.add_argument("--spnorm-method", type=str, default="zscore", choices=ALL_SPNORM_METHODS)
    parser.add_argument("--zscore-alpha", type=float, default=0.6)
    parser.add_argument("--config", type=str, default=None,
        help="Path to a YAML config file (configs/ahpa.yaml or configs/sra2.yaml)")
    parser.add_argument("--cached-vae-feature-dir", type=str, default="",
        help="Path to pre-extracted VAE features. If provided, dynamically loaded from disk.")
    parser.add_argument("--vae-feature-layer", type=str, default="mid_block_2",
        help="Which cached VAE feature layer to use during training when --cached-vae-feature-dir is set.")
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    if args.config:
        from omegaconf import OmegaConf
        config = OmegaConf.load(args.config)
        for key, value in config.items():
            key_underscore = key.replace('-', '_')
            if hasattr(args, key_underscore):
                setattr(args, key_underscore, value)
    if args.loss_mode not in {"ahpa", "sra2"}:
        parser.error("--loss-mode must be either 'ahpa' or 'sra2'")
    if isinstance(args.encoder_depth, str):
        args.encoder_depth = [int(x.strip()) for x in args.encoder_depth.split(",")]
    elif isinstance(args.encoder_depth, int):
        args.encoder_depth = [args.encoder_depth]
    if isinstance(args.vae_feature_layer, str):
        args.vae_feature_layer = [x.strip() for x in args.vae_feature_layer.split(",")]
    return args
if __name__ == "__main__":
    args = parse_args()
    main(args)
