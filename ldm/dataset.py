import os
import torch
from torch.utils.data import Dataset
import numpy as np
from datasets import load_from_disk
from torchvision.datasets import ImageFolder
from torchvision import transforms
from utils import center_crop_arr


class HFImgLatentDataset(Dataset):
    PRECOMPUTED = [
        "sdvae-ft-mse-f8d4",
    ]

    def __init__(self, vae_name, data_dir, split="train"):
        assert vae_name in self.PRECOMPUTED, f"VAE {vae_name} not found in {self.PRECOMPUTED}"
        split_str = "val" if split == "val" else ""
        self.img_dataset = load_from_disk(os.path.join(data_dir, "imagenet-latents-images", split_str))
        self.latent_dataset = load_from_disk(os.path.join(data_dir, f"imagenet-latents-{vae_name}", split_str))
        assert len(self.img_dataset) == len(self.latent_dataset), "Image and latent dataset must have the same length"

    def __getitem__(self, idx):
        img_elem = self.img_dataset[idx]
        image, label = img_elem["image"], img_elem["label"]
        image = np.array(image.convert("RGB")).transpose(2, 0, 1)
        latent = self.latent_dataset[idx]["data"]
        return torch.from_numpy(image), torch.tensor(latent), torch.tensor(label)

    def __len__(self):
        return len(self.img_dataset)


class ImageFolderLatentDataset(Dataset):
    PRECOMPUTED = [
        "sdvae-ft-mse-f8d4",
    ]

    def __init__(self, vae_name, data_dir, resolution=256, split="train"):
        assert vae_name in self.PRECOMPUTED, f"VAE {vae_name} not found in {self.PRECOMPUTED}"
        vae_split = "val" if split == "val" else ""
        self.img_dataset = ImageFolder(os.path.join(data_dir, "imagenet", split))
        self.transform_train = transforms.Lambda(
            lambda img: center_crop_arr(np.array(img.convert("RGB")), resolution)
        )
        self.latent_dataset = load_from_disk(os.path.join(data_dir, f"imagenet-latents-{vae_name}", vae_split))
        assert len(self.img_dataset) == len(self.latent_dataset), "Image and latent dataset must have the same length"

    def __getitem__(self, idx):
        image, label = self.img_dataset[idx]
        image = self.transform_train(image)
        image = image.transpose(2, 0, 1)
        latent = self.latent_dataset[idx]["data"]
        return torch.from_numpy(image), torch.tensor(latent), torch.tensor(label)

    def __len__(self):
        return len(self.img_dataset)


class HFImgLatentVaeFeatureDataset(Dataset):
    PRECOMPUTED = [
        "sdvae-ft-mse-f8d4",
    ]

    def __init__(self, vae_name, data_dir, feature_dir, split="train"):
        assert vae_name in self.PRECOMPUTED, f"VAE {vae_name} not found in {self.PRECOMPUTED}"
        split_str = "val" if split == "val" else ""
        self.img_dataset = load_from_disk(os.path.join(data_dir, "imagenet-latents-images", split_str))
        self.latent_dataset = load_from_disk(os.path.join(data_dir, f"imagenet-latents-{vae_name}", split_str))
        
        self.feature_dataset = load_from_disk(feature_dir)
        self.feature_keys = [k for k in self.feature_dataset.column_names if k != "labels"]
        self.feature_dataset.set_format("torch")
        
        assert len(self.img_dataset) == len(self.latent_dataset) == len(self.feature_dataset), "Image, latent, and feature dataset must have the same length"

    def __getitem__(self, idx):
        img_elem = self.img_dataset[idx]
        image, label = img_elem["image"], img_elem["label"]
        image = np.array(image.convert("RGB")).transpose(2, 0, 1)
        latent = self.latent_dataset[idx]["data"]
        feature_elem = self.feature_dataset[idx]
        vae_features = {
            k: feature_elem[k].float() if isinstance(feature_elem[k], torch.Tensor) else torch.tensor(feature_elem[k]).float()
            for k in self.feature_keys
        }
        return torch.from_numpy(image), torch.tensor(latent), vae_features, torch.tensor(label)

    def __len__(self):
        return len(self.img_dataset)
