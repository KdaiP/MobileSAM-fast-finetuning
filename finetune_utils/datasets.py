import torch
from torch.utils.data import Dataset
import numpy as np
import random
from pathlib import Path
from PIL import Image

class SAMDataset(Dataset):
    """
    SAMDataset is a simple custom dataset class for images and their corresponding masks.
    It assumes that for every '.jpg' file, there should be a '.png' mask file.
    """
    def __init__(self, root_dir, transform=None, max_bbox_shift=10, ):
        """
        Args:
            root_dir (string): Directory containing images and masks.
            transform (tuple, optional): A tuple of two optional transforms to be applied
                on an image and its mask respectively.
            bbox_shift (int, optional): Add random perturbation in the range [-bbox_shift, bbox_shift]
                to the bounding box coordinates.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.max_bbox_shift = max_bbox_shift

        # Get all .jpg files
        all_jpgs = list(self.root_dir.rglob('*.jpg'))
        
        # Filter out jpg files that have a corresponding .png mask.
        self.img_list = []
        for img_path in all_jpgs:
            mask_path = img_path.with_suffix('.png')
            if mask_path.exists():
                self.img_list.append(img_path)
            else:
                print(f"Warning: {img_path} doesn't have a corresponding mask!")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.img_list)

    def __getitem__(self, idx):
        """
        Fetch an image-mask pair by index.

        Args:
        - idx (int): Index of the desired sample.

        Returns:
        - tuple: An (image, mask) pair.
        """
        img_name = self.img_list[idx]
        mask_name = self.img_list[idx].with_suffix(".png")

        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).convert("L")

        # Apply transformations if any
        if self.transform:
            image = self.transform[0](image)
            mask = self.transform[1](mask)

        x_min, y_min, x_max, y_max = self.compute_bbox(mask.squeeze(0))

        # Add random perturbation for data augmentation
        c, w, h = mask.shape
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        # Optimized noise generation
        noise_w = torch.clamp(torch.randn(1) * bbox_width * 0.1, min=-self.max_bbox_shift, max=self.max_bbox_shift).round().int().item()
        noise_h = torch.clamp(torch.randn(1) * bbox_height * 0.1, min=-self.max_bbox_shift, max=self.max_bbox_shift).round().int().item()

        x_min = max(0, x_min + noise_w)
        x_max = min(w, x_max + noise_w)
        y_min = max(0, y_min + noise_h)
        y_max = min(h, y_max + noise_h)
        bboxes = torch.tensor([x_min, y_min, x_max, y_max])

        return image, mask, bboxes
    
    def compute_bbox(self, mask_tensor):
        """
        Compute the bounding box of the white region in a binary mask tensor.
        
        Args:
            mask_tensor (tensor): A binary mask tensor. Assumes white as 1 and black as 0.
            
        Returns:
            tensor: A tensor containing coordinates (x_min, y_min, x_max, y_max) of the bbox.
        """
        # Assuming input is a PIL Image. Convert to tensor and squeeze if necessary
        if len(mask_tensor.shape) > 2:
            mask_tensor = mask_tensor.squeeze(0)

        # Detect which rows and columns have white pixels
        rows_any_white = torch.any(mask_tensor == 1, dim=1)
        cols_any_white = torch.any(mask_tensor == 1, dim=0)

        # Get the min and max row and column indices with white pixels
        rows_white = torch.where(rows_any_white)[0]
        cols_white = torch.where(cols_any_white)[0]

        if rows_white.nelement() == 0 or cols_white.nelement() == 0:
            # No white pixels, return zeros
            return torch.tensor([0, 0, 0, 0])

        y_min, y_max = rows_white[0].item(), rows_white[-1].item()
        x_min, x_max = cols_white[0].item(), cols_white[-1].item()

        return x_min, y_min, x_max, y_max