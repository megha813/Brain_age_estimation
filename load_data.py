import os
import torch
import nibabel as nib
import numpy as np
import pandas as pd
import torch.nn.functional as F
from monai.transforms import Compose, Rand3DElastic, RandAffine, RandFlip, RandRotate, RandSpatialCrop
import monai.transforms as mt


def nii_loader(path, dtype=np.float32, mmap_mode='r'):
    """
    Load NIfTI file with memory mapping option for large files.

    Args:
        path: Path to NIfTI file
        dtype: Data type to cast to (default: float32)
        mmap_mode: Memory mapping mode (default: 'r' for read-only)
                   Set to None to load data into memory
    """
    img = nib.load(str(path))
    # Use memory mapping for large files
    data = img.get_fdata(dtype=dtype, caching='unchanged')
    return data


def read_table(path):
    """Read Excel table and return values"""
    return pd.read_excel(path, header=None).values


def white0(image, threshold=0):
    """
    Standardize voxels with value > threshold

    Args:
        image: Input image
        threshold: Threshold value

    Returns:
        Standardized image
    """
    image = image.astype(np.float32)
    mask = (image > threshold).astype(int)

    # Vectorized implementation to avoid unnecessary memory allocation
    image_h = image * mask

    # Calculate mean and std only for relevant voxels
    non_zero_voxels = np.sum(mask)
    if non_zero_voxels > 0:
        mean = np.sum(image_h) / non_zero_voxels

        # More memory efficient way to calculate std
        std_sum = np.sum((image_h - mean * mask) ** 2)
        std = np.sqrt(std_sum / non_zero_voxels)

        if std > 0:
            normalized = mask * (image - mean) / std
            # Use in-place operations to reduce memory usage
            image = normalized + image * (1 - mask)
            return image

    # Default case
    return np.zeros_like(image, dtype=np.float32)


class IMG_Folder(torch.utils.data.Dataset):
    transform = Compose([
        RandAffine(
            prob=0.5,
            rotate_range=(np.deg2rad(20), np.deg2rad(20), np.deg2rad(20)),  # rotation
            translate_range=(10, 10, 10),  # translation
            padding_mode="border"
        ),
        RandFlip(
            prob=0.5,
            spatial_axis=[0, 1, 2]  # random flip along any axis
        )
    ])

    
    """
    Dataset class for loading brain images with memory optimizations
    """

    def __init__(self, excel_path, data_path, loader=nii_loader, transforms=transform, preload=False):
        """
        Args:
            excel_path: Path to Excel file with metadata
            data_path: Path to directory with NIfTI files
            loader: Function to load NIfTI files
            transforms: Transforms to apply to images
            preload: Whether to preload all data into memory (default: False)
        """
        self.root = data_path
        self.sub_fns = sorted(os.listdir(self.root))
        
        self.table_refer = read_table(excel_path)
        self.loader = loader
        
        self.transform= transforms
        self.preload = preload

        # Create a mapping from subject ID to metadata for faster lookup
        self.metadata = {}
        for f in self.table_refer[1:]:      #i have added this to skip the first row of the excel sheet
            sid = str(f[0])
            # print(sid)
            slabel = int(f[1])
            # print(slabel)
            smale = f[2]
            if smale == 1:
                smale = 0
            elif smale == 2:
                smale = 1
            else:
                raise ValueError(f"Unexpected SEX_ID value: {smale}")
            # print(smale)
            self.metadata[sid] = (slabel, smale)
            

        # Optionally preload all data into memory
        if preload:
            self.cached_data = {}
            for sub_fn in self.sub_fns:
                if sub_fn in self.metadata:
                    sub_path = os.path.join(self.root, sub_fn)
                    self.cached_data[sub_fn] = self.loader(sub_path)

    def __len__(self):
        return len(self.sub_fns)

    def __getitem__(self, index):

        sub_fn = self.sub_fns[index]

        # Get metadata for this subject
        if sub_fn not in self.metadata:
            # print(f"wrong{sub_fn}")
            # Find manually if not in mapping (fallback)
            for f in self.table_refer[1:]:   #i have added this extra syntax for [1:] as the - error accessing the column header also
                sid = str(f[0])  #here the numbers 0,2,3 are changes to match with the excel i have given
                slabel = int(f[1])
                smale = f[2]
                if sid == sub_fn:
                    break
        else:
            slabel, smale = self.metadata[sub_fn]
            sid = sub_fn

        # Load image data
        if self.preload and sub_fn in self.cached_data:
            img = self.cached_data[sub_fn].copy()  # Make a copy to avoid modifying cached data
        else:
            sub_path = os.path.join(self.root, sub_fn)    #image_path constructed
            img = self.loader(sub_path)
            # print("Image shape:", img.shape)

        # Preprocessing
        img = white0(img)
        img = np.expand_dims(img, 0) 
        
        if self.transform is not None:
            img = self.transform(img)
        # img = img.squeeze(0)       #done to reduce the dimension of the image - so that the model can take this easily in the input for the permute
        # Convert to contiguous float tensor
        img = img[0]
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        
        # img= img.unsqueeze(0)

        # print("Image shape final:", img.shape)

        return (img, sid, slabel, smale)