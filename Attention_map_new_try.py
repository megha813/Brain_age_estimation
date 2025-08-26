import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import datasets
from scipy.ndimage import zoom
import nibabel as nib
import torch
from matplotlib.cm import viridis
from load_data import nii_loader

path='./70s/'

path_new='./70s/'

# Load the atlas
aal_atlas = datasets.fetch_atlas_aal()
atlas_filename = aal_atlas.maps
atlas_nii = nib.load(atlas_filename)
atlas_data = atlas_nii.get_fdata()
region_labels = np.unique(atlas_data)[1:]  # Exclude 0 (background)
region_mapping = {code: label for code, label in zip(region_labels, aal_atlas.labels)}

print(list(region_mapping.values()))

# Load your attention maps here as 2D numpy arrays
attn_map_axial = torch.load(path+"normalized_attn_map1.pt")
attn_map_coronal = torch.load(path+"normalized_attn_map2.pt")
attn_map_sagittal = torch.load(path+"normalized_attn_map3.pt")

# Adjust this function to handle 2D slices
def extend_and_resample(attn_map_2d, atlas_nii, slice_orientation):
    # Create an empty 3D volume
    attn_map_3d = np.zeros(atlas_nii.shape)

    # Depending on the orientation, place the 2D slice into the 3D volume
    if slice_orientation == 'axial':
        attn_map_3d[:, :, attn_map_3d.shape[2] // 2] = attn_map_2d
    elif slice_orientation == 'coronal':
        attn_map_3d[:, attn_map_3d.shape[1] // 2, :] = attn_map_2d
    elif slice_orientation == 'sagittal':
        attn_map_3d[attn_map_3d.shape[0] // 2, :, :] = attn_map_2d

    # Now resample to match the atlas resolution
    resampled_attn_map = zoom(attn_map_3d, (1, atlas_nii.header.get_zooms()[1] / 2, 1), order=0)

    return resampled_attn_map


# Extend and resample attention maps to 3D
attn_map_axial_3d = extend_and_resample(attn_map_axial, atlas_nii, 'axial')
attn_map_coronal_3d = extend_and_resample(attn_map_coronal, atlas_nii, 'coronal')
attn_map_sagittal_3d = extend_and_resample(attn_map_sagittal, atlas_nii, 'sagittal')
#
attn_map_average_3d = (attn_map_axial_3d+attn_map_coronal_3d+attn_map_sagittal_3d)/3
np.save(path+'3d_attention', attn_map_average_3d)

#普通三张attention map的话就用上面的两行代码,遮挡分析的话就用下面的两行代码

#attn_map_average_3d = nii_loader('../Gender/occlusion.nii.gz')

#attn_map_average_3d = nii_loader(path_new+'VGG_30.nii.gz')


# DataFrame to store mean attention values for each region and view
attention_df = pd.DataFrame(index=region_labels, columns=['Axial', 'Coronal', 'Sagittal'])

attention_df = pd.DataFrame(index=region_labels)


# Calculate mean attention in each region
for region in attention_df.index:
    if region == 0:  # Skip the background
        continue
    region_mask = (atlas_data == region)
    attention_df.loc[region, 'Axial'] = attn_map_axial_3d[region_mask].mean()
    attention_df.loc[region, 'Coronal'] = attn_map_coronal_3d[region_mask].mean()
    attention_df.loc[region, 'Sagittal'] = attn_map_sagittal_3d[region_mask].mean()
    attention_df.loc[region, 'Average'] = attn_map_average_3d[region_mask].mean()


# Replace region numbers with region names using the atlas's labels

attention_df.index = [region_mapping[code] for code in attention_df.index]

attention_df = attention_df * 100

fig, ax = plt.subplots(figsize=(15, 30))
colors = viridis(np.linspace(0, 1, len(attention_df)))
bars=ax.barh(attention_df.index, attention_df['Average'], color=colors, edgecolor='black')
ax.set_title('Average Attention')
ax.invert_yaxis()
ax.set_xlim(0, attention_df['Average'].max() + 0.1 * attention_df['Average'].max())  # Set limits for x-axis to make space for text

# Adding the text annotations on the bars
for bar in bars:
    width = bar.get_width()  # Get the width of the bar (the numerical value)
    label_x_pos = width + 0.02  # Set the position of the label slightly right of the bar
    ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center')


plt.tight_layout()
#plt.show()
plt.savefig(path_new+"attention_area.png")

