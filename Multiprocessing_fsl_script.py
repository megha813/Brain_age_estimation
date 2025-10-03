import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import pandas as pd




base_dir = "/home/omen/Documents/Megha/ABIDE_DATA/ABIDE_I"

output_dir = "/home/omen/Documents/Megha/ABIDE1_PREPROCESSED"

# cropped_image_folder = "/home/omen/Documents/Megha/preprocessed_ixi1/cropped"
bet_folder_loc ='/home/omen/Documents/Megha/ABIDE1_PREPROCESSED/bet'
bias_field_corrected_loc = '/home/omen/Documents/Megha/ABIDE1_PREPROCESSED/bias_field_corrected'
flirt_registered__loc= '/home/omen/Documents/Megha/ABIDE1_PREPROCESSED/linear_registered'
affine_mat__loc = '/home/omen/Documents/Megha/ABIDE1_PREPROCESSED/affine'
warp_coeff_loc ='/home/omen/Documents/Megha/ABIDE1_PREPROCESSED/warp'
non_linear_loc =  '/home/omen/Documents/Megha/ABIDE1_PREPROCESSED/non_linear_registered'
# complete_preprocessed_loc = '/home/omen/Documents/Megha/ABIDE1_PREPROCESSED/completely_preprocessed'

stages = {
    "bet":        bet_folder_loc,
    "bias_field": bias_field_corrected_loc,
    "flirt":      flirt_registered__loc,
    "affine":     affine_mat__loc,
    "warp":       warp_coeff_loc,
    "non_linear": non_linear_loc
    # "final_bet":  complete_preprocessed_loc,
    # "cropped_image" : cropped_image_folder
}


config_file = "T1_2_MNI152_2mm"
FSL_STANDARD_BRAIN = '/home/omen/fsl5.0.10/data/standard/MNI152_T1_2mm.nii.gz'
FSL_STANDARD_BRAIN_MASKED = '/home/omen/fsl5.0.10/data/standard/MNI152_T1_2mm_brain.nii.gz'
f_value= 0.4


file = pd.read_csv('/home/omen/Documents/Megha/ABIDE_I_relative_paths_with_age_sex.csv')
print(file.columns)
for rel_path in file['relative_path']:
    site = rel_path.split("/")[0]
    # site = os.path.basename(rel_path)
    # site = os.path.basename(rel_path).replace(".nii.gz", "")
    # t1_files = file[file['relative_path'].str.contains("T1", case=False)]
    print(site)

for rel_path in file['relative_path']:
    site = rel_path.split("/")[0]
    # print(site)
    out_fname = f"{site}.nii.gz" 
    print(out_fname)




rel_paths = list(file['relative_path'])

def Multiprocess(rel_path, base_dir, f_value, stages, config_file, FSL_STANDARD_BRAIN_MASKED):
    site = rel_path.split("/")[0]
    out_fname = f"{site}.nii.gz"  
    paths = {
        "bet":        os.path.join(stages["bet"], out_fname),
        "bias_field": os.path.join(stages["bias_field"], site),
        "flirt":      os.path.join(stages["flirt"], out_fname),
        "affine":     os.path.join(stages["affine"], f"{site}.mat"),
        "warp":       os.path.join(stages["warp"], out_fname),
        "non_linear": os.path.join(stages["non_linear"], out_fname)
    }

    for stage, p in paths.items():
        os.makedirs(os.path.dirname(p), exist_ok=True)

    input_image = os.path.join(base_dir, rel_path)
    print(f"Processing {input_image}")

    try:
        print(f"Running BET for {input_image}")
        subprocess.run([
            "bet", input_image, paths["bet"],
            "-f", str(f_value), "-g", "0", "-R"
        ], check=True)

        if not os.path.exists(paths["bet"]):
            print(f"BET did not produce output for: {input_image}")
            return f"FAILED: {input_image}"

        print("Running bias field correction (FAST)...")
        bias_base = paths["bias_field"]
        subprocess.run([
            "fast", "-B", "-t", "1", "-o", bias_base, paths["bet"]
        ], check=True)

        bias_restore = bias_base + "_restore.nii.gz"
        if not os.path.exists(bias_restore):
            print(f"FAST output not found: {bias_restore}")
            return f"FAILED: {input_image}"

        print("Running linear registration (FLIRT)...")
        subprocess.run([
            "flirt", "-ref", FSL_STANDARD_BRAIN_MASKED,
            "-in", bias_restore,
            "-omat", paths["affine"],
            "-out", paths["flirt"]
        ], check=True)

        print("Running non-linear registration (FNIRT)...")
        subprocess.run([
            "fnirt", f"--in={bias_restore}",
            f"--aff={paths['affine']}",
            f"--cout={paths['warp']}",
            f"--config={config_file}"
        ], check=True)

        print("Applying transformation (applywarp)...")
        subprocess.run([
            "applywarp",
            f"--ref={FSL_STANDARD_BRAIN_MASKED}",
            f"--in={bias_restore}",
            f"--warp={paths['warp']}",
            f"--out={paths['non_linear']}"
        ], check=True)

        return f"SUCCESS: {input_image}"

    except subprocess.CalledProcessError as e:
        return f"FAILED: {input_image}, Error: {e}"


def run_in_parallel(rel_paths, base_dir, f_value, stages, config_file, FSL_STANDARD_BRAIN_MASKED, max_workers=4):
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(Multiprocess, rel_path, base_dir, f_value, stages, config_file, FSL_STANDARD_BRAIN_MASKED): rel_path
            for rel_path in rel_paths
        }
        for future in as_completed(futures):
            results.append(future.result())

    return results



results = run_in_parallel(
    rel_paths=rel_paths,
    base_dir="dataset",
    f_value=f_value,
    stages=stages,
    config_file=config_file,
    FSL_STANDARD_BRAIN_MASKED=FSL_STANDARD_BRAIN_MASKED,
    max_workers=4   # number of parallel jobs
)

print("\n".join(results))