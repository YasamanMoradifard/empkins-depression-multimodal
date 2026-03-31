from pathlib import Path

# Change this to your d02_npy root
root = Path("/home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer_/d02_npy")

visual_dir = root / "visual"
audio_dir = root / "audio"

def delete_suffix_files(folder: Path, suffix: str = "_01.npy"):
    if not folder.exists():
        print(f"Folder does not exist: {folder}")
        return
    count = 0
    for path in folder.glob(f"*{suffix}"):
        print(f"Deleting: {path}")
        path.unlink()
        count += 1
    print(f"Deleted {count} files in {folder}")

delete_suffix_files(visual_dir, "_01_visual.npy")
delete_suffix_files(audio_dir, "_01.npy")

"""
import pandas as pd
import numpy as np
import argparse

path = "/home/vault/empkins/tpD/D02/processed_data/paper_implementation/opendbm_avec2019_npz_down2/4_1_training1_neg_aufgabe_01.npz"


def inspect_npz(path):
    # allow_pickle=True in case there are objects / dicts inside
    with np.load(path, allow_pickle=True) as data:
        print(f"File: {path}")
        print("-" * 60)
        print("Keys in this npz file:")
        print(data.files)
        print("-" * 60)

        # Loop through all arrays
        for key in data.files:
            arr = data[key]
            print(f"Key: {key}")
            print(f"  type:   {type(arr)}")
            # Scalars / objects may not have .shape
            try:
                print(f"  shape:  {arr.shape}")
            except AttributeError:
                print("  shape:  <no .shape attribute>")
            print(f"  dtype:  {getattr(arr, 'dtype', 'n/a')}")
            # show a tiny preview
            try:
                print(f"  preview: {arr.ravel()}")
            except Exception:
                print("  preview: <cannot preview>")
            print("-" * 60)


def main():
    #parser = argparse.ArgumentParser(description="Inspect contents of an .npz file.")
    #parser.add_argument("path", help="Path to the .npz file")
    #args = parser.parse_args()
    inspect_npz(path)


if __name__ == "__main__":
    main()

"""



"""
import pandas as pd
import numpy as np

path_npz = "/home/vault/empkins/tpD/D02/processed_data/paper_implementation/opendbm_avec2019_npz_down2/4_1_training1_neg_aufgabe_01.npz"

path_npy = "/home/vault/empkins/tpD/D02/Students/Yasaman/Video_data/MMFformer/Large-Scale-Multimodal-Depression-Detection-main/Large-Scale-Multimodal-Depression-Detection-main/d02_npy/All/train/video/276_ADK_2023-10-26_12-13_Training_1_Aufgabe_16_2023-10-26_12-14.npy"   # change this

data_npz = np.load(path_npz, allow_pickle=True)
data_npy = np.load(path_npy)

####################### NPZ
print("\n############################ NPZ ########################## \n")
# 1) What keys exist?
print("Keys:", data_npz.files)

# 2) Main feature matrix
feat = data_npz["feature"]
print("\nfeature:")
print("  shape:", feat.shape)
print("  dtype:", feat.dtype)
print("  first row, first 10 values:", feat[0, :10])

# 3) Label
label = data_npz["check_label"][0, 0]
print("\ncheck_label:", label, type(label))

# 4) Phase tag
phase = data_npz["phase_tag"][0]
print("\nphase_tag:", phase, type(phase))

# 5) Feature names (optional, for inspection only)
feat_names = data_npz["feature_names"]
print("\nfeature_names:")
print("  count:", len(feat_names))
print("  first 10:", feat_names[:10])


######################### NPY
print("\n\n############################ NPY ########################## \n")

print("Type:", type(data_npy))
print("Shape:", data_npy.shape)
print("Dtype:", data_npy.dtype)

# Peek a bit, depending on dimensions:
if data_npy.ndim == 1:
    print("First 10 values:", data_npy[:10])

elif data_npy.ndim == 2:
    print("First row, first 10 values:", data_npy[0, :10])

elif data_npy.ndim == 3:
    print("Slice [0], shape:", data_npy[0].shape)
    print("First row of first slice:", data_npy[0, 0, :10])

elif data_npy.ndim == 4:
    print("Assuming (T, H, W, C) or (T, C, H, W)")
    print("First sample shape:", data_npy[0].shape)

else:
    print("High-dimensional, just showing a small sample:")
    print(data_npy.reshape(-1)[:10])
"""