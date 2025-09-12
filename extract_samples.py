# Standard library
import glob
import os

# Third-party
from PIL import Image

N_frames = 1
EXPERIMENT_NAME="EDM_1200e_test_5"

dir_path=f"samples/{EXPERIMENT_NAME}"
os.makedirs("samples", exist_ok=True)
os.makedirs(dir_path, exist_ok=True)

# Create the frames
frames = []
img_fnames = glob.glob(os.path.join("/Users/erila85/Library/CloudStorage/OneDrive-Linköpingsuniversitet/Desktop/git/Berzelius_code/neural-lam/wandb/run-20250117_221624-sicpzc6v/files/media/images", "*.png"))

var_dict = {}

for fn in img_fnames:
    if "example_1_18" in fn:
        print(fn)
        # Strip "images/"
        # fn_split = fn[7:].split("_")
        fn_split = os.path.basename(fn).split("_")
        var_name = f"{fn_split[0]}_{fn_split[1]}"
        frame_i = int(fn_split[4])

        new_frame = Image.open(fn)

        print(var_name)
        var_dict[var_name] = new_frame

for var_name, frame in var_dict.items():
    # Save all into gif files file that loops forever
    frame.save(os.path.join(dir_path, f"{var_name}.pdf"))
