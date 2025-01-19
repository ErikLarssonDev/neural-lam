# https://pythonprogramming.altervista.org/png-to-gif/
from PIL import Image
import glob
import os

N_frames = 19
duration=700
EXPERIMENT_NAME="EDM_1200e"

dir_path=f"gifs/{EXPERIMENT_NAME}"
os.makedirs("gifs", exist_ok=True)
os.makedirs(dir_path, exist_ok=True)

# Create the frames
frames = []
img_fnames = glob.glob(os.path.join("/proj/berzelius-2022-164/users/x_erila/neural-lam/wandb/run-20250115_103417-r35gs1j7/files/media/images", "*.png"))

var_dict = {}

for fn in img_fnames:
    print(fn)
    if "example" in fn:
        # Strip "images/"
        # fn_split = fn[7:].split("_")
        fn_split = os.path.basename(fn).split("_")
        var_name = f"{fn_split[0]}_{fn_split[1]}"
        frame_i = int(fn_split[4])

        new_frame = Image.open(fn)

        if var_name not in var_dict:
            var_dict[var_name] = [None]*N_frames
        print(var_name)
        var_dict[var_name][frame_i] = new_frame

for var_name, frames in var_dict.items():
    # Save all into gif files file that loops forever
    frames[0].save(os.path.join(dir_path, f"{var_name}.gif"), format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=duration, loop=0)
