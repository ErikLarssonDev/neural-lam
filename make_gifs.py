from PIL import Image
import glob
import os

N_frames = 100  # Number of frames per GIF
duration = 100  # Duration per frame in ms
EXPERIMENT_NAME = "IR-SDE-600e-50-steps_2-T=[0,1]"  

# Define the directory to save GIFs
dir_path = f"gifs/{EXPERIMENT_NAME}"
os.makedirs("gifs", exist_ok=True)
os.makedirs(dir_path, exist_ok=True)

# Path to search for images recursively
img_dir = "/proj/berzelius-2022-164/users/x_erila/neural-lam/diffusion_steps"
img_fnames = glob.glob(os.path.join(img_dir, "**/state_*.png"), recursive=True)

var_dict = {}

# Process image filenames
for fn in img_fnames:
    path_parts = fn.split(os.sep)
    if len(path_parts) >= 2:
        var_name = os.path.join(path_parts[-3], path_parts[-2])  # Use "sde_state/u_850" as variable name
    else:
        var_name = "unknown"
    
    frame_i = int(os.path.basename(fn).split("_")[1].split(".")[0])  # Extract frame number

    new_frame = Image.open(fn)

    if var_name not in var_dict:
        var_dict[var_name] = [None] * N_frames
    var_dict[var_name][frame_i - 1] = new_frame  # Adjust index to 0-based

# Create and save GIFs
for var_name, frames in var_dict.items():
    # Remove None values and reverse frame order
    frames = [frame for frame in frames if frame is not None][::-1]
    
    if frames:
        safe_var_name = var_name.replace(os.sep, "_")  # Ensure valid filename
        frames[0].save(os.path.join(dir_path, f"{safe_var_name}.gif"), format='GIF',
                       append_images=frames[1:], save_all=True,
                       duration=duration, loop=0)
