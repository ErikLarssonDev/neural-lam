# test_model_memory.py

import torch
import json
from argparse import ArgumentParser
from neural_lam.models.graph_efm import GraphEFM
from neural_lam.models.graph_fm import GraphFM
from neural_lam.models.graphcast import GraphCast
from neural_lam.models.diffusion import Diffusion
from neural_lam.models.ir_sde import IR_SDE
from neural_lam.models.stochastic_interpolants import SI
from neural_lam.models.unet import UNET
from neural_lam.models.CorrDiff import CorrDiff
from neural_lam.models.edm_networks_2 import SongUNet

MODELS = {
    "graphcast": GraphCast,
    "graph_fm": GraphFM,
    "graph_efm": GraphEFM,
    "diffusion": Diffusion,
    "ir_sde": IR_SDE,
    "SI": SI,
    "unet": UNET,
    "CorrDiff": CorrDiff,
}


def main():
    device = "cuda"
    print(f"Using device: {device}")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    HIDDEN_DIM = 256
    IN_CHANNELS = 13 + 100
    OUT_CHANNELS = 13
    GRID_SHAPE = [400, 550]

    # Instantiate model
    model = SongUNet(img_resolution=torch.as_tensor(GRID_SHAPE),
                     in_channels=IN_CHANNELS,
                     out_channels=OUT_CHANNELS,
                     model_channels=HIDDEN_DIM,
                     target_idx=torch.arange(OUT_CHANNELS),
                     )
    model.to(device)
    model.train()

    # Fake input
    input_tensor = torch.randn(
        [1, IN_CHANNELS, *GRID_SHAPE], device=device)
    z = torch.ones(
        (input_tensor.shape[0]), device=input_tensor.device)

    # Forward + backward step
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    try:
        optimizer.zero_grad()
        output = model(input_tensor, z)
        # Fake loss: just mean of output
        loss = output.mean()
        loss.backward()
        optimizer.step()
        print("Success! Model fits input tensor without OOM.")
    except RuntimeError as e:
        print(f"Error during forward/backward: {e}")

    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
    reserved_mem = torch.cuda.max_memory_reserved(device) / 1024**3

    print(f"Peak allocated memory: {peak_mem:.2f} GB")
    print(f"Peak reserved memory:  {reserved_mem:.2f} GB")


if __name__ == "__main__":
    main()

# python3 neural_lam/test_memory.py --model CorrDiff --n_workers 4 --batch_size 1 --epochs 1 --lr 0.00001 --pred_residual --hidden_dim 32 --
