from scipy.stats import wasserstein_distance_nd
import torch
import time
tensor_1 = torch.randn(17, 268, 238)
tensor_2 = torch.randn(17, 268, 238)
from geomloss import SamplesLoss
loss = SamplesLoss("sinkhorn", p=2)
start_time = time.time()

# distance = wasserstein_distance_nd(tensor_1, tensor_2)
distance = loss(tensor_1, tensor_2)
# distance = torch.nn.functional.mse_loss(tensor_1, tensor_2)
end_time = time.time()
print(distance)
print("Elapsed time: ", (end_time - start_time))