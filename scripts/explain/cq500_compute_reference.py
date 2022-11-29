import os
import sys
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

root_dir = "../../"
sys.path.append(root_dir)
from dataset import CQ500Dataset

data_dir = os.path.join(root_dir, "data")
cq500_dir = os.path.join(data_dir, "CQ500")

dataset = CQ500Dataset(cq500_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

running_sum = torch.zeros(3, 512, 512)
count = 0
for series, _ in tqdm(dataloader):
    series = series.squeeze()
    count += series.size(0)
    series_sum = series.sum(0)
    running_sum += series_sum
mean = running_sum / count
torch.save(mean, os.path.join(cq500_dir, "explanations", "reference.pt"))

unnorm_mean = mean * dataset.STD[:, None, None] + dataset.MEAN[:, None, None]

_, ax = plt.subplots(figsize=(5, 5))
ax.imshow(unnorm_mean.permute(1, 2, 0), cmap="gray")
ax.axis("off")
plt.savefig(
    os.path.join(cq500_dir, "explanations", "reference.png"), bbox_inches="tight"
)
plt.close()
