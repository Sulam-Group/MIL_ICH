import argparse
import os
import sys

import hshap
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--label_complexity", action="store_true", default=False)
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

gpu = args.gpu
LABEL_COMPLEXITY = args.label_complexity

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "../../"
sys.path.append(root_dir)

from dataset import RSNADataset
from model import HemorrhageDetector
from utils import label_complexity_models, models, series_explainers

if LABEL_COMPLEXITY:
    models = label_complexity_models
    val_subset = np.load(
        os.path.join(root_dir, "scripts", "predict", "rsna_val_subset.npy")
    )
    prediction_file_name = "predictions_fixed"
else:
    prediction_file_name = "predictions"

model_dir = os.path.join(root_dir, "models")
data_dir = os.path.join(root_dir, "data")
prediction_dir = os.path.join(data_dir, "predictions")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class f(nn.Module):
    def __init__(self, model):
        super(f, self).__init__()
        self.model = model

    def forward(self, x):
        A = self.model.attention_activation(
            self.model.attention_mechanism(x).t(), dim=1
        )
        z = torch.mm(A, x)
        x = self.model.classifier(z)
        return x


dataset = RSNADataset(
    data_dir=data_dir,
    op="val",
    augment=False,
    weak_supervision=True,
    m=None,
    idx=val_subset if LABEL_COMPLEXITY else None,
)
dataloader = DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
)

for _, model_name, weak_supervision in models:
    if not weak_supervision:
        continue

    if LABEL_COMPLEXITY:
        m = int(model_name.split("/")[-2])

    if not os.path.exists(
        os.path.join(model_dir, model_name + ".pt")
    ) or not os.path.exists(
        os.path.join(prediction_dir, model_name, prediction_file_name)
    ):
        continue

    print(f"Explaining series with model: {model_name}")
    model_state_dict = torch.load(
        os.path.join(model_dir, model_name + ".pt"), map_location=device
    )
    model = HemorrhageDetector(
        encoder="resnet18",
        n_dim=128,
        hidden_size=64,
        embedding_dropout=0.50,
        attention_dropout=0.25,
        attention_activation="sparsemax",
    )
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    _f = f(model)
    empty_output = torch.zeros(1, 1, device=device)
    bag_hexp = hshap.src.BagExplainer(model=_f, empty_output=empty_output)

    prediction_df = pd.read_pickle(
        os.path.join(prediction_dir, model_name, prediction_file_name)
    )
    predictions = prediction_df.to_dict(orient="records")

    for _, explainer in series_explainers["weak_supervision"]:
        if "hshap" not in explainer:
            continue

        print(explainer)
        _, threshold_mode, threshold, s = explainer.split("_")
        threshold = int(threshold)
        s = int(s)

        if explainer in prediction_df.columns:
            continue

        for series_idx, data in enumerate(tqdm(dataloader)):
            row = predictions[series_idx]

            series, target, _ = data

            series = series.squeeze()
            series = series.to(device)

            output = model(series, attention=True, return_aux=True)
            H = output["embeddings"]

            series_explanation = bag_hexp.explain(
                bag=H,
                label=0,
                s=s,
                threshold_mode=threshold_mode,
                threshold=threshold,
                softmax_activation=False,
                binary_map=False,
            )
            row[explainer] = series_explanation.tolist()

    prediction_df = pd.DataFrame(predictions)
    pd.to_pickle(
        prediction_df, os.path.join(prediction_dir, model_name, prediction_file_name)
    )
