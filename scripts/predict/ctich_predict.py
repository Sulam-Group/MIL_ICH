import argparse
import os
import sys

import pandas as pd
import torch
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

from dataset import CTICHDataset
from model import HemorrhageDetector

model_dir = os.path.join(root_dir, "models")
data_dir = os.path.join(root_dir, "data")
ctich_dir = os.path.join(data_dir, "CT-ICH")

dataset = CTICHDataset(ctich_dir, return_hemorrhage_type=True)
dataloader = DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
)

if LABEL_COMPLEXITY:
    from utils import label_complexity_models as models
else:
    from utils import models
prediction_file_name = "predictions"

for _, model_name, weak_supervision in models:
    prediction_dir = os.path.join(ctich_dir, "predictions", model_name)
    os.makedirs(prediction_dir, exist_ok=True)

    if not os.path.exists(os.path.join(model_dir, model_name + ".pt")):
        continue

    if LABEL_COMPLEXITY and os.path.exists(
        os.path.join(prediction_dir, prediction_file_name)
    ):
        continue

    print(f"Predicting with model: {model_name}")
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

    predictions = []
    for i, data in enumerate(tqdm(dataloader)):
        series, target, labels = data

        series = series.squeeze()
        series = series.to(device)

        if weak_supervision:
            output = model(series, attention=True, return_aux=True)
            global_logit, attention, embeddings = (
                output["logit"],
                output["attention"],
                output["embeddings"],
            )
            global_prediction = (global_logit >= 0.5).long().item()
            global_logit = global_logit.item()
            single_slice_logits = model.classifier(embeddings)
            attention = attention.squeeze().tolist()
        else:
            single_slice_logits = model(series, attention=False, return_aux=False)
            global_logit = None
            attention = None
            global_prediction = (single_slice_logits >= 0.5).long().sum().item() >= 1

        target = target.long().item()
        single_slice_logits = single_slice_logits.squeeze().tolist()
        labels = labels[0].numpy().tolist()

        predictions.append(
            {
                "series_idx": i,
                "target": target,
                "global_logit": global_logit,
                "global_prediction": global_prediction,
                "labels": labels,
                "attention": attention,
                "single_slice_logits": single_slice_logits,
            }
        )

    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_pickle(os.path.join(prediction_dir, prediction_file_name))
