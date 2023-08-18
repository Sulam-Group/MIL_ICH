import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--weak_supervision", action="store_true", default=None)
args = parser.parse_args()

gpu = args.gpu

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "../../"
sys.path.append(root_dir)

from dataset import CQ500Dataset
from model import HemorrhageDetector
from utils import explainers as exp_mapper
from utils import models

model_dir = os.path.join(root_dir, "models")
data_dir = os.path.join(root_dir, "data")
cq500_dir = os.path.join(data_dir, "CQ500")
image_dir = os.path.join(cq500_dir, "images")
prediction_dir = os.path.join(cq500_dir, "predictions")

plain_thick_series_df = pd.read_csv(os.path.join(cq500_dir, "plain_thick_series.csv"))
dataset = CQ500Dataset(cq500_dir)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

for model_title, model_name, weak_supervision in models:
    if (
        args.weak_supervision is not None
        and args.weak_supervision is not weak_supervision
    ):
        continue

    explanation_dir = os.path.join(cq500_dir, "explanations", model_name)
    os.makedirs(explanation_dir, exist_ok=True)

    prediction_df = pd.read_pickle(
        os.path.join(prediction_dir, model_name, "predictions")
    )
    prediction_df.set_index("series_idx", inplace=True)

    if not weak_supervision:
        global_prediction = prediction_df["single_slice_logits"].apply(
            lambda logits: max(logits)
        )
    else:
        global_prediction = prediction_df["global_logit"]

    pred = global_prediction.tolist()
    target = prediction_df["target"].tolist()

    fpr, tpr, thresholds = metrics.roc_curve(target, pred, pos_label=1)
    d = np.linalg.norm(np.stack((fpr, 1 - tpr), axis=1), axis=1)
    t = thresholds[np.argmin(d)]

    print(f"Explaining with model: {model_name}")
    model_state_dict = torch.load(
        os.path.join(model_dir, model_name + ".pt"), map_location=device
    )
    model = HemorrhageDetector(
        encoder="resnet18",
        n_dim=128,
        hidden_size=64,
        attention_activation="sparsemax",
    )
    model.load_state_dict(model_state_dict)
    model.eval()
    model = model.to(device)
    torch.set_grad_enabled(False)

    for exp in exp_mapper:
        exp["explainer"] = exp["init"](model, cq500_dir, device)
        print(f"Initialized {exp['name']}")

    explained_sop_ids = []
    for series_idx, data in enumerate(dataloader):
        series_row = plain_thick_series_df.iloc[series_idx]
        exam_dir = series_row["exam_dir"]
        series_dir = series_row["series_dir"]
        sorted_ids = np.load(
            os.path.join(image_dir, exam_dir, series_dir, "sorted_ids.npy")
        )
        sop_ids = np.load(os.path.join(image_dir, exam_dir, series_dir, "sop_ids.npy"))
        sop_ids = sop_ids[sorted_ids]

        series_prediction = prediction_df.iloc[series_idx]
        if global_prediction.iloc[series_idx] >= t:
            series, _ = data
            series = series.squeeze()

            if weak_supervision:
                est = np.array(series_prediction["hshap_absolute_0_1"])

                A = sum(est)
                est = [w / A for w in est]

                pred = [True if w >= (1 / len(est)) else False for w in est]
            else:
                est = np.array(series_prediction["single_slice_logits"])
                pred = [True if p >= t else False for p in est]

            sop_ids_to_explain = sop_ids[pred]
            images_to_explain = series[pred]
            for exp in exp_mapper:
                exp_name = exp["name"]
                explainer_dir = os.path.join(explanation_dir, exp_name)
                os.makedirs(explainer_dir, exist_ok=True)
                explainer = exp["explainer"]
                explain = exp["explain"]
                kwargs = exp["kwargs"]

                explained_sop_ids.extend(sop_ids_to_explain)
                np.save(
                    os.path.join(explainer_dir, f"explained_sop_ids.npy"),
                    explained_sop_ids,
                )
                for i, (sop_id, image_t) in enumerate(
                    zip(sop_ids_to_explain, images_to_explain)
                ):
                    image_t = image_t.to(device)
                    image_np = image_t.permute(1, 2, 0).cpu().numpy()

                    t0 = time.time()
                    explanation = explain(explainer, image_t, **kwargs)
                    runtime = round(time.time() - t0, 6)
                    print(
                        f"{exp_name}: {i+1}/{len(sop_ids_to_explain)} ({series_idx}/{len(dataset)}) runtime = {runtime:.4f} s"
                    )
                    np.save(os.path.join(explainer_dir, sop_id[1]), explanation)
