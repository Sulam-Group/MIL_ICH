import json
import os
import sys
from functools import reduce

import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from sklearn import metrics
from tqdm import tqdm

root_dir = "../../"
sys.path.append(root_dir)
from dataset import CTICHDataset
from utils import explainers, models, window

data_dir = os.path.join(root_dir, "data")
cq500_dir = os.path.join(data_dir, "CQ500")
ctich_dir = os.path.join(data_dir, "CT-ICH")
bhx_dir = os.path.join(data_dir, "BHX")

cq500_image_dir = os.path.join(cq500_dir, "images")
ctich_image_dir = os.path.join(ctich_dir, "images")


def ctich_f1():
    explanation_dirs = [
        os.path.join(ctich_dir, "explanations", model) for _, model, _ in models
    ]

    explained_slice_ids = (
        np.load(
            os.path.join(explanation_dir, explainer["name"], "explained_slice_ids.npy")
        )
        for explanation_dir in explanation_dirs
        for explainer in explainers
    )
    explained_slice_ids = reduce(np.intersect1d, explained_slice_ids)

    dataset = CTICHDataset(ctich_dir)

    diagnosis_df = pd.read_csv(
        os.path.join(ctich_dir, "hemorrhage_diagnosis_raw_ct.csv")
    )
    diagnosis_df.set_index(["PatientNumber", "SliceNumber"], inplace=True)

    annotation_df = pd.read_csv(os.path.join(ctich_dir, "annotations.csv"))
    annotation_df.set_index(["PatientNumber", "SliceNumber"], inplace=True)

    count = 0
    f1_df = []
    for explained_slice_id in tqdm(explained_slice_ids):
        series_idx, slice_idx = explained_slice_id.split("_")

        patient_id = dataset.series[int(series_idx)]
        patient_number = int(patient_id.lstrip("0"))
        slice_idx = int(slice_idx)
        slice_number = slice_idx + 1

        slice_row = diagnosis_df.loc[patient_number, slice_number]

        hemorrhage_types = set()
        for k, t in [
            ("Intraventricular", "IVH"),
            ("Intraparenchymal", "IPH"),
            ("Subarachnoid", "SAH"),
            ("Epidural", "EDH"),
            ("Subdural", "SDH"),
        ]:
            if slice_row[k] == 1:
                hemorrhage_types.add(t)

        if len(hemorrhage_types) > 1:
            hemorrhage_types = set()
            hemorrhage_types.add("Any")
        else:
            hemorrhage_types.add("Any")

        if not int(slice_row["No_Hemorrhage"]):
            count += 1
            image = np.load(
                os.path.join(ctich_image_dir, f"{patient_id}_{slice_idx}.npy")
            )
            image = window(image, window_level=40, window_width=80)

            ground_truth = np.zeros_like(image)
            for _, annotation_row in annotation_df.loc[
                patient_number, slice_number
            ].iterrows():
                annotation = annotation_row["data"].replace("'", '"')
                annotation = json.loads(annotation)
                bbox_x = int(annotation["x"])
                bbox_y = int(annotation["y"])
                bbox_width = int(annotation["width"])
                bbox_height = int(annotation["height"])
                ground_truth[
                    bbox_y : bbox_y + bbox_height, bbox_x : bbox_x + bbox_width
                ] = 1
            ground_truth = ground_truth.flatten()

            for (model_title, _, _), explanation_dir in zip(models, explanation_dirs):
                for i, explainer in enumerate(explainers):
                    explainer_title = explainer["title"]
                    explainer = explainer["name"]

                    explanation = np.load(
                        os.path.join(
                            explanation_dir,
                            explainer,
                            f"{series_idx}_{slice_idx}.npy",
                        )
                    )
                    explanation = explanation.flatten()
                    _t = threshold_otsu(explanation, nbins=1024)

                    (
                        precision,
                        recall,
                        score,
                        _,
                    ) = metrics.precision_recall_fscore_support(
                        ground_truth,
                        explanation > _t,
                        beta=1,
                        pos_label=1,
                        average="binary",
                        zero_division=0,
                    )
                    f1_df.append(
                        {
                            "model": model_title,
                            "explainer": explainer_title,
                            "model_explainer": f"{model_title}, {explainer_title}",
                            "patient_number": patient_number,
                            "slice_idx": slice_idx,
                            "hemorrhage_types": hemorrhage_types,
                            "t": "otsu",
                            "precision": precision,
                            "recall": recall,
                            "f1": score,
                        }
                    )
    f1_df = pd.DataFrame(f1_df)
    f1_df.to_pickle(os.path.join(ctich_dir, "explanations", f"image_level_f1"))
    print(f"f1 score evaluated on {count} images")


def cq500_f1():
    hem_type_dict = {
        "Intraventricular": "IVH",
        "Intraparenchymal": "IPH",
        "Subarachnoid": "SAH",
        "Epidural": "EDH",
        "Subdural": "SDH",
        "Chronic": "SDH",
    }

    plain_thick_series_df = pd.read_csv(
        os.path.join(cq500_dir, "plain_thick_series.csv")
    )
    manual_annotation_df = pd.read_csv(
        os.path.join(bhx_dir, "1_Initial_Manual_Labeling.csv")
    )

    explanation_dirs = [
        os.path.join(cq500_dir, "explanations", model) for _, model, _ in models
    ]

    explained_sop_ids = [
        np.load(
            os.path.join(explanation_dir, explainer["name"], "explained_sop_ids.npy")
        )[:, 1]
        for explanation_dir in explanation_dirs
        for explainer in explainers
    ]
    explained_sop_ids = reduce(np.intersect1d, explained_sop_ids)

    f1_df = []
    count = 0
    for _, series_row in tqdm(
        plain_thick_series_df.iterrows(), total=len(plain_thick_series_df)
    ):
        exam_dir = series_row["exam_dir"]
        series_dir = series_row["series_dir"]
        series_image_dir = os.path.join(cq500_image_dir, exam_dir, series_dir)
        sop_ids = np.load(
            os.path.join(cq500_image_dir, exam_dir, series_dir, "sop_ids.npy")
        )
        for image_name, sop_id in sop_ids:
            if sop_id in explained_sop_ids:
                sop_annotation_df = manual_annotation_df[
                    manual_annotation_df["SOPInstanceUID"] == sop_id
                ]
                if len(sop_annotation_df) > 0:
                    count += 1
                    image = np.load(
                        os.path.join(
                            series_image_dir, image_name.replace(".dcm", ".npy")
                        )
                    )
                    if image.shape != (512, 512):
                        print("Reshape needed")
                        continue
                    image = window(image, window_level=40, window_width=80)

                    hemorrhage_types = set()
                    ground_truth = np.zeros((512, 512))
                    for _, annotation_row in sop_annotation_df.iterrows():
                        annotation = annotation_row["data"].replace("'", '"')
                        annotation = json.loads(annotation)
                        bbox_x = int(annotation["x"])
                        bbox_y = int(annotation["y"])
                        bbox_width = int(annotation["width"])
                        bbox_height = int(annotation["height"])
                        ground_truth[
                            bbox_y : bbox_y + bbox_height, bbox_x : bbox_x + bbox_width
                        ] = 1
                        hemorrhage_types.add(hem_type_dict[annotation_row["labelName"]])
                    if len(hemorrhage_types) > 1:
                        hemorrhage_types = set()
                        hemorrhage_types.add("Any")
                    else:
                        hemorrhage_types.add("Any")
                    ground_truth = ground_truth.flatten()

                    for (model_title, _, _), explanation_dir in zip(
                        models, explanation_dirs
                    ):
                        for i, explainer in enumerate(explainers):
                            explainer_title = explainer["title"]
                            explainer = explainer["name"]

                            explanation = np.load(
                                os.path.join(
                                    explanation_dir,
                                    explainer,
                                    f"{sop_id}.npy",
                                )
                            )
                            explanation = explanation.flatten()
                            _t = threshold_otsu(explanation, nbins=1024)
                            (
                                precision,
                                recall,
                                score,
                                _,
                            ) = metrics.precision_recall_fscore_support(
                                ground_truth,
                                explanation > _t,
                                beta=1,
                                pos_label=1,
                                average="binary",
                                zero_division=0,
                            )
                            f1_df.append(
                                {
                                    "model": model_title,
                                    "explainer": explainer_title,
                                    "model_explainer": f"{model_title}, {explainer_title}",
                                    "sop_id": sop_id,
                                    "hemorrhage_types": list(hemorrhage_types),
                                    "t": "otsu",
                                    "precision": precision,
                                    "recall": recall,
                                    "f1": score,
                                }
                            )
    f1_df = pd.DataFrame(f1_df)
    f1_df.to_pickle(os.path.join(cq500_dir, "explanations", "image_level_f1"))
    print(f"f1 score evaluated on {count} images")
