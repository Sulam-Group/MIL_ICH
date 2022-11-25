import os
import numpy as np
import pandas as pd
import torch
import hshap
import matplotlib.patches as patches
import json
from pytorch_grad_cam import GradCAM
from itertools import chain
from sklearn import metrics


def annotate(annotation_df, ax):
    for _, annotation_row in annotation_df.iterrows():
        annotation = annotation_row["data"].replace("'", '"')
        annotation = json.loads(annotation)
        bbox_x = annotation["x"]
        bbox_y = annotation["y"]
        bbox_width = annotation["width"]
        bbox_height = annotation["height"]

        bbox = patches.Rectangle(
            (bbox_x, bbox_y),
            bbox_width,
            bbox_height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(bbox)


def window(image, window_level, window_width):
    image_min = window_level - window_width // 2
    image_max = window_level + window_width // 2
    image[image < image_min] = image_min
    image[image > image_max] = image_max
    image = (image - image_min) / (image_max - image_min)
    return image


models = [
    (
        r"$\mathcal{SL}$",
        "sl_model",
        False,
    ),
    (
        r"$\mathcal{WL}$",
        "wl_model",
        True,
    ),
]

M = [
    12,
    24,
    40,
    52,
    64,
    80,
    100,
    152,
    200,
    252,
    520,
    796,
    1000,
    10000,
    17000,
]


def R(m):
    if m <= 252:
        return 20
    if m == 520:
        return 15
    if m in [796, 1000]:
        return 10
    if m in [3000, 6000, 10000]:
        return 6
    if m >= 17000:
        return 1


label_complexity_models = [
    (
        r"$\mathcal{WL}$" if weak_supervision else r"$\mathcal{SL}$",
        f"label_complexity/{'weak' if weak_supervision else 'strong'}_supervision/{m}/{r}",
        weak_supervision,
    )
    for weak_supervision in [False, True]
    for m in M
    for r in range(R(m))
] + [
    (r"$\mathcal{SL}$", "label_complexity/strong_supervision/600000/0", False),
]

series_explainers = {
    "weak_supervision": [
        (r"attention weights", "attention_r"),
        (r"h-Shap", "hshap_absolute_0_2"),
    ],
    "full_supervision": [("single-slice logits", "single_slice_logits")],
}


def find_hemorrhages(labels):
    hemorrhages = []
    hemorrhage = False
    for i, l in enumerate(labels):
        if l == 1:
            if not hemorrhage:
                hemorrhage = True
                hemorrhages.append([i])
            else:
                hemorrhages[-1].append(i)
        if l == 0 and hemorrhage:
            hemorrhage = False
    return hemorrhages


def hits(row, estimator, offset):
    true_hemorrhages = row["true_hemorrhages"]
    pred_hemorrhages = row["pred_hemorrhages"]
    est = row[estimator]

    est_pred_hemorrhages = [
        est[pred_hemorrhage[0] : (pred_hemorrhage[-1] + 1)]
        for pred_hemorrhage in pred_hemorrhages
    ]
    top_slices = [
        pred_hemorrhage[np.argmax(est_pred_hemorrhage)]
        for pred_hemorrhage, est_pred_hemorrhage in zip(
            pred_hemorrhages, est_pred_hemorrhages
        )
    ]

    assert len(pred_hemorrhages) == len(top_slices)
    hits = set()
    true_hemorrhage_length = [len(h) for h in true_hemorrhages]
    hit = []
    for true_hemorrhage in true_hemorrhages:
        miss = True
        for i, n in enumerate(top_slices):
            if (n >= true_hemorrhage[0] - offset) and (
                n <= true_hemorrhage[-1] + offset
            ):
                hits.add(i)
                miss = False
        hit.append(int(not miss))
    n_correct = sum(hit)
    n_predicted = n_correct + len(set(range(len(pred_hemorrhages))) - hits)

    return (
        true_hemorrhage_length,
        hit,
        est_pred_hemorrhages,
        top_slices,
        len(true_hemorrhages),
        offset,
        n_correct,
        n_predicted,
    )


def location_performance_t(
    model_title,
    prediction_df,
    weak_supervision,
    cutoff="d",
    offset=0,
    min_length=None,
    m=None,
):
    if not weak_supervision:
        pred = prediction_df["single_slice_logits"].tolist()
        pred = list(chain.from_iterable(pred))

        target = prediction_df["labels"].tolist()
        target = list(chain.from_iterable(target))
    else:
        pred = prediction_df["global_logit"].tolist()
        target = prediction_df["target"].tolist()

    fpr, tpr, thresholds = metrics.roc_curve(target, pred, pos_label=1)

    if cutoff == "d":
        d = np.linalg.norm(np.stack((fpr, 1 - tpr), axis=1), axis=1)
        t = thresholds[np.argmin(d)]
    elif cutoff == "youden":
        j = tpr - fpr
        t = thresholds[np.argmax(j)]
    else:
        raise ValueError(f"Unknown cutoff: {cutoff}")

    df = []
    _estimators = series_explainers[
        "weak_supervision" if weak_supervision else "full_supervision"
    ]
    for estimator_title, estimator in _estimators:
        if not weak_supervision and min_length == None:
            min_length = 4

        if weak_supervision and "attention" in estimator:
            if min_length == None:
                min_length = 2

            threshold = estimator.split("_")[-1]
            if threshold == "0":
                est_threshold = lambda _: 0
            if threshold == "r":
                est_threshold = lambda row: 1 / len(row["labels"])
            estimator = "attention"

        if weak_supervision and "hshap" in estimator:
            if min_length == None:
                min_length = 3

            if estimator not in prediction_df.columns:
                continue
            est_threshold = lambda row: 1 / len(row["labels"])

        def _normalize(est):
            if not weak_supervision:
                return est

            if "hshap" not in estimator:
                return est

            A = sum(est)
            est = [w / A for w in est]
            return est

        estimator_df = prediction_df.apply(
            lambda row: {
                "model_title": model_title,
                "estimator_title": estimator_title,
                "est_name": f"{model_title} ({estimator_title})",
                "min_length": min_length,
                "t": t,
                "m": m,
                "true_hemorrhages": find_hemorrhages(row["labels"]),
                "pred_hemorrhages": list(
                    filter(
                        lambda s: len(s) >= min_length,
                        find_hemorrhages(
                            [
                                1
                                if w >= (est_threshold(row) if weak_supervision else t)
                                else 0
                                for w in _normalize(row[estimator])
                            ]
                        ),
                    )
                )
                if (weak_supervision and row["global_logit"] >= t)
                or (not weak_supervision and max(row["single_slice_logits"]) >= t)
                else [],
                estimator: row[estimator],
            },
            axis=1,
            result_type="expand",
        )
        estimator_df[
            [
                "true_hemorrhage_length",
                "hit",
                "est_pred_hemorrhages",
                "top_slices",
                "P",
                "offset",
                "TP",
                "PP",
            ]
        ] = estimator_df.apply(
            lambda row: hits(row, estimator, offset=offset),
            axis=1,
            result_type="expand",
        )
        estimator_df["precision"] = estimator_df["TP"] / (estimator_df["PP"] + 1e-8)
        estimator_df["recall"] = estimator_df["TP"] / (estimator_df["P"] + 1e-8)
        estimator_df["f1"] = estimator_df.apply(
            lambda row: np.NaN
            if (row["P"] == 0) and (row["PP"] == 0)
            else 2
            * row["precision"]
            * row["recall"]
            / (row["precision"] + row["recall"] + 1e-8),
            axis=1,
        )
        estimator_df = estimator_df[estimator_df["f1"].notna()]
        df.append(estimator_df)
    return pd.concat(df)


def cam_init(model, *args):
    cam = GradCAM(model=model, target_layers=[model.encoder.layer4[-1]], use_cuda=True)
    return cam


def hexp_init(model, data_dir, device):
    ref = torch.load(
        os.path.join(data_dir, "explanations", "reference.pt"), map_location=device
    )
    hexp = hshap.src.Explainer(model=model, background=ref)
    return hexp


def cam_explain(cam, image_t, **kwargs):
    torch.set_grad_enabled(True)
    image_t = image_t.unsqueeze(0)
    explanation = cam(input_tensor=image_t)
    torch.set_grad_enabled(False)
    return explanation.squeeze()


def hexp_explain(hexp, image_t, **kwargs):
    R = kwargs["R"]
    A = kwargs["A"]
    s = kwargs["s"]
    threshold_mode = kwargs["threshold_mode"]
    threshold_value = kwargs["threshold_value"]

    torch.set_grad_enabled(False)
    explanation = torch.zeros(1, image_t.size(1), image_t.size(2))

    for r in R:
        for a in A:
            if r == 0 and a != 0:
                continue
            else:
                roll_row = -int(r * np.sin(a))
                roll_column = int(r * np.cos(a))

                saliency_map = hexp.explain(
                    image_t.clone(),
                    label=0,
                    s=s,
                    threshold_mode=threshold_mode,
                    threshold=threshold_value,
                    softmax_activation=False,
                    batch_size=2,
                    binary_map=False,
                    roll_row=roll_row,
                    roll_column=roll_column,
                    attention=False,
                    return_aux=False,
                )
                explanation += saliency_map
    return explanation.squeeze() / (len(R) * len(A))


image_explainers = [
    {
        "title": "GRAD-CAM",
        "name": "gradcam",
        "init": cam_init,
        "explain": cam_explain,
        "kwargs": {},
    },
    {
        "title": r"h-Shap",
        "name": "hexp/absolute_0_64_cycle_3_12",
        "init": hexp_init,
        "explain": hexp_explain,
        "kwargs": {
            "threshold_mode": "absolute",
            "threshold_value": 0.0,
            "s": 64,
            "R": np.linspace(0, 64, 4),
            "A": np.linspace(0, 2 * np.pi, 12),
        },
    },
]

bad_image_idx = [
    11933,
    13167,
    14786,
    18889,
    23057,
    27930,
    28280,
    30219,
    31099,
    49547,
    51148,
    54803,
    60439,
    60664,
    66835,
    70724,
    71545,
    73302,
    80326,
    87191,
    92941,
    95408,
    98876,
    102752,
    103300,
    107578,
    112766,
    120811,
    122325,
    122631,
    124568,
    129830,
    131375,
    134394,
    135825,
    140869,
    141009,
    141083,
    153031,
    165440,
    173385,
    193620,
    195753,
    201277,
    208877,
    219064,
    220852,
    225441,
    227901,
    236187,
    240481,
    241565,
    247716,
    250227,
    261277,
    264452,
    265789,
    268057,
    269724,
    272157,
    283192,
    300121,
    305078,
    311228,
    314908,
    315030,
    316886,
    341616,
    342265,
    343591,
    346422,
    349896,
    355273,
    377314,
    377452,
    379891,
    383257,
    393672,
    395247,
    405499,
    411411,
    413143,
    416566,
    421182,
    423681,
    429387,
    436461,
    437250,
    437405,
    444053,
    446739,
    447970,
    453793,
    459729,
    461317,
    461641,
    474199,
    475867,
    479760,
    492856,
    497749,
    499633,
    505237,
    514100,
    516935,
    518755,
    521637,
    524714,
    530357,
    530500,
    537159,
    545155,
    550627,
    550817,
    561374,
    562532,
    563396,
    563525,
    578539,
    578874,
    586089,
    594891,
    599572,
]
