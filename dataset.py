import os
import json
import torch
import torchvision.transforms as t
import torchio.transforms as tio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from random import shuffle
from utils import window


class RSNADataset(Dataset):
    def __init__(
        self,
        data_dir,
        op,
        augment,
        weak_supervision,
        m=None,
        idx=None,
    ):
        image_path = os.path.join(data_dir, f"{op}_images.json")
        series_path = os.path.join(data_dir, f"{op}_series.json")
        self.series_dir = os.path.join(data_dir, "series")
        self.dicom_dir = os.path.join(data_dir, "stage_2_train")
        with open(image_path, "r", encoding="utf-8") as image_f:
            self.images = json.load(image_f)
            if op == "train":
                from utils import bad_image_idx

                self.images = np.delete(self.images, bad_image_idx, axis=0).tolist()
            if not weak_supervision and m is not None:
                # separate positive and negative images
                p_idx = [idx for idx, image in enumerate(self.images) if int(image[1])]
                n_idx = [
                    idx for idx, image in enumerate(self.images) if not int(image[1])
                ]
                # compute positive to negative ratio
                a = len(p_idx) / len(n_idx)
                # sample subset with m labels
                mp = int(m * a)
                p_idx = np.random.choice(p_idx, mp, replace=False).tolist()
                n_idx = np.random.choice(n_idx, m - mp, replace=False).tolist()
                # shuffle subset
                s_idx = p_idx + n_idx
                shuffle(s_idx)
                self.images = [self.images[idx] for idx in s_idx]
        with open(series_path, "r", encoding="utf-8") as series_f:
            self.series_dictionary = json.load(series_f)
            if weak_supervision and m is not None:
                # separate positive and negative series
                p_ids = [
                    k
                    for k, series in self.series_dictionary.items()
                    if int(series["target"])
                ]
                n_ids = [
                    k
                    for k, series in self.series_dictionary.items()
                    if not int(series["target"])
                ]
                # compute positive to negative ratio
                a = len(p_ids) / len(n_ids)
                # sample subset with m labels
                mp = int(m * a)
                p_ids = np.random.choice(p_ids, mp, replace=False).tolist()
                n_ids = np.random.choice(n_ids, m - mp, replace=False).tolist()
                # shuffle subset
                s_ids = p_ids + n_ids
                shuffle(s_ids)
                self.series_dictionary = {k: self.series_dictionary[k] for k in s_ids}
            if weak_supervision and idx is not None:
                self.series_dictionary = {k: self.series_dictionary[k] for k in idx}
        self.series_ids = list(self.series_dictionary.keys())
        self.transform = tio.Compose(
            [
                tio.RandomFlip(axes=(0, 1), p=0.5),
                tio.RandomAffine(
                    scales=(0.1, 0.05, 0.0),
                    translation=(1, 2, 1, 2, 0, 0),
                    degrees=(0, 0, 0, 0, 10, 20),
                ),
                tio.RandomAffine(
                    scales=(0.05, 0.1, 0.0),
                    translation=(1, 2, 1, 2, 0, 0),
                    degrees=(0, 0, 0, 0, 10, 20),
                ),
                tio.OneOf(
                    [
                        tio.RandomNoise(mean=0, std=0.4),
                        tio.RandomBiasField(),
                        tio.RandomAnisotropy(axes=(0, 1), downsampling=(4, 7)),
                        tio.RandomGamma(log_gamma=0.12),
                        tio.RandomGhosting(num_ghosts=(3, 5), intensity=0.7),
                    ]
                ),
            ]
        )
        self.MEAN = torch.tensor([0.485, 0.456, 0.406])
        self.STD = torch.tensor([0.229, 0.224, 0.225])
        self.normalize = t.Normalize(
            mean=self.MEAN,
            std=self.STD,
        )
        self.augment = augment
        self.weak_supervision = weak_supervision

    def __len__(self):
        return len(self.series_ids) if self.weak_supervision else len(self.images)

    def __getitem__(self, idx):
        if self.weak_supervision:
            series_id = self.series_ids[idx]
            series_obj = self.series_dictionary[series_id]

            series = series_obj["series"]
            target = series_obj["target"]

            series = np.array(series)
            labels = series[:, 1]

            series = torch.load(os.path.join(self.series_dir, series_id)).float()
            target = torch.Tensor([int(target)]).float()
            labels = torch.Tensor(labels.astype(int)).float()

            series = window(series, window_level=40, window_width=80)
            series = series.unsqueeze(1)
            series = series.repeat(1, 3, 1, 1)

            if self.augment:
                # sample a random subset of the scan
                series_length = len(series)
                uniform = torch.ones(series_length) / series_length
                subset_length = torch.randint(10, series_length, (1,)).item()
                subset_idx = torch.multinomial(
                    uniform, subset_length, replacement=False
                )

                series = series[subset_idx]
                labels = labels[subset_idx]

                # augment each slice in the scan
                with torch.no_grad():
                    for i, image in enumerate(series):
                        series[i] = (
                            self.transform(image.unsqueeze(0).permute(1, 2, 3, 0))
                            .permute(3, 0, 1, 2)
                            .squeeze()
                        )
            else:
                sorted_ids = np.load(
                    os.path.join(self.series_dir, f"{series_id}_sorted_ids.npy")
                )
                series = series[sorted_ids]
                labels = labels[sorted_ids]

            series = self.normalize(series)
            return series, target, labels

        else:
            image_data = self.images[idx]

            image_id = image_data[0]
            image = np.load(
                os.path.join(self.dicom_dir, f"ID_{image_id}.npy"), allow_pickle=True
            )
            image = torch.Tensor(image).float()
            image = image.squeeze()
            image = window(image, window_level=40, window_width=80)
            image = image.repeat(3, 1, 1)
            if self.augment:
                image = (
                    self.transform(image.unsqueeze(0).permute(1, 2, 3, 0))
                    .permute(3, 0, 1, 2)
                    .squeeze()
                )
            image = self.normalize(image)

            label = torch.Tensor([int(image_data[1])]).float()
            return image, label


class CQ500Dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_dir = os.path.join(self.data_dir, "images")
        self.series_dir = os.path.join(self.data_dir, "series")
        self.plain_thick_series_df = pd.read_csv(
            os.path.join(self.data_dir, "plain_thick_series.csv")
        )
        self.MEAN = torch.tensor([0.485, 0.456, 0.406])
        self.STD = torch.tensor([0.229, 0.224, 0.225])
        self.resize = t.Resize((512, 512))
        self.normalize = t.Normalize(
            mean=self.MEAN,
            std=self.STD,
        )

    def __len__(self):
        return len(self.plain_thick_series_df)

    def __getitem__(self, idx):
        row = self.plain_thick_series_df.iloc[idx]
        series_id = row["series_id"]
        target = row["global_label"]
        exam_dir = row["exam_dir"]
        series_dir = row["series_dir"]

        series = torch.load(os.path.join(self.series_dir, series_id)).float()
        target = torch.Tensor([int(target)]).float()

        sorted_ids = np.load(
            os.path.join(self.image_dir, exam_dir, series_dir, "sorted_ids.npy")
        )
        series = series[sorted_ids]

        series = window(series, window_level=40, window_width=80)
        series = series.unsqueeze(1).repeat(1, 3, 1, 1)
        series = self.resize(series)
        series = self.normalize(series)

        return series, target


class CTICHDataset(Dataset):
    def __init__(self, data_dir, return_hemorrhage_type=False):
        self.data_dir = data_dir
        self.series_dir = os.path.join(self.data_dir, "series")
        self.series = os.listdir(self.series_dir)
        self.diagnosis = pd.read_csv(
            os.path.join(self.data_dir, "hemorrhage_diagnosis_raw_ct.csv")
        )
        self.MEAN = torch.tensor([0.485, 0.456, 0.406])
        self.STD = torch.tensor([0.229, 0.224, 0.225])
        self.normalize = t.Normalize(
            mean=self.MEAN,
            std=self.STD,
        )
        self.return_hemorrhage_type = return_hemorrhage_type

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        series_id = self.series[idx]
        patient_number = series_id.lstrip("0")

        series = torch.load(os.path.join(self.series_dir, series_id)).float()

        inv_labels = self.diagnosis[
            self.diagnosis["PatientNumber"] == int(patient_number)
        ]["No_Hemorrhage"].tolist()
        labels = [not l for l in inv_labels]
        target = sum(labels) > 0

        labels = torch.Tensor(labels).float()
        target = torch.Tensor([target]).float()

        series = window(series, window_level=40, window_width=80)
        series = series.unsqueeze(1).repeat(1, 3, 1, 1)
        series = self.normalize(series)
        assert series.size(0) == len(labels)

        return series, target, labels
