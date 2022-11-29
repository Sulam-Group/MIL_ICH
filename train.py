import os
import argparse
import torch
import wandb
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import HemorrhageDetector
from dataset import RSNADataset

parser = argparse.ArgumentParser()
parser.add_argument("--weak_supervision", action="store_true", default=False)
parser.add_argument("--m", type=int, default=None)
parser.add_argument("--start_idx", type=int, default=0)
parser.add_argument("--R", type=int, default=20)
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

weak_supervision = args.weak_supervision
m = args.m
R = args.R
start_idx = args.start_idx
gpu = args.gpu

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = os.path.join("data")
rsna_dir = os.path.join(data_dir, "RSNA")
if m is not None:
    model_dir = os.path.join(
        "models",
        "label_complexity",
        "weak_supervision" if weak_supervision else "strong_supervision",
        str(m),
    )
else:
    model_dir = os.path.join("models")
os.makedirs(model_dir, exist_ok=True)


def focal_loss(pred, target):
    alpha = 0.60 if weak_supervision else 0.86
    y_hat = pred * (target == 1) + (1 - pred) * (target == 0)
    gamma = 5 * (y_hat < 0.2) + 3 * (y_hat >= 0.2)
    loss = -alpha * (1 - y_hat) ** gamma * torch.log(y_hat)
    return loss.mean()


for r in range(start_idx, R):
    run_name = f"[{gpu}] {'global' if weak_supervision else 'local'}_{m}_{r}"
    wandb.init(
        project="label-complexity-hemorrhage-detection",
        entity="jacopoteneggi",
        name=run_name,
    )

    model = HemorrhageDetector(
        encoder="resnet18",
        n_dim=128,
        hidden_size=64,
        embedding_dropout=0.50,
        attention_dropout=0.25,
        attention_activation="sparsemax",
    )
    model = model.to(device)

    criterion = focal_loss
    optimizer = (
        torch.optim.SGD(
            params=model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4
        )
        if weak_supervision
        else torch.optim.Adam(params=model.parameters(), lr=1e-5, weight_decay=1e-7)
    )
    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.3)

    ops = ["train", "val"]
    datasets = {
        op: RSNADataset(
            data_dir=rsna_dir,
            op=op,
            augment=op == "train",
            weak_supervision=weak_supervision,
            m=m if op == "train" else None,
        )
        for op in ops
    }
    dataloaders = {
        op: DataLoader(
            d,
            batch_size=1 if weak_supervision else 4,
            num_workers=4,
            shuffle=op == "train",
            persistent_workers=True,
        )
        for op, d in datasets.items()
    }

    if m is not None:
        num_epochs = 5000
        patience = 3
        count_no_improve = 0
    else:
        num_epochs = 15

    best_val_accuracy = 0

    running_loss = 0.0
    running_correct = 0
    running_count = 0
    for epoch_idx in range(num_epochs):

        model.train()
        torch.set_grad_enabled(True)

        for data in tqdm(dataloaders["train"]):
            if weak_supervision:
                series, target, _ = data
                series = series.squeeze()
            else:
                series, target = data
            n = target.size(0)

            series = series.to(device)
            target = target.to(device)

            output = model(series, attention=weak_supervision)

            output = output.squeeze()
            target = target.squeeze()

            loss = criterion(output, target)
            prediction = output >= 0.5

            running_loss += loss.detach() * n
            running_correct += torch.sum(prediction == target)
            running_count += n

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if running_count == 52:
                wandb.log(
                    {
                        f"train_loss": running_loss / running_count,
                        f"train_accuracy": running_correct / running_count,
                    }
                )
                running_loss = 0.0
                running_correct = 0
                running_count = 0

        val_step = 1 if (m is None or m > 1000) else 16
        if (epoch_idx + 1) % val_step == 0:
            model.eval()
            torch.set_grad_enabled(False)

            val_loss = 0.0
            val_correct = 0

            for data in tqdm(dataloaders["val"]):
                if weak_supervision:
                    series, target, _ = data
                    series = series.squeeze()
                else:
                    series, target = data
                n = target.size(0)

                series = series.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                output = model(series, attention=weak_supervision)

                output = output.squeeze()
                target = target.squeeze()

                loss = criterion(output, target)
                prediction = output >= 0.5

                val_loss += loss.detach() * n
                val_correct += torch.sum(prediction == target)

            val_loss = val_loss / len(datasets["val"])
            val_accuracy = val_correct / len(datasets["val"])
            wandb.log(
                {
                    f"val_loss": val_loss,
                    f"val_accuracy": val_accuracy,
                }
            )

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                count_no_improve = 0
                best_model = model
                if m is not None:
                    torch.save(
                        best_model.state_dict(), os.path.join(model_dir, f"{str(r)}.pt")
                    )
                else:
                    torch.save(
                        best_model.state_dict(),
                        os.path.join(
                            model_dir,
                            f"{'wl_model' if weak_supervision else 'sl_model'}.pt",
                        ),
                    )
            elif m is not None:
                count_no_improve += 1

        if (m is not None) and (count_no_improve > patience):
            break
        scheduler.step()

    wandb.finish(exit_code=0, quiet=True)
