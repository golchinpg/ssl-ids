from tqdm.auto import tqdm
import argparse
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from models import SCARF
from utils import NTXent, store_pandas_df
from datasets import ExampleDataset, get_dataset


def train_epoch(model, criterion, train_loader, optimizer, epoch, device):  # device,
    model.train()
    epoch_loss = 0.0
    batch = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for anchor, positive, _ in batch:
        anchor, positive = anchor.to(device), positive.to(device)  # .to(device)
        optimizer.zero_grad()
        emb_anchor, emb_positive = model(anchor, positive)
        loss = criterion(emb_anchor, emb_positive)
        loss.backward()
        optimizer.step()
        epoch_loss += anchor.size(0) * loss.item()
        batch.set_postfix({"loss": loss.item()})
    return epoch_loss / len(train_loader.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train SCARF")
    parser.add_argument("--path",  default="/Users/pegah/Desktop/KOM/Datasets/preprocessed_csv/allattack_mondaybenign.csv", type=str,)
    parser.add_argument("--batch_size", default=128, type=int,)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--embedding_dim", default=45, type=int)
    parser.add_argument("--corruption_rate", default=0.6, type=float)
    parser.add_argument(
        "--chkpt_path",
        type=str,
    )
    args = parser.parse_args()
    (
        x_train_normal,
        x_test_normal,
        x_attack,
        y_train_normal,
        y_test_normal,
        y_attack,
    ) = get_dataset(args.path, separate_norm_attack=True, test_size=0.3)
    train_ds = ExampleDataset(
        x_train_normal.to_numpy(),
        y_train_normal.to_numpy(),
        columns=x_train_normal.columns,
    )
    store_pandas_df(pd.concat([x_train_normal, y_train_normal], axis=1), "train_normal.csv")
    store_pandas_df(pd.concat([x_test_normal, y_test_normal], axis=1), "test_normal.csv")
    store_pandas_df(pd.concat([x_attack, y_attack], axis=1), "attack.csv")
    print("Train and Test Datasets are stored in the current directory")

    print("start training ...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    log_name = f"scarf1_embedding_dim={args.embedding_dim}_corruption_rate={args.corruption_rate}_lr={args.lr}_batch_size={args.batch_size}_epochs={args.epochs}"
    writer = SummaryWriter(f"logs/f{log_name}")
    args.input_dim = train_ds.shape[1]
    model = SCARF(
        input_dim=args.input_dim,
        emb_dim=args.embedding_dim,
        corruption_rate=args.corruption_rate,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    ntxent_loss = NTXent().to(device)

    loss_history = []
    for epoch in range(1, args.epochs + 1):
        epoch_loss = train_epoch(
            model, ntxent_loss, train_loader, optimizer, epoch, device
        )
        loss_history.append(epoch_loss)
        writer.add_scalar(
            "Training Loss modified scarf",
            epoch_loss,
            global_step=epoch,
            walltime=0.001,
        )
        print("saving model")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss_history": loss_history,
                'args': args,
                'train_data_columns': train_ds.columns + ['Label']
            },
            f'/Users/pegah/Desktop/KOM/GitHub/ssl-ids/checkpoints/{log_name}.pth',
        )
