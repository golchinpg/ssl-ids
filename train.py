import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from tqdm.auto import tqdm
import argparse
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from models import SCARF
from utils import NTXent, store_pandas_df, load_pandas_df, concatenate_datasets
from datasets import ExampleDataset, get_dataset

# Get the directory of the current script (train.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Define the path to the new_checkpoint folder
checkpoint_dir = os.path.join(current_dir, 'new_checkpoints')
# Create the new_checkpoints folder if it doesn't exist
os.makedirs(checkpoint_dir, exist_ok=True)


def train_epoch(model, criterion, train_loader, optimizer, epoch, device):  # device,
    model.train()
    epoch_loss = 0.0
    batch = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for anchor, rnd_sample_1, rnd_sample_2, _ in batch:
        anchor, rnd_sample_1, rnd_sample_2 = anchor.to(device), rnd_sample_1.to(device), rnd_sample_2.to(device)  # .to(device)
        optimizer.zero_grad()
        emb_anchor, emb_positive = model(anchor, rnd_sample_1, rnd_sample_2)
        loss = criterion(emb_anchor, emb_positive)
        loss.backward()
        optimizer.step()
        epoch_loss += anchor.size(0) * loss.item()
        batch.set_postfix({"loss": loss.item()})
    return epoch_loss / len(train_loader.dataset)


def train(args):
    aug_info = f"cr_rt={args.corruption_rate}_ach_cr_rt{args.anchor_corruption_rate}_msk_rt{args.mask_rate}_ach_msk_rt{args.anchor_mask_rate}"
    log_name = f"scarf1_embdd_dim={args.embedding_dim}_lr={args.lr}_bs={args.batch_size}_epochs={args.epochs}_tempr={args.temprature}_V={args.version}_{aug_info}"
    writer = SummaryWriter(f"new_logs/f{log_name}") 
    model = SCARF(
        input_dim=args.input_dim,
        emb_dim=args.embedding_dim,
        corruption_rate=args.corruption_rate,
        anchor_corruption_rate=args.anchor_corruption_rate,
        mask_rate=args.mask_rate,
        anchor_mask_rate=args.anchor_mask_rate
    )  
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    ntxent_loss = NTXent(temperature=args.temprature).to(device)

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
            os.path.join(checkpoint_dir, f'{log_name}.pth')
        )    



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train SCARF")
    parser.add_argument("--dataset_path",  default="/home/pegah/Codes/ssl-ids/Dataset/merged_1-6.csv", type=str,)
    parser.add_argument("--batch_size", default=2046, type=int,)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--embedding_dim", default=45, type=int)
    parser.add_argument("--corruption_rate", default=0.3, type=float)
    parser.add_argument("--temprature", default=0.5, type=float)
    parser.add_argument("--version", default='onlyunsw', type=str)
    parser.add_argument("--positive_cr_list", default=[0.4], type=list , help="list of corruption rates to create positive pair")
    parser.add_argument("--anchor_cr_list", default= [0.2], type = list, help="list of rates for creating currepted anchor")
    parser.add_argument("--positive_mr_list", default = [0], type = list,  help="list of positive mask rates")
    parser.add_argument("--anchor_mr_list", default = [0], type=list,  help="list of anchor mask rates")
    
    parser.add_argument(
        "--chkpt_path",
        type=str,
    )
    args = parser.parse_args()
    #concatenated_df_normal.csv
    print('batchsize:', args.batch_size)
    
    print(f"training with only {args.dataset_path.split('/')[-1]} benign flows ...")
    x_train_normal,x_test_normal,_,y_train_normal,y_test_normal,_ = get_dataset(
        args.dataset_path, training_with_attacks= True,
        separate_norm_attack= True, test_size=0.3)
    print(y_train_normal.value_counts())
    #create torch dataset
    train_ds = ExampleDataset(
        x_train_normal.to_numpy(),
        y_train_normal.to_numpy(),
        columns=x_train_normal.columns,
    )
    #Create the tmp_dir
    tmp_dir = os.path.join(current_dir, 'tmp_folder/')
    # Create the tmp_folder if it doesn't exist
    os.makedirs(tmp_dir, exist_ok=True)
    print(tmp_dir+str(args.dataset_path.split('/')[-1]))
    store_pandas_df(x_train_normal, tmp_dir+str(args.dataset_path.split('/')[-1].split('.')[0])+"_training_normalflows.csv") 

    print("start training ...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    args.input_dim = train_ds.shape[1]
    cr = args.positive_cr_list
    cr_a = args.anchor_cr_list
    mr = args.positive_mr_list
    mr_a = args.anchor_mr_list
    for corr_rate in cr:
        for anchor_corr_rate in cr_a:
            for mask_rate in mr:
                for anchor_mask_rate in mr_a:
                    args.corruption_rate = corr_rate
                    args.anchor_corruption_rate = anchor_corr_rate
                    args.mask_rate = mask_rate
                    args.anchor_mask_rate = anchor_mask_rate
                    train(args)
    