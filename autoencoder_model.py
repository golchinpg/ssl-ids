import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils import load_pandas_df
from sklearn.preprocessing import MinMaxScaler
import argparse
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import numpy as np

# Get the directory of the current script (train.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Define the path to the new_checkpoint folder
AE_checkpoint_dir = os.path.join(current_dir, 'AE_checkpoints')
# Create the new_checkpoints folder if it doesn't exist
os.makedirs(AE_checkpoint_dir, exist_ok=True)

class AE(nn.Module):
    """Simple multi-layer perceptron with ReLu activation and optional dropout layer"""
    def __init__(self, input_size, encoding_dim):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features= input_size, out_features= 30),
            nn.ReLU(),
            nn.Linear(30, encoding_dim),
            #nn.ReLU(),
            #nn.Linear(20, 15)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 35),
            nn.ReLU(),
            nn.Linear(35, input_size), 
            nn.Sigmoid()
        )
    def forward(self, x):
        x= self.encoder(x)
        x= self.decoder(x)
        return(x)
    def get_embeddings(self, input):
        return self.encoder(input)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train AE")
    parser.add_argument("--training_ds",  default="/home/pegah/Codes/ssl-ids/Dataset/concatenated_df_normal.csv", type=str,)
    parser.add_argument("--batch_size", default=512, type=int,)
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--encoding_dim", default=20, type=float)
    parser.add_argument("--input_size", default=45, type=float)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    #create an AE model
    model = AE(args.input_size, args.encoding_dim).to(device)

    #loss function and optimizer
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(),lr = args.lr)

    #Dataset
    dataset_path = "/home/pegah/Codes/ssl-ids/Dataset/concatenated_df_normal.csv"
    train_df = load_pandas_df(dataset_path)
    print("shape of dataset:", train_df.shape)
    X, _, y, _ = train_test_split(train_df.iloc[:, :-1], train_df["Label"], test_size=0.5, random_state=42)
    X_train, y_train = X, y
    for col in X_train.columns:
        if 'Unnamed' in col:
            X_train = X_train.drop(col, axis = 1)
    print("shape of dataset:", X_train.shape)
    #minmax normalization
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    print(X_train_normalized.shape)
    #convert dataframe to pytorch tensors
    X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    print("tensor shape:", X_train_tensor.shape)
    #dataset = TensorDataset(X_train_tensor)

    #Dataloader
    dataloader = DataLoader(X_train_tensor, batch_size=args.batch_size, shuffle=True)
    print(dataloader)

    log_name = f"AE_embdd_dim={args.encoding_dim}_lr={args.lr}_bs={args.batch_size}_epochs={args.epochs}"


    #Start training
    loss_history = []
    for epoch in range(args.epochs):
        for batch_data in dataloader:
            #print(batch_data.size())
            batch_data = batch_data.to(device)
            #forward pass
            outputs = model(batch_data)
            #compute the loss
            loss = criterion(outputs, batch_data)
            loss_history.append(loss)
            #backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{args.epochs}], loss: {loss.item():.9f}')
        torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_history": loss_history,
                    'args': args,
                    'train_data_columns': X_train.columns
                },
                os.path.join(AE_checkpoint_dir, f'{log_name}.pth'),
            )


