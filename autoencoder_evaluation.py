import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd

# Set seed for reproducibility
torch.manual_seed(42)

# Load your dataset using pandas
train_dataset_name = "allattack_mondaybenign.csv"  # Replace with the actual file name
df_train = pd.read_csv("/home/pegah/Codes/ssl-ids/Dataset/" + train_dataset_name, header=0, sep=',')
for col in df_train.columns:
    if 'Unnamed' in col:
        df_train = df_train.drop(col, axis=1)

# Drop columns with "Unnamed" in the name
print(df_train.shape)

# Split the data into features (X) and labels (y)
X = df_train.iloc[:, :-1]
y = df_train["Label"]
print(y.unique())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(X_train.shape)
print(y_train.shape)
# Convert data to PyTorch tensors and move to GPU
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
X_train_tensor = torch.FloatTensor(X_train.values).to(device)
y_train_tensor = torch.FloatTensor(y_train.values).to(device)
X_test_tensor = torch.FloatTensor(X_test.values).to(device)
y_test_tensor = torch.FloatTensor(y_test.values).to(device)

# Define the autoencoder model and move to GPU
class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(64, encoding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_size),
            nn.Sigmoid(),  # Sigmoid activation for binary classification
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the model and move to GPU
encoding_dim = 10
autoencoder = Autoencoder(X_train.shape[1], encoding_dim).to(device)

# Set up loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Create DataLoader for training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training the autoencoder
num_epochs = 10

for epoch in range(num_epochs):
    for data in train_loader:
        inputs, truelabel = data
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = autoencoder(inputs)

        loss = criterion(outputs, truelabel)
        loss.backward()
        optimizer.step()

# Evaluate the model and calculate AUROC
autoencoder.eval()

with torch.no_grad():
    y_pred = autoencoder(X_test_tensor)
    auroc = roc_auc_score(y_test.cpu().numpy(), y_pred.cpu().numpy())
    print(f'AUROC: {auroc}')