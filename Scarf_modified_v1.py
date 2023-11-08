import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import random
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter 

def get_separate_dataset(dataset_path:str ):
    def extract_train_test(df):
        data, target = df.iloc[:, :-1], df['Label']
        train_data, test_data, train_target, test_target = train_test_split(
        data, 
        target, 
        test_size=0.3, 
        stratify=target
        )
        return train_data, test_data, train_target, test_target
    
    merged_file = dataset_path+'allattack_mondaybenign.csv'

    df = pd.read_csv(merged_file, header=0, sep=",")
    for col in df.columns:
                if "Unnamed" in col:
                    df = df.drop(col, axis=1)
                if "ms" in col and not "duration" in col:
                    df = df.drop(col, axis=1)
    df_normal = df[df['Label']==0]
    df_attack = df[df['Label']==1]
    X_attack, y_attack  = df_attack.iloc[:, :-1], df_attack['Label']
    #scaler = MinMaxScaler()
    #preprocessed_dataset_scaled = scaler.fit_transform(df)
    X_train_normal, X_test_normal, y_train_normal, y_test_normal = extract_train_test(df_normal)
    #df_normalized = pd.DataFrame(preprocessed_dataset_scaled, columns=df.columns, index=df.index)
    return X_train_normal, X_test_normal, X_attack, y_train_normal, y_test_normal, y_attack
    
    


def get_dataset(dataset_path:str ):
    
    merged_file = dataset_path+'allattack_mondaybenign.csv'

    df = pd.read_csv(merged_file, header=0, sep=",")
    for col in df.columns:
                if "Unnamed" in col:
                    df = df.drop(col, axis=1)
                if "ms" in col and not "duration" in col:
                    df = df.drop(col, axis=1)

    #scaler = MinMaxScaler()
    #preprocessed_dataset_scaled = scaler.fit_transform(df)

    #df_normalized = pd.DataFrame(preprocessed_dataset_scaled, columns=df.columns, index=df.index)
    #data, target = df_normalized.iloc[:, :-1], df_normalized['Label']
    data, target = df.iloc[:, :-1], df['Label']
    train_data, test_data, train_target, test_target = train_test_split(
    data, 
    target, 
    test_size=0.2, 
    stratify=target
    )
    return train_data, test_data, train_target, test_target



    """
    df_positive = df[df['Label']==0]
    df_attack = df[df['Label']==1]
    #print(df_normalized.shape)
    #print(df_normalized.columns)
    X_positive = df_positive.iloc[:, :-1]
    y_positive = df_positive["Label"]
    X_train_positive, X_test_positive, y_train_positive, y_test_positive = train_test_split(
                X_positive, y_positive, test_size=0.3, random_state=42
            )
    X_attack = df_attack.iloc[:, :-1]
    y_attack = df_attack["Label"]
    X_train_attack, X_test_attack, y_train_attack, y_test_attacke = train_test_split(
                X_attack, y_attack, test_size=0.3, random_state=42
            )
    return X_train_positive, y_train_positive,  X_train_attack, X_test_positive, X_test_attack
    """

class ExampleDataset(Dataset):
    def __init__(self, data, target, columns=None):
        self.data = np.array(data)
        self.target = np.array(target)
        self.columns = columns

    def __getitem__(self, index):
        # the dataset must return a pair of samples: the anchor and a random one from the
        # dataset that will be used to corrupt the anchor
        random_idx = np.random.randint(0, len(self))
        random_sample = torch.tensor(self.data[random_idx], dtype=torch.float)
        sample = torch.tensor(self.data[index], dtype=torch.float)

        return sample, random_sample

    def __len__(self):
        return len(self.data)

    def to_dataframe(self):
        return pd.DataFrame(self.data, columns=self.columns)

    @property
    def shape(self):
        return self.data.shape



class NTXent(nn.Module):
    def __init__(self, temperature=1.0):
        """NT-Xent loss for contrastive learning using cosine distance as similarity metric as used in [SimCLR](https://arxiv.org/abs/2002.05709).
        Implementation adapted from https://theaisummer.com/simclr/#simclr-loss-implementation

        Args:
            temperature (float, optional): scaling factor of the similarity metric. Defaults to 1.0.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """Compute NT-Xent loss using only anchor and positive batches of samples. Negative samples are the 2*(N-1) samples in the batch

        Args:
            z_i (torch.tensor): anchor batch of samples
            z_j (torch.tensor): positive batch of samples

        Returns:
            float: loss
        """
        batch_size = z_i.size(0)

        # compute similarity between the sample's embedding and its corrupted view
        z = torch.cat([z_i, z_j], dim=0)
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0) # size is 2*bs

        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)).float()#.to(device)
        numerator = torch.exp(positives / self.temperature)
        denominator = mask * torch.exp(similarity / self.temperature)

        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)

        return loss

class MLP(torch.nn.Sequential):
    """Simple multi-layer perceptron with ReLu activation and optional dropout layer"""

    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.0):
        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_dim))

        super().__init__(*layers)


class SCARF(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        encoder_depth=3,
        head_depth=1,
        corruption_rate=0.6,
        encoder=None,
        pretraining_head=None,
    ):
        """Implementation of SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption.
        It consists in an encoder that learns the embeddings.
        It is done by minimizing the contrastive loss of a sample and a corrupted view of it.
        The corrupted view is built by remplacing a random set of features by another sample randomly drawn independently.

            Args:
                input_dim (int): size of the inputs
                emb_dim (int): dimension of the embedding space
                encoder_depth (int, optional): number of layers of the encoder MLP. Defaults to 4.
                head_depth (int, optional): number of layers of the pretraining head. Defaults to 2.
                corruption_rate (float, optional): fraction of features to corrupt. Defaults to 0.6.
                encoder (nn.Module, optional): encoder network to build the embeddings. Defaults to None.
                pretraining_head (nn.Module, optional): pretraining head for the training procedure. Defaults to None.
        """
        super().__init__()

        if encoder:
            self.encoder = encoder
        else:
            self.encoder = MLP(input_dim, emb_dim, encoder_depth)

        if pretraining_head:
            self.pretraining_head = pretraining_head
        else:
            self.pretraining_head = MLP(emb_dim, emb_dim, head_depth)

        # initialize weights
        self.encoder.apply(self._init_weights)
        self.pretraining_head.apply(self._init_weights)
        self.corruption_len = int(corruption_rate * input_dim)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, anchor, random_sample):
        batch_size, m = anchor.size()

        # 1: create a mask of size (batch size, m) where for each sample we set the
        # jth column to True at random, such that corruption_len / m = corruption_rate
        # 3: replace x_1_ij by x_2_ij where mask_ij is true to build x_corrupted

        corruption_mask = torch.zeros_like(anchor, dtype=torch.bool)
        for i in range(batch_size):
            corruption_idx = torch.randperm(m)[: self.corruption_len]
            corruption_mask[i, corruption_idx] = True

        positive = torch.where(corruption_mask, random_sample, anchor)

        # compute embeddings
        emb_anchor = self.encoder(anchor)
        emb_anchor = self.pretraining_head(emb_anchor)

        emb_positive = self.encoder(positive)
        emb_positive = self.pretraining_head(emb_positive)

        return emb_anchor, emb_positive

    def get_embeddings(self, input):
        return self.encoder(input)
    
def train_epoch(model, criterion, train_loader, optimizer,  epoch):#device,
    model.train()
    epoch_loss = 0.0
    batch = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    for anchor, positive in batch:
        anchor, positive = anchor, positive#.to(device)

        # reset gradients
        optimizer.zero_grad()

        # get embeddings
        emb_anchor, emb_positive = model(anchor, positive)

        # compute loss
        loss = criterion(emb_anchor, emb_positive)
        loss.backward()

        # update model weights
        optimizer.step()

        # log progress
        epoch_loss += anchor.size(0) * loss.item()
        batch.set_postfix({"loss": loss.item()})

    return epoch_loss / len(train_loader.dataset)


def dataset_embeddings(model, loader):#, device):
    model.eval()
    embeddings = []

    with torch.no_grad():
        for anchor, _ in tqdm(loader):
            anchor = anchor#.to(device)
            embeddings.append(model.get_embeddings(anchor))

    embeddings = torch.cat(embeddings).numpy()

    return embeddings


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


path = "/Users/pegah/Desktop/KOM/Datasets/preprocessed_csv/"
train_data, test_data, train_target, test_target = get_dataset(path)
X_train_normal, X_test_normal, X_attack, y_train_normal, y_test_normal, y_attack = get_separate_dataset(path)
# to torch dataset
combi_train_ds = ExampleDataset( #attack and normal flows
    train_data.to_numpy(), 
    train_target.to_numpy(), 
    columns=train_data.columns
)
combi_test_ds = ExampleDataset(
    test_data.to_numpy(), 
    test_target.to_numpy(), 
    columns=test_data.columns
)
train_ds = ExampleDataset( #onlin normal flows
    X_train_normal.to_numpy(), 
    y_train_normal.to_numpy(), 
    columns=X_train_normal.columns
)
test_normal_ds = ExampleDataset(
    X_test_normal.to_numpy(), 
    y_test_normal.to_numpy(), 
    columns=X_test_normal.columns
)
test_attack_ds = ExampleDataset(
    X_attack.to_numpy(), 
    y_attack.to_numpy(), 
    columns=X_attack.columns
)
print(f"Train set: {train_ds.shape}")
print(f"Test Normal set: {test_normal_ds.shape}")
print(f"Test Attack set: {test_attack_ds.shape}")
#train_ds.to_dataframe().head()

#training
print('start training ...')
batch_size = 128
epochs = 40
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('mps')
#print(device)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

writer = SummaryWriter(f"logs/scarf1")


model = SCARF(
    input_dim=train_ds.shape[1], 
    emb_dim=23,
    corruption_rate=0.6,
)#.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
ntxent_loss = NTXent()#.to(device)

loss_history = []

for epoch in range(1, epochs + 1):
    epoch_loss = train_epoch(model, ntxent_loss, train_loader, optimizer,  epoch)#device,
    loss_history.append(epoch_loss)
    writer.add_scalar("Training Loss modified scarf", epoch_loss, global_step=epoch, walltime=0.001)     
print('saving model') 
torch.save(model, '/Users/pegah/Desktop/KOM/Codes/models/scarf_withoutnormalize_separate_c6_ep40.pth' )

print('loading ....')
#model = torch.load('/Users/pegah/Desktop/KOM/Codes/models/scarf_withoutnormalize_separate.pth')
#train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
train_combi_loader = DataLoader(combi_train_ds, batch_size=batch_size, shuffle=False)
test_normal_loader = DataLoader(test_normal_ds, batch_size=batch_size, shuffle=False)
test_attack_loader = DataLoader(test_attack_ds, batch_size=batch_size, shuffle=False)


# get embeddings for training and test set
train_embeddings = dataset_embeddings(model, train_combi_loader)#, device)
test_normal_embeddings = dataset_embeddings(model, test_normal_loader)#, device)
test_attack_embeddings = dataset_embeddings(model, test_attack_loader)

print(train_embeddings.shape)
print(test_normal_embeddings.shape)
print(test_attack_embeddings.shape)

print("###########   LR on the raw data #############")
clf = LogisticRegression()

# vanilla dataset: train the classifier on the original data
clf.fit(train_data, train_target)
vanilla_predictions_normal = clf.predict(X_test_normal)
vanilla_predictions_attack = clf.predict(X_attack)


print('classification report on normal set:',classification_report(y_test_normal, vanilla_predictions_normal))
print('classification report on attack set:',classification_report(y_attack, vanilla_predictions_attack))


print('############ LR on Embedding data ###############')
# embeddings dataset: train the classifier on the embeddings
clf.fit(train_embeddings, train_target)
vanilla_predictions_normal = clf.predict(test_normal_embeddings)
vanilla_predictions_attack = clf.predict(test_attack_embeddings)


print('classification report on normal set:',classification_report(y_test_normal, vanilla_predictions_normal))
print('classification report on attack set:',classification_report(y_attack, vanilla_predictions_attack))

