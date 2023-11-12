import torch
import numpy as np
import random
from typing import List
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd


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
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # size is 2*bs

        mask = (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)
        ).float()  # .to(device)
        numerator = torch.exp(positives / self.temperature)
        denominator = mask * torch.exp(similarity / self.temperature)

        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)

        return loss


@torch.no_grad()
def get_embeddings_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    to_numpy: bool = True,
    normalize: bool = False,
):
    model.eval()
    embeddings = []
    labels = []
    for anchor, _, label in loader:
        anchor = anchor.to(device)
        embeddings.append(model.get_embeddings(anchor))
        labels.append(label)
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    if to_numpy:
        embeddings = embeddings.cpu().numpy()
        labels = labels.cpu().numpy()
    return embeddings, labels


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def store_pandas_df(pandas_df: pd.DataFrame, path):
    pandas_df.to_csv(path)


def load_pandas_df(path: str, columns: List =None):
    if columns is None:
        df = pd.read_csv(path, header=0, sep=",")
        for col in df.columns:
            if "Unnamed" in col:
                df = df.drop(col, axis=1)
            if "ms" in col and not "duration" in col:
                df = df.drop(col, axis=1)
        return df
    else:
        return pd.read_csv(path, header=0, sep=",", usecols=columns)
