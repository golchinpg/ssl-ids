import sys
import argparse

import torch
from torch import Tensor
from datasets import get_dataset, ExampleDataset
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from utils import *
from models import SCARF
from sklearn.metrics import roc_auc_score
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation with weighted k-NN on ImageNet")
    parser.add_argument("--supervised_model", default='RF', type= str)
    parser.add_argument("--supervised_training_dataset", default="/Users/pegah/Desktop/KOM/Datasets/preprocessed_csv/allattack_mondaybenign.csv", type=str)
    parser.add_argument("--model_chkpt_path", default="/Users/pegah/Desktop/KOM/GitHub/ssl-ids/checkpoints/scarf1_embedding_dim=45_corruption_rate=0.6_lr=0.001_batch_size=128_epochs=40.pth", type=str)
    parser.add_argument(
        "--batch_size", default=512, type=int, help="Per-GPU batch-size"
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.model_chkpt_path)
    train_args = ckpt["args"]
    model = SCARF(
        input_dim=train_args.input_dim,
        emb_dim=train_args.embedding_dim,
        corruption_rate=train_args.corruption_rate,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    train_df = load_pandas_df(
        args.supervised_training_dataset
    )
    x_train, y_train = train_df.iloc[:, :-1], train_df["Label"]
    train_ds = ExampleDataset(  # onlin normal flows
        x_train.to_numpy(),
        y_train.to_numpy(),
        columns=x_train.columns,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    train_embeddings, train_labels = get_embeddings_labels(
        model, train_loader, device, to_numpy=False, normalize=True
    )
    unknown_df = load_pandas_df("/Users/pegah/Desktop/KOM/GitHub/ssl-ids/1.csv")
    x_unknown, y_unknown = unknown_df.iloc[:, :-1], unknown_df["Label"]
    unknown_ds = ExampleDataset(  # onlin normal flows
        x_unknown.to_numpy(),
        y_unknown.to_numpy(),
        columns=x_unknown.columns,
    )
    unknown_loader = DataLoader(
        unknown_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    unknown_embeddings, unknown_labels = get_embeddings_labels(
        model, unknown_loader, device, to_numpy=False, normalize=True
    )
    if args.supervised_model =='LR':
        print("###########   LR on the raw data #############")
        clf = LogisticRegression()
        clf.fit(x_train, y_train)
        prediction = clf.predict(x_unknown)
        print('classification report on raw set:',classification_report(y_unknown, prediction))

        print('############ LR on Embedding data ###############')
        clf.fit(train_embeddings, train_labels)
        prediction_emb = clf.predict(unknown_embeddings)
        print('classification report on embedded set:',classification_report(unknown_labels, prediction_emb))

    if args.supervised_model =='RF':
        print("###########   RF on the raw data #############")
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(x_train, y_train)
        prediction = clf.predict(x_unknown)
        print('classification report on raw set:',classification_report(y_unknown, prediction))

        print('############ RF on Embedding data ###############')
        clf.fit(train_embeddings, train_labels)
        prediction_emb = clf.predict(unknown_embeddings)
        print('classification report on embedded set:',classification_report(unknown_labels, prediction_emb))
