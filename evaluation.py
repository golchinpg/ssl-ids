import sys
import argparse

import torch
from torch import Tensor
from datasets import get_dataset, ExampleDataset
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from utils import *
from models import SCARF
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def OOD_classifier(train_features, test_features, k, T, cn, args):
    train_features = train_features.t()
    batch_size = 1000
    cos_sim_lst = []
    num_test_feat = test_features.size(0)
    for strat_idx in range(0, num_test_feat, batch_size):
        end_idx = min((strat_idx + batch_size), num_test_feat)
        curr_test_features = test_features[strat_idx:end_idx]
        curr_bs = curr_test_features.size(0)
        similarity = torch.mm(curr_test_features, train_features)
        if k != -1:
            similarity, indices = similarity.topk(k, largest=True, sorted=True)
        if T != -1:
            similarity = (similarity - 0.1).div_(T).exp_()
        cos_sim = similarity.mean(dim=1)
        cos_sim = cos_sim.view(curr_bs, cn).mean(dim=1)
        cos_sim_lst.append(cos_sim.cpu())
    cos_sim = torch.cat(cos_sim_lst, dim=0)
    return cos_sim


@torch.no_grad()
def knn_classifier(
    train_features: Tensor,
    train_labels: Tensor,
    test_features: Tensor,
    test_labels: Tensor,
    k: int,
    T: float,
    num_classes: int,
    device: torch.device,
):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[idx : min((idx + imgs_per_chunk), num_test_images), :]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1).long(), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        print(type(correct))
        print(correct.size())
        top5 = (
            top5 + correct.narrow(1, 0, min(5, k)).sum().item()
        )  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation with weighted k-NN on ImageNet")
    parser.add_argument(
        "--knn_temperature",
        default=0.04,
        type=float,
        help="Temperature used in the voting coefficient",
    )
    parser.add_argument(
        "--batch_size", default=512, type=int, help="Per-GPU batch-size"
    )
    parser.add_argument(
        "--run_knn", default=False, type=bool, help="Whether to run kNN"
    )
    parser.add_argument("--model_chkpt_path", default="/Users/pegah/Desktop/KOM/GitHub/ssl-ids/checkpoints/scarf1_embedding_dim=45_corruption_rate=0.6_lr=0.001_batch_size=128_epochs=40.pth", type=str)
    parser.add_argument("--main_dir", default="/Users/pegah/Desktop/KOM/GitHub/ssl-ids/")

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
    model.eval()
    train_normal_df = load_pandas_df(
        args.main_dir+"train_normal.csv",
    )
    train_normal_x, train_normal_y = (
        train_normal_df.iloc[:, :-1],
        train_normal_df["Label"],
    )
    train_ds = ExampleDataset(  # onlin normal flows
        train_normal_x.to_numpy(),
        train_normal_y.to_numpy(),
        columns=train_normal_x.columns,
    )
    unknown_df = load_pandas_df(args.main_dir+"1.csv")
    normal_unknown_df = unknown_df[unknown_df['Label']==0]
    attack_unknown_df = unknown_df[unknown_df['Label']==1]

    normal_unknown_x, normal_unknown_y = normal_unknown_df.iloc[:, :-1], normal_unknown_df["Label"]
    attack_unknown_x, attack_unknown_y = attack_unknown_df.iloc[:, :-1], attack_unknown_df["Label"]
    normal_unknown_ds = ExampleDataset(
        normal_unknown_x.to_numpy(),
        normal_unknown_y.to_numpy(),
        columns=normal_unknown_x.columns,
    )
    attack_unknown_ds = ExampleDataset(
        attack_unknown_x.to_numpy(),
        attack_unknown_y.to_numpy(),
        columns=attack_unknown_x.columns,
    )

    test_normal_df = load_pandas_df(args.main_dir+"test_normal.csv")
    test_normal_x, test_normal_y = test_normal_df.iloc[:, :-1], test_normal_df["Label"]
    test_normal_ds = ExampleDataset(
        test_normal_x.to_numpy(),
        test_normal_y.to_numpy(),
        columns=test_normal_x.columns,
    )

    attack_df = load_pandas_df(args.main_dir+"attack.csv")
    attack_x, attack_y = attack_df.iloc[:, :-1], attack_df["Label"]

    test_attack_ds = ExampleDataset(
        attack_x.to_numpy(), attack_y.to_numpy(), columns=attack_x.columns
    )
    # model = torch.load('/Users/pegah/Desktop/KOM/Codes/models/scarf_withoutnormalize_separate.pth')
    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    train_normal_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    
    test_normal_loader = DataLoader(
        test_normal_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    test_attack_loader = DataLoader(
        test_attack_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    
    unknown_attack_loader = DataLoader(
        attack_unknown_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    unknown_normal_loader = DataLoader(
        normal_unknown_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    train_embeddings, train_labels = get_embeddings_labels(
        model, train_normal_loader, device, to_numpy=False, normalize=True
    )
    
    test_normal_embeddings, test_normal_labels = get_embeddings_labels(
        model, test_normal_loader, device, to_numpy=False, normalize=True
    )
    test_attack_embeddings, test_attack_labels = get_embeddings_labels(
        model, test_attack_loader, device, to_numpy=False, normalize=True
    )
    
    attack_unknown_embeddings, attack_unknown_labels = get_embeddings_labels(
        model, unknown_attack_loader, device, to_numpy=False, normalize=True
    )
    normal_unknown_embeddings, normal_unknown_labels = get_embeddings_labels(
        model, unknown_normal_loader, device, to_numpy=False, normalize=True
    )

    print("Train Embedding Dim", torch.tensor(train_embeddings).size() if type(train_embeddings) is not torch.Tensor else train_embeddings.size())
    print("Test Normal Embedding Dim", torch.tensor(test_normal_embeddings).size() if type(test_normal_embeddings) is not torch.Tensor else test_normal_embeddings.size())
    print("Test Attack Embedding Dim", torch.tensor(test_attack_embeddings).size() if type(test_attack_embeddings) is not torch.Tensor else test_attack_embeddings.size())
    print("Attack Unknown Embedding Dim", torch.tensor(attack_unknown_embeddings).size() if type(attack_unknown_embeddings) is not torch.Tensor else attack_unknown_embeddings.size())
    print("Normal Unknown Embedding Dim", torch.tensor(normal_unknown_embeddings).size() if type(normal_unknown_embeddings) is not torch.Tensor else normal_unknown_embeddings.size())

    # should be modified
    
    def auroc_calculations(test_normal_embeddings, test_normal_labels, test_attack_embeddings ):
        if args.run_knn:
            print("###########   KNN on Embedings. Make sure of data correct data labels #############")
            top1, top5 = knn_classifier(
                train_embeddings,
                train_labels,
                test_normal_embeddings,
                test_normal_labels,
                10,
                0.04,
                num_classes=5,
                device=device,
            )
            print(f"10NN_Top1: {top1}")
            print(f"10NN_Top5: {top5}")

        scores_in = OOD_classifier(
            train_embeddings, test_normal_embeddings, -1, 0.04, 1, args
        )
        scores_out = OOD_classifier(
            train_embeddings, test_attack_embeddings, -1, 0.04, 1, args
        )

        labels = torch.cat((torch.ones(scores_in.size(0)), torch.zeros(scores_out.size(0))))
        scores = torch.cat((scores_in, scores_out))
        auroc = roc_auc_score(labels.numpy(), scores.cpu().numpy())
        print(f"AUROC: {auroc}")
    
    auroc_calculations(normal_unknown_embeddings, normal_unknown_labels, attack_unknown_embeddings)
    # # get embeddings for training and test set
    # train_data, test_data, train_target, test_target = get_dataset(args.path, separate_norm_attack=False, test_size=0.2)
    #
    # combi_train_ds = ExampleDataset(  # attack and normal flows
    #     train_data.to_numpy(), train_target.to_numpy(), columns=train_data.columns
    # )
    # combi_test_ds = ExampleDataset(
    #     test_data.to_numpy(), test_target.to_numpy(), columns=test_data.columns
    # )

    # train_embeddings = dataset_embeddings(model, train_combi_loader)  # , device)
    # test_normal_embeddings = dataset_embeddings(model, test_normal_loader)  # , device)
    # test_attack_embeddings = dataset_embeddings(model, test_attack_loader)
    #
    # print(train_embeddings.shape)
    # print(test_normal_embeddings.shape)
    # print(test_attack_embeddings.shape)
    #
    # print("###########   LR on the raw data #############")
    # clf = LogisticRegression()
    # # vanilla dataset: train the classifier on the original data
    # clf.fit(train_data, train_target)
    # vanilla_predictions_normal = clf.predict(X_test_normal)
    # vanilla_predictions_attack = clf.predict(x_attack)
    #
    # print(
    #     "classification report on normal set:",
    #     classification_report(y_test_normal, vanilla_predictions_normal),
    # )
    # print(
    #     "classification report on attack set:",
    #     classification_report(y_attack, vanilla_predictions_attack),
    # )
    #
    # print("############ LR on Embedding data ###############")
    # # embeddings dataset: train the classifier on the embeddings
    # clf.fit(train_embeddings, train_target)
    # vanilla_predictions_normal = clf.predict(test_normal_embeddings)
    # vanilla_predictions_attack = clf.predict(test_attack_embeddings)
    # print(
    #     "classification report on normal set:",
    #     classification_report(y_test_normal, vanilla_predictions_normal),
    # )
    # print(
    #     "classification report on attack set:",
    #     classification_report(y_attack, vanilla_predictions_attack),
    # )
