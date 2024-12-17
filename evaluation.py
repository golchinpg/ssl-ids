import os
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
from models import SCARF, MLP
from sklearn.metrics import roc_auc_score

# Set CUDA environment variables
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
@torch.no_grad()
def OOD_classifier(train_features, test_features, k, T, cn, args):  # features must be l2 normalized
    train_features = train_features.t()
    batch_size = 500
    cos_sim_lst = []
    cos_sim_max_lst = []
    num_test_feat = test_features.size(0)
    
    for strat_idx in range(0, num_test_feat, batch_size):
        end_idx = min((strat_idx + batch_size), num_test_feat)
        curr_test_features = test_features[strat_idx:end_idx]
        curr_bs = curr_test_features.size(0)
        similarity = torch.mm(curr_test_features, train_features)
        cos_sim_max, _ = similarity.max(dim=1)

        if k != -1:
            similarity, indices = similarity.topk(k, largest=True, sorted=True)
        if T != -1:
            similarity = (similarity - 0.1).div_(T).exp_()
        cos_sim = similarity.mean(dim=1)
        
        cos_sim_lst.append(cos_sim.cpu())
        cos_sim_max_lst.append(cos_sim_max)

    cos_sim = torch.cat(cos_sim_lst, dim=0)
    cos_sim_max = torch.cat(cos_sim_max_lst, dim=0)
    return cos_sim, cos_sim_max


def auroc_calculations(train_embeddings, test_normal_embeddings, test_attack_embeddings, args):
    print('t == 0.04')
    scores_in, scores_in_max = OOD_classifier(
        train_embeddings, test_normal_embeddings, -1, 0.04, 1, args
    )
    scores_out, scores_out_max = OOD_classifier(
        train_embeddings, test_attack_embeddings, -1, 0.04, 1, args
    )

    labels = torch.cat((torch.ones(scores_in.size(0)), torch.zeros(scores_out.size(0))))
    scores = torch.cat((scores_in, scores_out))
    auroc = roc_auc_score(labels.numpy(), scores.cpu().numpy())
    print(f"AUROC: {auroc}")

    labels_max_cos = torch.cat((torch.ones(scores_in_max.size(0)), torch.zeros(scores_out_max.size(0))))
    scores_max_cos = torch.cat((scores_in_max, scores_out_max))
    auroc_max_cos = roc_auc_score(labels_max_cos.numpy(), scores_max_cos.cpu().numpy())
    print(f"AUROC for max cosine similarity: {auroc_max_cos}")


if __name__ == "__main__":
    main_dir = os.path.dirname(os.path.abspath(__file__))
    test_dataset_dir = os.path.join(main_dir, 'Dataset/')
    chk_dir = os.path.join(main_dir, "checkpoints/")

    parser = argparse.ArgumentParser("Evaluation of SSCL-IDS")
    parser.add_argument(
        "--batch_size", default=256, type=int, help="Per-GPU batch-size"
    )
    parser.add_argument(
        "--model_chkpt", 
        default="scarf1_embdd_dim=45_lr=0.001_bs=2046_epochs=100_tempr=0.5_V=onlyctu13_cr_rt=0.4_ach_cr_rt0.2_msk_rt0_ach_msk_rt0.pth", 
        type=str, help="Checkpoint file name"
    )
    parser.add_argument(
        "--test_dataset_name", type=str, required=True, help="Test dataset filename (CSV)"
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(os.path.join(chk_dir, args.model_chkpt))
    train_args = ckpt["args"]
    
    model = SCARF(
        input_dim=train_args.input_dim,
        emb_dim=train_args.embedding_dim,
        corruption_rate=train_args.corruption_rate,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load training data
    train_normal_df = load_pandas_df(os.path.join(main_dir, 'training_cicbotnet.csv'))
    train_label_df = pd.DataFrame({'Label': [0] * len(train_normal_df)})
    train_normal_x, train_normal_y = train_normal_df, train_label_df
    train_ds = ExampleDataset(train_normal_x.to_numpy(), train_normal_y.to_numpy(), columns=train_normal_x.columns)
    train_normal_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    train_embeddings, train_labels = get_embeddings_labels(
        model, train_normal_loader, device, to_numpy=False, normalize=True
    )

    # Process test dataset
    test_dataset_name = args.test_dataset_name
    list_of_datasets = [os.path.join(test_dataset_dir, test_dataset_name)]
    for dataset_unknown_name in list_of_datasets:
        print(f'Test dataset: {dataset_unknown_name}')
        unknown_df = load_pandas_df(dataset_unknown_name)
        normal_unknown_df = unknown_df[unknown_df['Label'] == 0]
        attack_unknown_df = unknown_df[unknown_df['Label'] == 1]

        normal_unknown_x, normal_unknown_y = normal_unknown_df.iloc[:, :-1], normal_unknown_df["Label"]
        attack_unknown_x, attack_unknown_y = attack_unknown_df.iloc[:, :-1], attack_unknown_df["Label"]
        
        if normal_unknown_df.shape[0] > 0:
            normal_unknown_ds = ExampleDataset(
                normal_unknown_x.to_numpy(),
                normal_unknown_y.to_numpy(),
                columns=normal_unknown_x.columns,
            )
        else:
            print(f"There is no normal flow in this dataset!")

        attack_unknown_ds = ExampleDataset(
            attack_unknown_x.to_numpy(),
            attack_unknown_y.to_numpy(),
            columns=attack_unknown_x.columns,
        )
        
        unknown_attack_loader = DataLoader(attack_unknown_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
        if normal_unknown_df.shape[0] > 0:
            unknown_normal_loader = DataLoader(normal_unknown_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

        attack_unknown_embeddings, attack_unknown_labels = get_embeddings_labels(
            model, unknown_attack_loader, device, to_numpy=False, normalize=True
        )
        if normal_unknown_df.shape[0] > 0:
            normal_unknown_embeddings, normal_unknown_labels = get_embeddings_labels(
                model, unknown_normal_loader, device, to_numpy=False, normalize=True
            )
        if normal_unknown_df.shape[0] > 0:
            auroc_calculations(train_embeddings, normal_unknown_embeddings, attack_unknown_embeddings, args)
        else:
            auroc_calculations(train_embeddings, train_embeddings, attack_unknown_embeddings, args)

        print("Train Embedding Dim", torch.tensor(train_embeddings).size() if type(train_embeddings) is not torch.Tensor else train_embeddings.size())
        print("Attack "" Embedding Dim", torch.tensor(attack_unknown_embeddings).size() if type(attack_unknown_embeddings) is not torch.Tensor else attack_unknown_embeddings.size())
        
    