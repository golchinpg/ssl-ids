import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from utils import *
from models import SCARF, MLP
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from datasets import get_dataset, ExampleDataset
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Define the fine-tune MLP model
class MLP(nn.Sequential):
    """Simple multi-layer perceptron with ReLu activation and optional dropout layer"""

    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.0):
        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, hidden_dim))

        super().__init__(*layers)


class FineTuneModel(nn.Module):
    def __init__(self, 
                 input_dim=45,
                 emb_dim=45,
                 encoder_depth=5,
                 num_classes=2):
        super(FineTuneModel, self).__init__()
        # Pretrained SCARF model
        self.encoder = MLP(input_dim, emb_dim, encoder_depth)
        # MLP layer
        #self.mlp_layer = nn.Linear(input_dim, num_classes)
        self.mlp_layer= MLP(emb_dim, 2, 1, 0)

    def forward(self, x):
        # Forward pass through the SCARF model
        x = self.encoder(x)
        # Forward pass through the MLP layer
        x = self.mlp_layer(x)
        return x

if __name__ == "__main__":
    main_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser("Evaluation with weighted k-NN on ImageNet")
    parser.add_argument(
        "--batch_size", default=1024, type=int, help="Per-GPU batch-size"
    )
    parser.add_argument("--model_chkpt_path", default="/home/pegah/Codes/ssl-ids/new_checkpoints/scarf1_embdd_dim=45_lr=0.001_bs=2046_epochs=100_tempr=0.5_V=900_cr_rt=0.4_ach_cr_rt0.2_msk_rt0_ach_msk_rt0.pth" , type=str)
                        #scarf1_embedding_dim=45_corruption_rate=0.3_lr=0.001_batch_size=2046_epochs=200_temprature=0.5_version=101.pth", type=str)
    parser.add_argument("--dataset_dir", default="/home/pegah/Codes/ssl-ids/Dataset/ISCX-SlowDoS_1.csv")
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("learning_rate", default=0.001)
    parser.add_argument("--num_epochs", default=30 , type=int)
    parser.add_argument("--apply_smote", default = True)



    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model_chkpt_path)
    train_args = ckpt["args"]
    pretrained_model = SCARF(
        input_dim=train_args.input_dim,
        emb_dim=train_args.embedding_dim,
        corruption_rate=0 #train_args.corruption_rate,
    )
    #check the availability of the paths
    if not os.path.isfile(args.model_chkpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at {args.model_chkpt_path}")
    if not os.path.isfile(args.dataset_dir):
        raise FileNotFoundError(f"Dataset file not found at {args.dataset_dir}")
    
    pretrained_model.load_state_dict(ckpt["model_state_dict"])
    model = FineTuneModel()
    model.encoder = pretrained_model.encoder
    model = model.to(device)
    model_raw = MLP(input_dim=45, hidden_dim=2, n_layers=1, dropout=0.3)
    model_raw = model_raw.to(device)

    #print(">>>>>>>>>>>>>>>>>>>>>>>>>>")
    #for par1, par2 in zip(pretrained_model.encoder.parameters(), model.encoder.parameters()):
    #    print(par1 == par2)

    
    learning_rate = args.learning_rate
    # Create the fine-tune model
    #fine_tune_model = FineTuneModel(model, input_dim, num_classes)
    #fine_tune_model = fine_tune_model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer_raw = optim.Adam(model_raw.parameters(), lr=learning_rate)

    def Split_dataset( X, y, test_size):
        #X = df.iloc[:, :-1]
        #y = df["Label"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print("shape of X_train and X_test:", X_train.shape, X_test.shape)
        return (X_train, y_train, X_test, y_test)
    #['allattack_mondaybenign.csv', 'botnet43_truncated.csv','ISCX-SlowDoS_1.csv',
                        #'merged_1-6.csv', 'botnet_iscx_training.csv']
    df = load_pandas_df(args.dataset_dir) #Add your data points to fine tune the model
    print(f"Dataset {str(args.dataset_dir).split('/')[-1]} is added to fine-tune!")
    X, y = (
            df.iloc[:, :-1],
            df["Label"],
        )
    X_use, y_use, X_test, y_test = Split_dataset(
                    X, y, 0.3
                )
    test_ds = ExampleDataset(  # onlin normal flows
        X_test.to_numpy(),
        y_test.to_numpy(),
        columns=X.columns,
    )
    print('test ds is ready!')
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    print('test loader is ready!')
    #test_embeddings, test_labels = get_embeddings_labels(
    #    model, test_loader, device, to_numpy=True, normalize=True
    #)
    def create_train_loader(X_train, y_train, bs):
        train_ds = ExampleDataset(  # onlin normal flows
                X_train.to_numpy(),
                y_train.to_numpy(),
                columns=X.columns,
            )
        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True, drop_last=False
            )
        return train_loader

    #training_portion = [ 0.00005, 0.0005, 0.001, 0.01,0.5,0.9]# 0.000005,0.00005, 0.00005, 0.0001, 0.0003,
    training_portion = [1]
    for portion in training_portion:
        print('############### train with '+str(portion))
        X_train, y_train = X_use, y_use
        print('X_train:', X_train.shape)
        #print('X_test:',X_test.shape)
        bs = min(X_train.shape[0], args.batch_size)
        if args.apply_smote == True:
            # Apply SMOTE to the training data
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            train_loader = create_train_loader(X_train_resampled, y_train_resampled, bs)
            """
            _, _, X_train, y_train = Split_dataset(
                        X_use, y_use , portion
                    )
            print('portion is ', float(len(X_use))*portion)
            """
            
            
        else:
            train_loader = create_train_loader(X_train, y_train, bs)

        
        #print('train loader is ready!')
        print('Start training ....')
        num_epochs = args.num_epochs
        for epoch in range(num_epochs):
            model.train()
            for X_train, _, _, y_train in train_loader:
                X_train = X_train.to(device)
                y_train = y_train.to(device).long()
                optimizer.zero_grad()
                # Forward pass
                outputs = model(X_train)
                # Compute loss
                loss = criterion(outputs, y_train)
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        """
        print("Start training Raw model ...")
        for epoch in range(num_epochs):
            model_raw.train()
            for X_train, _, _, y_train in train_loader:
                X_train = X_train.to(device)
                y_train = y_train.to(device).long()
                optimizer_raw.zero_grad()
                # Forward pass
                outputs_raw = model_raw(X_train)
                # Compute loss
                loss_raw = criterion(outputs_raw, y_train)
                # Backward pass and optimization
                loss_raw.backward()
                optimizer_raw.step()
            #print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_raw.item():.4f}')
        """
        print('Evaluation starts ...')
        model.eval()
        #model_raw.eval()
        # Forward pass to get predicted probabilities
        with torch.no_grad():
            probs = []
            labels = []
            preds = []
            for X_test, _,_, y_test in test_loader:
                X_test = X_test.to(device)
                y_test = y_test.to(device)
                logits = model(X_test)
                pred = logits.argmax(dim = 1).cpu().numpy()
                preds.extend(pred)
                """
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                predicted_probabilities = probabilities[:, 1].cpu().numpy()
                probs.extend(predicted_probabilities)
                """
                labels.extend(y_test.cpu().numpy())

        #print('labels shape:', np.array(labels).shape)
        #print('probs shape:', np.array(probs).shape)
        # Calculate AUROC and F1-score
        #auroc = roc_auc_score(np.array(labels), np.array(probs))
        print('classification report of ED:', classification_report(np.array(labels), np.array(preds)))
        #print(f'AUROC of ED: {auroc:.4f}')
        """
        #raw data:
        with torch.no_grad():
            probs_raw = []
            labels_raw = []
            for X_test, _,_, y_test in test_loader:
                X_test = X_test.to(device)
                y_test = y_test.to(device)
                logits_raw = model_raw(X_test)
                probabilities_raw = torch.nn.functional.softmax(logits_raw, dim=1)
                predicted_probabilities_raw = probabilities_raw[:, 1].cpu().numpy()
                probs_raw.extend(predicted_probabilities_raw)
                labels_raw.extend(y_test.cpu().numpy())
        auroc_rd = roc_auc_score(np.array(labels_raw), np.array(probs_raw))
        print(f'AUROC of RD: {auroc_rd:.4f}')
        """

