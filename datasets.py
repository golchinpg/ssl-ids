import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


class ExampleDataset(Dataset):
    def __init__(self, data, target, columns=None):
        self.data = np.array(data)
        self.target = np.array(target)
        self.columns = columns
        print(np.unique(self.target))
        if len(np.unique(self.target))>1:
            self.normal_data = self.data[self.target==0]
            self.attack_data = self.data[self.target==1]
        else:
            if np.unique(self.target)[0] == 0:
                self.normal_data = self.data
            else:
                self.attack_data = self.data

    def __getitem__(self, index):
        # the dataset must return a pair of samples: the anchor and a random one from the
        # dataset that will be used to corrupt the anchor
        if self.target[index] == 0:
            random_idx = np.random.randint(0, len(self.normal_data))
            random_sample_1 = torch.tensor(self.normal_data[random_idx], dtype=torch.float)
            random_idx = np.random.randint(0, len(self.normal_data))
            random_sample_2 = torch.tensor(self.normal_data[random_idx], dtype=torch.float)            
            
        else:
            random_idx = np.random.randint(0, len(self.attack_data))
            random_sample_1 = torch.tensor(self.attack_data[random_idx], dtype=torch.float)
            random_idx = np.random.randint(0, len(self.attack_data))
            random_sample_2 = torch.tensor(self.attack_data[random_idx], dtype=torch.float) 

        sample = torch.tensor(self.data[index], dtype=torch.float)
        target = torch.tensor(self.target[index], dtype=torch.float)
        return sample, random_sample_1, random_sample_2, target

    def __len__(self):
        return len(self.data)

    def to_dataframe(self):
        return pd.DataFrame(self.data, columns=self.columns)

    @property
    def shape(self):
        return self.data.shape


def get_dataset(
    dataset_path: str, training_with_attacks: bool = False, separate_norm_attack: bool = False, test_size: float = 0.3
):
    def extract_train_test(df, test_size=test_size):
        data, target = df.iloc[:, :-1], df["Label"]
        train_data, test_data, train_target, test_target = train_test_split(
            data, target, test_size=test_size, stratify=target
        )
        return train_data, test_data, train_target, test_target

    df = pd.read_csv(dataset_path, header=0, sep=",")
    for col in df.columns:
        if "Unnamed" in col:
            df = df.drop(col, axis=1)
        if "ms" in col and not "duration" in col:
            df = df.drop(col, axis=1)
    if training_with_attacks:
        if separate_norm_attack:
            df_normal = df[df["Label"] == 0]
            df_attack = df[df["Label"] == 1]
            (
                x_train_normal,
                x_test_normal,
                y_train_normal,
                y_test_normal,
            ) = extract_train_test(df_normal, test_size=test_size)
            x_attack, y_attack = df_attack.iloc[:, :-1], df_attack["Label"]
            return (
                x_train_normal,
                x_test_normal,
                x_attack,
                y_train_normal,
                y_test_normal,
                y_attack,
            )
        else:
            x_train, x_test, y_train, y_test = extract_train_test(df, test_size=test_size)
            x_test_normal, x_attack = x_test[y_test == 0], x_test[y_test == 1]
            y_test_normal, y_attack = y_test[y_test == 0], y_test[y_test == 1]

            return x_train, x_test_normal, x_attack, y_train, y_test_normal, y_attack
    else:
        x_train, x_test, y_train, y_test = extract_train_test(df, test_size=test_size)
        return x_train, x_test, y_train, y_test


    # scaler = MinMaxScaler()
    # preprocessed_dataset_scaled = scaler.fit_transform(df)

    # df_normalized = pd.DataFrame(preprocessed_dataset_scaled, columns=df.columns, index=df.index)
