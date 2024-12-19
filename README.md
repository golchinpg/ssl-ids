# Self-Supervised Contrastive Learning-based Intrusion Detection System

Welcome to the **SSL-IDS** repository! This project provides an intrusion detection system based on Self-Supervised Contrastive learning.
This repository is the code of paper **"SSCL-IDS: Enhancing Generalization of Intrusion Detection with Self-Supervised Contrastive Learning"** which was published at IFIP Networking Conference, 2024 (https://www.kom.tu-darmstadt.de/assets/178ce317-528a-4013-944e-1cdf00f8c852/GRH24.pdf). 

If you use any script of this repository, please cite the work:

P. Golchin, N. Rafiee, M. Hajizadeh, A. Khalil, R. Kundel and R. Steinmetz, "SSCL-IDS: Enhancing Generalization of Intrusion Detection with Self-Supervised Contrastive Learning," 2024 IFIP Networking Conference (IFIP Networking), Thessaloniki, Greece, 2024, pp. 404-412, doi: 10.23919/IFIPNetworking62109.2024.10619725.
keywords: {Training;Degradation;Transfer learning;Training data;Network intrusion detection;Telecommunication traffic;Contrastive learning;Network Intrusion Detection System;Machine Learning;Self-Supervised Learning},

## Prerequisites
- Required Python libraries (see `requirements.txt`)

## Getting Started

### Preprocessing: Traffic Conversion
The first step is to convert the traffic data from a PCAP file into feature files that can be used for training or testing with the models. Use the script `traffic_converter.py` to perform the conversion:

```bash
python traffic_converter.py test.pcap
```
Replace test.pcap with the name of your PCAP file.

After preprocessing, the script generates 46 flow features and saves them in the Dataset directory in CSV format. These files are ready for use in training or evaluating the SSCL-IDS model.

### Evaluating SSCL-IDS with Your Network Traffic Dataset
In this phase, you can evaluate the detection performance of SSCL-IDS for your network traffic dataset. Run the evaluation script:

```bash
python evaluation.py \
    --test_dataset_name "your_test_dataset.csv" \
    --batch_size 256 \
    --model_chkpt "your_checkpoint.pth" \
    --train_from_scratch False \
```

    - `--test_dataset_name`: Path to the CSV file of the test dataset.
    - `--batch_size`: Batch size for evaluation (default: 256).
    - `--model_chkpt`: Path to the model checkpoint file (default: `"scarf1_embdd_dim=45_lr=0.001_bs=2046_epochs=100_tempr=0.5_V=onlyctu13_cr_rt=0.4_ach_cr_rt=0.2_msk_rt0_ach_msk_rt0.pth"`).
    - `--train_from_scratch`: This will change the assumption of training data with only normal flows. If you only want to evaluate the SSCL-IDS with your dataset, set it False. If you wanna train the SSCL-IDS from scratch and then evaluate it set it as True.
The script will output the AUROC score and AUROC for maximum cosine similarity for the evaluation dataset.

### Fine-tune SSCL-IDS with Your Network Traffic Dataset
If you want to fine-tune SSCL-IDS models with additional MLP layers on your traffic network dataset, which has the ground truth labels (0: benign and 1:attack), it is possible to run the following script:
```bash
python fine_tune.py \
    --model_chkpt_path <path_to_model_checkpoint> \
    --dataset_dir <path_to_dataset> \
    --batch_size_fine_tuning 1024 \
    --num_epochs 30 \
    --learning_rate 0.001 \
    --apply_smote True \
```
    - `--model_chkpt_path`: Path to the model checkpoint file
    - `--dataset_dir`: Path to your network traffic dataset
    - `--batch_size_fine_tuning`: Batch size for training
    - `--num_epochs`: Number of training epochs
    - `--learning_rate`: Learning rate for the optimizer
    - `--apply_smote`: Apply SMOTE oversampling (if your network traffic dataset is imbalanced)

This script will output the classification report.

### Training SSCL-IDS with Your Data from Scratch
The **SSCL-IDS** is trained only with normal network traffic flows. If you want to train it from scratch with your normal network traffic, use the following script:
```bash
python train.py --dataset_path <path_to_your_prepared_dataset_for_training> \
    --batch_size 2046 \
    --epochs 100 \
    --lr 0.001 \
    --embedding_dim 45 \
    --temprature 0.5 \ 
    --version "experiment 1" \
    --positive_cr_list [0.3, 0.4] \
    --anchor_cr_list [0.1] \
    --positive_mr_list [0.3,0.5] \
    --anchor_mr_list [0] \
```

    - `--dataset_path (str): Path to the CSV dataset file.
    - `--batch_size (int): Batch size for training.
    - `--epochs (int): Number of training epochs.
    - `--lr (float): Learning rate for the optimizer.
    - `--embedding_dim (int): Dimensionality of the SCARF embedding space.
    - `--temprature (float): Temperature parameter for the NTXent loss.
    - `--positive_cr_list (list): list of Corruption rates for creating positive pairs.
    - `--anchor_cr_list (list): list of Corruption rates for anchors.
    - `--positive_mr_list (list): list of Mask rates for creating positive pairs.
    - `--anchor_mr_list (list): list of Mask rates for anchors.


