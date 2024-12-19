# Self-Supervised Contrastive Learning-based Intrusion Detection System

Welcome to the **SSL-IDS** repository! This project provides an intrusion detection system based on Self-Supervised Contrastive learning.
This repository is the code of paper **"SSCL-IDS: Enhancing Generalization of Intrusion Detection with Self-Supervised Contrastive Learning"** which was published at IFIP Networking Conference, 2024 (https://www.kom.tu-darmstadt.de/assets/178ce317-528a-4013-944e-1cdf00f8c852/GRH24.pdf). 

## Abstract
With the increasing diversity and complexity of cyber attacks on computer networks, there is a growing demand for Intrusion Detection Systems (IDS) that can accurately categorize new unknown network flows. Machine learning- based IDS (ML-IDS) offers a potential solution by learning underlying network traffic characteristics. However, ML-IDS encounters performance degradation in predicting the traffic with a different distribution from its training dataset (i.e., new unseen data), especially for attacks that mimic benign (non- attack) traffic (e.g., multi-stage attacks). Diversity in attack types intensifies the lack of labeled attack traffic, which leads to reduced detection performance and generalization capabilities of ML-IDS. The generalization refers to the model's capacity to identify new and unseen samples, even in cases where their distribution deviates from the training data used for the ML-IDS. To address these issues, this paper introduces SSCL-IDS, a Self-Supervised Contrastive Learning IDS designed to increase the generalization of ML-IDS. The proposed SSCL-IDS is exclusively trained on benign flows, enabling it to acquire a generic representation of benign traffic patterns and reduce the reliance on annotated network traffic datasets. The proposed SSCL-IDS demonstrates a substantial improvement in detection and generalization across diverse datasets compared to supervised (over 27%) and unsupervised (over 15%) baselines due to its ability to learn a more effective representation of benign flow attributes. Additionally, by leveraging transfer learning with SSCL-IDS as a pretrained model, we achieve AUROC scores surpassing 80% when fine-tuning with less than 20 training samples. Without fine-tuning, the average AUROC score across different datasets resembles random guessing.

<img src="/Screenshot 2024-12-19 at 17.29.29.png" alt="SSCL-IDS Example"/>

## Citation
If you use any script of this repository, please cite the work:

P. Golchin, N. Rafiee, M. Hajizadeh, A. Khalil, R. Kundel and R. Steinmetz, "SSCL-IDS: Enhancing Generalization of Intrusion Detection with Self-Supervised Contrastive Learning," 2024 IFIP Networking Conference (IFIP Networking), Thessaloniki, Greece, 2024, pp. 404-412, doi: 10.23919/IFIPNetworking62109.2024.10619725.
keywords: {Training;Degradation;Transfer learning;Training data;Network intrusion detection;Telecommunication traffic;Contrastive learning;Network Intrusion Detection System;Machine Learning;Self-Supervised Learning},

## Table of Content

- [**Prerequisites**](##prerequisites)
- [**Create a Compatible Network Traffic Dataset**](##preprocessing)
- [**Evaluating SSCL-IDS with Your Network Traffic Dataset**](##evaluation )
- [**Fine-tune SSCL-IDS with Your Network Traffic Dataset**](##fine_tune)
- [**Training SSCL-IDS from Scratch**](##training)
  
## <span id="prerequisites">Prerequisites</span>
- Required Python libraries (see `requirements.txt`)

## Getting Started

## <span id="preprocessing">Preprocessing: Traffic Conversion</span>
The first step is to convert the traffic data from a PCAP file into feature files that can be used for training or testing with the models. Use the script `traffic_converter.py` to perform the conversion:

```bash
python traffic_converter.py test.pcap
```
Replace test.pcap with the name of your PCAP file.

After preprocessing, the script generates 46 flow features and saves them in the Dataset directory in CSV format. These files are ready for use in training or evaluating the SSCL-IDS model.

## <span id="evaluation">Evaluating SSCL-IDS with Your Network Traffic Dataset</span>
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

## <span id="fine_tune">Fine-tune SSCL-IDS with Your Network Traffic Dataset</span>

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

## <span id="training">Training SSCL-IDS with Your Data from Scratch</span>
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
    --chkpt_path <checkpoint_path> \
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
    - `--chkpt_path: The directory in which you want to save the checkpoint


