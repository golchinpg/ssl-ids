# Self-Supervised Contrastive Learning-based Intrusion Detection System

Welcome to the **SSL-IDS** repository! This project provides tools to preprocess network traffic data in PCAP format and convert it into feature files compatible with the models described in the **SSCL-IDS** paper.

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
python evaluation.py --test_dataset_name "your_test_dataset.csv" --batch_size 256 --model_chkpt "your_checkpoint.pth"
```

    - `--test_dataset_name`: Path to the CSV file of the test dataset.
    - `--batch_size`: Batch size for evaluation (default: 256).
    - `--model_chkpt`: Path to the model checkpoint file (default: `"scarf1_embdd_dim=45_lr=0.001_bs=2046_epochs=100_tempr=0.5_V=onlyctu13_cr_rt=0.4_ach_cr_rt=0.2_msk_rt0_ach_msk_rt0.pth"`).

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

    


