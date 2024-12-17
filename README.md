# Self-Supervised Contrastive Learning-based Intrusion Detection System

Welcome to the **SSL-IDS** repository! This project provides tools to preprocess network traffic data in PCAP format and convert it into feature files compatible with the models described in the **SSCL-IDS** paper.

## Getting Started

### Step 1: Traffic Conversion
The first step is to convert the traffic data from a PCAP file into feature files that can be used for training or testing with the models. Use the script `traffic_converter.py` to perform the conversion:

```bash
python traffic_converter.py test.pcap
```
Replace test.pcap with the name of your PCAP file.

After preprocessing, the script generates 46 flow features and saves them in the Dataset directory in CSV format. These files are ready for use in training or evaluating the SSCL-IDS model.
