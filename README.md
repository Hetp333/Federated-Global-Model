# Federated Global Model for Hospital Readmission Prediction

This repository contains the trained global MLP model from the research project **“A Hyperledger Fabric-Based Architecture for Privacy-Aware and Auditable Federated Learning in Healthcare.”**

The model predicts whether a patient will be readmitted within 30 days. It was trained using the Diabetes 130-US Hospitals dataset in a simulated federated learning environment with three participating institutions.

## Training Overview

* **Framework:** PyTorch
* **Aggregation:** Federated Averaging (FedAvg)
* **Communication rounds:** 6
* **Local epochs per round:** 1
* **Optimizer:** Adam
* **Learning rate:** 0.0003

Hyperledger Fabric coordinates training rounds and records model references, while IPFS provides off-chain model storage.

## Download and Inspect

Download the `.pt` file from this repository. To inspect its parameters:

```python
import torch

# Replace with the downloaded file's name if different.
model_path = "global_model.pt"

state_dict = torch.load(
    model_path,
    map_location="cpu",
    weights_only=True,
)

for name, tensor in state_dict.items():
    print(f"{name}: shape={tuple(tensor.shape)}")
```

## Intended Use

This model is provided for research and educational use. It was evaluated in a simulated setting and is not intended for clinical decision-making.
