from utils import  run_train
import os, json
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hyperparameters = {
    "grid height": 16,
    "grid width": 16,
    "hidden size": 768,
    "decoder layers": 2,
    "encoder layers": 2,
    "encoder heads": 4,
    "decoder heads": 8,
    "latent channels": 128,
    "viz channels": 3,
    "context window": 4096,
    "mask ratio": 0.05
}

with open("val_texts.json", "r", encoding="utf-8") as f:
    VAL_TEXTS = json.load(f)

run_train(device, hyperparameters, 5000, 5e-4, VAL_TEXTS)
