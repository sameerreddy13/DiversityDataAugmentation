from transformers import (
  BartTokenizerFast, BartForConditionalGeneration, 
  AdamW
)
import torch
import numpy as np
from typing import Union

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# SEED
seed_everything(42)
# DEVICE
if torch.cuda.is_available():
  DEV = "cuda"
else:
  DEV = "cpu"
LOG_EPS = 0 # 1e-8
MODEL_KEY = 'sshleifer/distilbart-cnn-6-6'
# MODEL_KEY = 'facebook/bart-base'

def get_adamw(model, lr=1e-5, eps=1e-8, wd=0.):
  '''
  Get adamw optimizer
  '''
  return AdamW(model.parameters(), lr=lr, eps=eps, weight_decay=wd)

def convert_to_numpy(arr: Union[np.ndarray, torch.Tensor, list]) -> np.ndarray:
  '''
  Convert array to numpy
  '''
  if isinstance(arr, torch.Tensor):
    arr = arr.cpu().numpy()
  if isinstance(arr, list):
    arr = np.asarray(arr)
  return arr

def flat_accuracy(logits, labels) -> float:
  '''
  Helper for accuracy between logits and labels in Tensor format or numpy arrays
  '''
  logits, labels = map(convert_to_numpy, (logits, labels))
  pred_flat = np.argmax(logits, axis=1).flatten()
  labels_flat = labels.flatten()
  return np.sum(pred_flat == labels_flat) / len(labels_flat)


