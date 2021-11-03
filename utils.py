from transformers import (
  BartTokenizerFast, BartForConditionalGeneration, 
  AdamW
)
import torch

if torch.cuda.is_available():
  dev = "cuda"
else:
  dev = "cpu"
print(f'Using device {dev}')
# MODEL_KEY = 'sshleifer/distilbart-cnn-6-6'
MODEL_KEY = 'facebook/bart-base'
def load_bart_model(layers=None):
  '''
  Load pretrained BartForConditionalGeneration 
  '''
  config = dict()
  if layers:
    assert type(layers) == int
    config = dict(encoder_layers=layers, decoder_layers=layers)
  model = BartForConditionalGeneration.from_pretrained(MODEL_KEY, **config)
  return model

def load_bart_tokenizer():
  '''
  Load pretrained bart tokenizer 
  '''
  return BartTokenizerFast.from_pretrained(MODEL_KEY)

def get_adamw(model, lr=1e-5, eps=1e-8):
  '''
  Get adamw optimizer
  '''
  return AdamW(model.parameters(), lr=lr, eps=eps)


