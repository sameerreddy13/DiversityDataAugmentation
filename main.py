import torch
import transformers
from transformers import BartTokenizerFast, PreTrainedTokenizerFast, BartModel, BartForConditionalGeneration
import datasets
import pdb 
if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"


print(f'Using device {dev}')
# MODEL_KEY = 'facebook/bart-base'
MODEL_KEY = 'sshleifer/distilbart-cnn-12-3'

def load_sst(start=0, end=10):
  return datasets.load_dataset('glue', 'sst2', split=f'train[{start}:{end}]')

def load_model():
  layers = 1 # default is 12
  config = dict(encoder_layers=layers, decoder_layers=layers)
  model = BartForConditionalGeneration.from_pretrained(MODEL_KEY, **config)
  model = model.to(dev)
  return model

def tokenize_sst(sst_dataset):
  tokenizer = BartTokenizerFast.from_pretrained(MODEL_KEY)
  ftok = lambda z: tokenizer(z, truncation=True, padding='longest', return_tensors='pt')
  tokenized = ftok([s for s in sst_dataset['sentence']])
  return tokenized, tokenizer

def get_inputs(encodings):
  input_ids = encodings.input_ids
  input_ids = input_ids.to(dev)
  attention_mask = encodings.attention_mask
  attention_mask = attention_mask.to(dev)
  return {
    "input_ids": input_ids,
    "attention_mask": attention_mask
  }


if __name__ == '__main__':
  sst_dataset = load_sst()
  print(sst_dataset.features)
  encodings, tokenizer = tokenize_sst(sst_dataset)
  model = load_model()
  print('\n\nMODEL SIZE')
  num_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print("No. params:", num_p)
  print("\nINPUTS")
  inputs = get_inputs(encodings)
  print("input ids shape:", inputs['input_ids'].shape)
  print("\nOUTPUT LOGITS")
  # Seq2SeqLMOutput Type
  output = model(**inputs)
  print(output.logits.shape)
  gen_output = model.generate(**inputs, max_length=100, output_scores=True, return_dict_in_generate=True)
  print("\nSEQUENCES")
  print(gen_output['sequences'].shape)
  print(type(gen_output['sequences']))
  # print([s.shape for s in gen_output['scores']])
  print(gen_output['sequences_scores'])
  print(tokenizer.decode(gen_output['sequences'][0]))
