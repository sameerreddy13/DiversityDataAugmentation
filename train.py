import torch
from torch import optim
import transformers
from transformers import (BartTokenizerFast, PreTrainedTokenizerFast, BartModel, BartForConditionalGeneration, 
  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList, MaxLengthCriteria,
  AutoModelForSeq2SeqLM)
from transformers.generation_utils import GenerationMixin
import datasets
from fuzzywuzzy import fuzz
gm = GenerationMixin

import pdb 
if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
# dev = "cpu"

print(f'Using device {dev}')
# MODEL_KEY = 'sshleifer/distilbart-cnn-12-3'
MODEL_KEY = 'facebook/bart-base'

def load_sst(start=0, end=10):
  return datasets.load_dataset('glue', 'sst2', split=f'train[{start}:{end}]')

def load_model():
  layers = 4 # default is 12
  config = dict(encoder_layers=layers, decoder_layers=layers)

  model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_KEY, **config)
  model = model.to(dev)
  return model

def tokenize_sentences(sentences):
  tokenizer = BartTokenizerFast.from_pretrained(MODEL_KEY)
  ftok = lambda z: tokenizer(z, truncation=True, padding='longest', return_tensors='pt')
  tokenized = ftok([s for s in sentences])
  tokenized.input_ids = tokenized.input_ids.to(dev)
  tokenized.attention_mask = tokenized.attention_mask.to(dev)

  return tokenized, tokenizer

def reward_match(s_in, s_out):
  '''
  Reward based on fuzzy match ratio. Encourage similarity (toy example)
  '''
  r = fuzz.ratio(s_in, s_out) / 100
  return r

def get_inputs(model, input_ids):
  decoder_start_token_id = model.config.decoder_start_token_id
  bos_token_id = model.config.bos_token_id
  model_kwargs = dict()
  # prepare attention mask and encoder output
  model_kwargs["attention_mask"] = gm._prepare_attention_mask_for_generation(
      model, input_ids, pad_token_id, eos_token_id)
  encoder_input_ids = input_ids if model.config.is_encoder_decoder else None
  if model.config.is_encoder_decoder:
      model_kwargs = gm._prepare_encoder_decoder_kwargs_for_generation(model, input_ids, model_kwargs)

      input_ids = gm._prepare_decoder_input_ids_for_generation(
          model, input_ids, decoder_start_token_id=decoder_start_token_id, bos_token_id=bos_token_id)

  model_kwargs["use_cache"] = None

  logits_processor = gm._get_logits_processor(
      model,
      repetition_penalty=None,
      bad_words_ids=None,
      min_length=1,
      max_length=40,
      eos_token_id=None,
      prefix_allowed_tokens_fn=None,
      num_beam_groups=None,
      diversity_penalty=None,
      no_repeat_ngram_size=None,
      encoder_no_repeat_ngram_size=None,
      encoder_input_ids=encoder_input_ids,
      forced_bos_token_id=None,
      forced_eos_token_id=None,
      num_beams=None,
      remove_invalid_values=True)
  return input_ids, logits_processor, model_kwargs

def compute_sequence_score(sequence_ids, sequence_scores):
  '''
  sequence_ids: sequence token ids of shape (max_length, )
  sequence_scores: sequence_scores of shape (max_length - 1, vocab_size) containing pre-softmax scores
  '''
  sequence_scores = torch.log_softmax(sequence_scores, 1)
  policy_scores = []
  for i, id in enumerate(sequence_ids[1:]):
    # get score for chosen action i.e which token was generated
    score = sequence_scores[i][id] 
    policy_scores.append(score)
  # We should have a score for each token in the sequence
  return torch.tensor(policy_scores, requires_grad=True, device=dev).sum()
  
# def discount_r(rewards, gamma=0.99):



if __name__ == '__main__':
  LR = 1e-3
  USE_AMS = False
  EPOCHS = 100
  interval = 10
  TEST = False
  sst_dataset = load_sst(end=1)
  if TEST:
    test_sents = ['test', 'testing', 'test z', 'test y', 'test w']
    encodings, tokenizer = tokenize_sentences(['test', 'testing', 'test z', 'test y', 'test w']) 
  else:
    encodings, tokenizer = tokenize_sentences(sst_dataset['sentence'])
  model = load_model()
  pad_token_id = model.config.pad_token_id
  eos_token_id = model.config.eos_token_id
  if TEST:
    batches = [(list(range(len(test_sents))), test_sents, encodings)]
  else:
    batches = [(sst_dataset['idx'], sst_dataset['sentence'], encodings)]
  optimizer = optim.Adam(model.parameters(), lr=LR, amsgrad=USE_AMS)
  for epoch in range(EPOCHS):
    for b in batches:
      optimizer.zero_grad()
      indices, sentences, e = b
      inputs = get_inputs(model, e.input_ids)
      input_ids, logits_processor, model_kwargs = inputs
      # pdb.set_trace()
      # model(input_ids, )
      # outputs = gm.sample(
      #     model,
      #     input_ids,
      #     logits_processor=logits_processor,
      #     pad_token_id=pad_token_id,
      #     eos_token_id=eos_token_id,
      #     output_scores=True,
      #     return_dict_in_generate=True,
      #     **model_kwargs)
      # input_ids = X['input_ids']
      # X['decoder_input_ids'] = input_ids
      # del X['input_ids']
      # prev_output_tokens = input_ids.clone()
      # index_of_eos = (input_ids.ne(model.config.pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
      # prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
      # prev_output_tokens[:, 1:] = input_ids[:, :-1]
      # logits_processor = LogitsProcessorList([
      #   MinLengthLogitsProcessor(10, eos_token_id=model.config.eos_token_id)
      # ])
      # output = model(**X, return_dict=True)
      # eo = output['encoder_last_hidden_state'].to(dev)
      # X['encoder_outputs'] = output

      # print(X.keys())
      # pred = model.generate(**X)

      # pred = model.greedy_search(
      #   input_ids, **X, logits_processor=logits_processor
      # )
      loss = model(input_ids=e.input_ids, labels=e.input_ids).loss
      if epoch % interval == 0:
        # gen_sentences = [tokenizer.decode(s, skip_special_tokens=True).encode('utf-8') for s in outputs['sequences']]
        gen_sentences = model.generate(input_ids=e.input_ids, attention_mask=e.attention_mask)
        gen_sentences = [tokenizer.decode(s, skip_special_tokens=True).encode('utf-8') for s in gen_sentences]
      # get generated sequences of ids and reshape for selecting log probs corresponding to actions
      # seq = outputs['sequences'][:,1:]
      # seq = seq.reshape((seq.shape[1], -1))
      # log_probs = torch.log_softmax(torch.stack(outputs['scores'], dim=0).squeeze(), dim=1)
      # selected_probs=torch.gather(log_probs, 1, seq)
      # rewards = []
      # for s_in, s_out in zip(sentences, gen_sentences):
      #   rewards.append(reward_match(s_in, s_out.decode('utf-8')))
      # rewards = torch.tensor(rewards, requires_grad=False, device=dev)
      # rewards = rewards
      # loss = (rewards * -selected_probs).mean()


      # loss = 
      loss.backward()
      optimizer.step()

      if epoch % interval == 0:
        print(f"--------------------------EPOCH {epoch}--------------------------")
        # print("REWARD/SCORES:", rewards, selected_probs)
        print("LOSS:", loss)
        print("GENERATED:", gen_sentences)
        print("TARGETS:", sentences)
        # print(log_probs, list(model.named_parameters())[:3])
