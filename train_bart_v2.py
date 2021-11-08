import torch
from transformers import logging
from tqdm import tqdm
import pdb
from typing import List, Dict, Union
import os
import numpy as np
from utils import load_bart_model, load_bart_tokenizer, get_adamw, DEV
from data import SSTLoader, TokenizerWrapper
from bart_rl import BartReinforce
logging.set_verbosity_error()
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def mask_span(seq_ids: torch.LongTensor, tokenizer: TokenizerWrapper, span_range=(4, 5)):
      start, end = 2, 4
      for i in range(start, end):
            seq_ids[i] = tokenizer.mask_token_id # mask one token

def compute_rewards(inputs: List[str], outputs: List[str]) -> torch.FloatTensor:
    '''
    Get rewards between input and output sentences

    Returns:
    :torch.FloatTensor of shape (batch_size, 1): Batch size length tensor of rewards
    '''
    rewards = torch.rand(len(inputs))
    R = torch.as_tensor(rewards).unsqueeze(-1).to(DEV)
    Rb = R.mean()
    return R - Rb

def main():
    # Training config
    batch_size = 1
    epochs = 10
    print_interval = epochs // 5
    lim = 5 
    episode_config = dict(
        epsilon=0.1, 
        temperature=0.7, 
        topk=100
    )
    encode_config = dict(
        add_special_tokens = True, 
        padding=True, truncation=True, 
        return_tensors='pt'
    )
    # Load data, model, optimizer
    bart = load_bart_model()
    optimizer = get_adamw(bart, lr=1e-6)
    tokenizer = TokenizerWrapper(load_bart_tokenizer(), encode_config)
    # Load data into batches
    sst2 = SSTLoader(tokenizer, batch_size=batch_size, lim=lim)
    train_loader = sst2.get_train_loader()
    # Init seq2seq RL model
    rl_model = BartReinforce(bart)
    rl_model.freeze_encoder_params()
    # Train
    for epoch in range(epochs):
        bart.train()    
        with tqdm(train_loader, unit="batch", disable=False) as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            for batch_num, batch in enumerate(tepoch):
                optimizer.zero_grad()
                # Generate episodes from batch of inputs
                max_in_len = batch[0].shape[1]
                out = rl_model.generate_episodes(batch, max_length=max_in_len+1, **episode_config)
                # Compute rewards
                input_sentences, output_sentences = map(tokenizer.decode, (batch[0], out))
                r = compute_rewards(input_sentences, output_sentences)
                loss_batch = (r * -rl_model.log_probs).sum(1)
                loss = loss_batch.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(bart.parameters(), 1.0)
                optimizer.step()
                rl_model.clear_episode_batch()  
                if epoch % print_interval == 0:
                    print("Rewards:", r.squeeze().cpu())
                    [print("Input:", i, '\n', 'Output', o, '\n') for i,o in zip(input_sentences, output_sentences)]

if __name__ == '__main__':
  main()