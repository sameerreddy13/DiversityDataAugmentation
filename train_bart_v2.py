import torch
from transformers import logging
from transformers import trainer_utils
from tqdm import tqdm
from typing import List, Dict, Union
import os
import numpy as np
from utils import get_adamw, DEV
from data import SSTLoader, TokenizerWrapper
from bart_rl import BartReinforce, load_bart_model, load_bart_tokenizer
from reward import RewardWrapper
import argparse
SEED = 42

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, help="GPU id")
    return p.parse_args()

def mask_span(seq_ids: torch.LongTensor, tokenizer: TokenizerWrapper, span_range=(4, 5)):
      start, end = 2, 4
      for i in range(start, end):
            seq_ids[i] = tokenizer.mask_token_id # mask one token

def compute_rewards(reward_f: RewardWrapper, inputs: List[str], 
                    outputs: List[str], labels: torch.LongTensor, 
                    subtract_mean: bool = False, verbose = False) -> torch.FloatTensor:
    '''
    Get rewards between input and output sentences

    Returns:
    :torch.FloatTensor of shape (batch_size, 1): Batch size length tensor of rewards
    '''
    rewards = []
    for s1, s2, label in zip(inputs, outputs, labels.numpy()):
        r_dict = reward_f.compute_rewards(s1, s2, label)
        editd =     r_dict['edit_distance']     # + 0.0
        # iou =       r_dict['iou_ungrams']       - 0.5
        es =        r_dict['embed_similarity']  - 0.7
        con =       r_dict['clf_consistency']
        alpha = 0.1
        r1 = (1-alpha) * (con + es) 
        # r2 = alpha * editd
        r2 = 0
        rewards.append(r1 + r2)
        # print(con, es, r1)
        # print(r2)
    rewards = torch.as_tensor(rewards, device=reward_f.device)
    if subtract_mean and rewards.shape[0] > 1:
        rewards = rewards - rewards.mean()
    return rewards

def call_config_functions():
    trainer_utils.set_seed(SEED)
    torch.autograd.set_detect_anomaly(True)
    logging.set_verbosity_error()


def main():
    args = parse_args()
    call_config_functions()
    device = DEV
    if args.gpu is not None:
        device = f"cuda:{args.gpu}"

    # Training config
    batch_size = 1
    epochs = 100
    print_interval = 10 # epochs // 5
    verbose = False
    use_tqdm = True
    lim = 10
    reward_baseline_sub_mean = True
    # Model config
    freeze_encoder_params = True
    optim_config = dict(
        lr=5e-6,
        wd=0.01
    )
    episode_config = dict(
        epsilon=0.00, 
        temperature=0.7, 
        topk=200,
        min_length=15,
        verbose=verbose
    )
    max_out_len = lambda input_len: int(input_len * 1.3)

    # Load data, model, optimizer
    bart = load_bart_model()
    optimizer = get_adamw(bart, **optim_config)
    tokenizer = TokenizerWrapper(load_bart_tokenizer())
    # Load data into batches
    sst2 = SSTLoader(tokenizer, batch_size=batch_size, lim=lim)
    train_loader = sst2.get_train_loader()

    # Init seq2seq RL model and reward functions
    rl_model = BartReinforce(bart, device=device)
    if freeze_encoder_params:
        rl_model.freeze_encoder_params()
    reward_f = RewardWrapper(device=device)
    # Train
    for epoch in range(epochs):
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            # Train epoch
            bart.train()    
            for batch_num, batch in enumerate(tepoch): # batch = Tuple(input_ids, attention_mask, labels)
                # Generate episodes from batch of input
                max_in_len = batch[0].shape[1]
                out = rl_model.generate_episodes(batch, max_length=max_out_len(max_in_len), **episode_config)
                # Decode sequences to sentence strings
                input_sentences, output_sentences = map(tokenizer.decode, (batch[0], out))
                output_sentences = [sst2.preprocess_sentence(s) for s in output_sentences]
                # Compute rewards
                r = compute_rewards(reward_f, input_sentences, output_sentences, 
                                    labels=batch[2], subtract_mean=reward_baseline_sub_mean, verbose=verbose)
                # Compute loss
                # rl_model.log_probs = torch.nan_to_num(rl_model.log_probs, neginf=0)
                # loss_batch = (r.unsqueeze(-1) * -rl_model.log_probs).sum(1)
                # loss = loss_batch.mean()
                loss = torch.tensor([0.0], requires_grad=True, device=device)
                for i in range(batch_size):
                    z = -rl_model.log_probs[i]
                    z = z[~torch.isinf(z)]
                    loss = loss + (r[i] * z).sum()
                loss = loss/batch_size
                # import pdb; pdb.set_trace()
                # Backward
                loss.backward()
                # try:
                #     torch.nn.utils.clip_grad_norm_(bart.parameters(), 1.0, error_if_nonfinite=True)
                # except Exception as e:
                #     print(list(bart.parameters())[-1].grad)
                #     import pdb; pdb.set_trace()
                #     _ = 1
                optimizer.step()
                # Clear state
                rl_model.clear_episode_batch()  
                optimizer.zero_grad()
                # Print info to stdout
                if print_interval > 0 and epoch % print_interval == 0:
                    [
                        tqdm.write(f"Input:\n{i}\nOutput:\n{o}\n") for i,o in zip(input_sentences, output_sentences)
                    ]
                    tqdm.write(f"Loss: {loss.item()}")
                    tqdm.write(f"Rewards: {r.squeeze().cpu()}")
                    tqdm.write(f"{'-'*15}")

if __name__ == '__main__':
  main()