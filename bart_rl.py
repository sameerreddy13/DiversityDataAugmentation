from transformers import MaxLengthCriteria, TemperatureLogitsWarper
import torch
from utils import dev
class BartReinforce():
  '''
  BART for RL - specifically policy gradient with REINFORCE
  '''
  def __init__(self, model):
    self.model = model
    self.model = self.model.to(dev)
    # Contains actions from latest batch. List of token_id LongTensors
    self.actions = list()
    # Contains log_probs of actions from latest batch. FloatTensor(batch_size, num_steps) of log probs
    self.log_probs = None
    # if MULTI_GPU:
        # self.parallel_forward = torch.nn.DataParallel(self.model, device_ids=device_ids).to(dev)
    # else:
    self.pad_token_id = self.model.config.pad_token_id
    self.eos_token_id = self.model.config.eos_token_id
    self.bos_token_id = self.model.config.bos_token_id
    # This is 2 which equals <s/>, the eos_token_id
    self.decoder_start_token_id = self.model.config.decoder_start_token_id
    self.is_encoder_decoder = self.model.config.is_encoder_decoder

  @property
  def encoder(self):
    '''
    Getter for encoder
    '''
    return self.model.model.encoder

  def freeze_encoder_params(self):
    '''
    Freeze encoder params from updates
    '''
    for layer in self.encoder.parameters():
      layer.requires_grad= False

  def clear_episode_batch(self):
    '''
    Clear actions and log probs from last batch of episodes
    '''
    self.actions = list()
    self.log_probs = None

  def sample_policy(self, probs):
    '''
    Epsilon-greedy sampling from softmax distribution

    Args:
      probs :torch.FloatTensor of shape (batch_size, vocab_size): Softmax probs for token
    '''
    # epsilon-greedy (note this goes across batch), use uniform probs
    if torch.rand(1).item() < self.epsilon:
      sample_probs = torch.ones(probs.shape)/probs.shape[1]
    # use policy probs
    else:
      sample_probs = probs
    return torch.distributions.Categorical(sample_probs).sample()

  def run_step(self, probs, unfinished_sequences):
    '''
    Sample next tokens for batch and store actions and log probabilities
    '''
    next_tokens = self.sample_policy(probs).to(dev)
    next_tokens = next_tokens * unfinished_sequences + self.pad_token_id * (1 - unfinished_sequences)
    selected_log_probs = torch.log(probs.gather(1, next_tokens.unsqueeze(-1)))
    self.actions.append(next_tokens.cpu())
    if self.log_probs is None:
      self.log_probs = selected_log_probs
    else:
      self.log_probs = torch.cat((self.log_probs, selected_log_probs), 1)  
    return next_tokens

  def prepare_inputs_for_decoder(self, input_ids, model_kwargs):
    '''
    Run encoder and set up decoder input ids

    Returns:
      :torch.LongTensor of shape (batch_size, vocab_size): Decoder input ids
      :dict: Updated model_kwargs with encoder_outputs
    '''
    # Should be True for BART models. Run encoder and set up decoder inputs
    if self.is_encoder_decoder:
      encoder_input_ids, attention_mask = input_ids, model_kwargs['attention_mask']
      # Get encoder outputs of type BaseModelOutput
      self.encoder.to(dev)
      model_kwargs['encoder_outputs'] = self.encoder(input_ids=encoder_input_ids, attention_mask=attention_mask)
      model_kwargs = self.model._prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)
      # Set input_ids as decoder_input_ids
      if "decoder_input_ids" in model_kwargs:
          input_ids = model_kwargs.pop("decoder_input_ids")
      else:
          input_ids = self.model._prepare_decoder_input_ids_for_generation(
              input_ids, decoder_start_token_id=self.decoder_start_token_id, bos_token_id=self.bos_token_id
          )
      if "encoder_outputs" not in model_kwargs:
          raise ValueError("Make sure that `model_kwargs` include `encoder_outputs`.")
    return input_ids, model_kwargs


  def generate_episodes(self, batch, max_length=None, temperature=1.0, epsilon=0.1):
    '''
    Generate episodes over batch of sequences

    Args:
      batch :List[torch.Tensor]: Contains the below elements (in order)
        input_ids :shape (batch_size, seq_length): 
          Input sequence for generation.
        attention_mask :shape (batch_size, seq_length): 
          Attention mask.
        labels :shape (batch_size, ): 
          Label for each sequence.
      max_length :int: 
        Max output sequence length
      temperature :float: 
        Rescale logits before softmax by  `logits = logits/temperature`. Higher temperatures t result in 
        softer probability distribution which goes to uniform as t->infinity.
    '''
    # Set epsilon for this batch
    self.epsilon = epsilon
    model_kwargs = dict()
    input_ids, attention_mask, labels = batch
    input_ids, attention_mask, labels = input_ids.to(dev), attention_mask.to(dev), labels.to(dev)
    model_kwargs['attention_mask'] = attention_mask    
    input_ids, model_kwargs = self.prepare_inputs_for_decoder(input_ids, model_kwargs)
    max_length = max_length if max_length is not None else self.model.config.max_length      
    # For setting sequence length limit
    stopping_criteria = MaxLengthCriteria(max_length)
    # Get distribution pre_processing samplers
    logits_warper = TemperatureLogitsWarper(temperature)

    ## Generation ##
    # Keep track of which sequences are already finished
    ###
    # Initially unfinished_sequences = sequence of 1s with length batch size
    # and cur_len = tensor of shape (batch_size, 1) containing decoder_start_token_id.
    ####
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    cur_len = input_ids.shape[-1]
    while True:
      ## Run decoder for one time step ##
      # Dictionary with masks, decoder input ids, and encoder outputs
      model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
      # Seq2SeqLMOutput
      outputs = self.model(**model_inputs, return_dict=True)
      # Logits of shape (batch_size, 1, vocab_size) -> (batch_size, vocab_size)
      next_token_scores = logits_warper(input_ids, outputs.logits[:, -1, :])
      probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
      next_tokens = self.run_step(probs, unfinished_sequences)
      ## Update for next step ##
      # append next tokens
      input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
      # update past to past_key_values from outputs, attention mask should be same as from model_inputs
      model_kwargs = self.model._update_model_kwargs_for_generation(
        outputs, model_kwargs, is_encoder_decoder=self.is_encoder_decoder
      )
      # update length
      cur_len = cur_len + 1
      # If eos_token was found in one sentence, set sentence to finished
      unfinished_sequences = unfinished_sequences.mul((next_tokens != self.eos_token_id).long())
      # stop when each sentence is finished, or if we exceed the maximum length
      if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, None):
        break
    # print(f"Max Generation steps = {cur_len}")
    return input_ids