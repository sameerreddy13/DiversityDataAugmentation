import torch
import torch.utils.data as torch_data
from typing import List, Dict, Union
import datasets

# def tokenize_string(s):


class TokenizerWrapper():
    '''
    Wrapper for tokenizer
    '''
    def __init__(self, tokenizer, encode_config={}):
        self.t = tokenizer
        self.encode_config = encode_config

    @property
    def mask_token_id(self):
        return self.t.mask_token_id

    def encode(self, sentences: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        '''
        Return tokenized sentences

        Returns
          encodings: dict with 'input_ids' and 'attention_mask'
        ''' 
        return self.t(sentences, **self.encode_config)

    def decode(self, encodings: torch.LongTensor, skip_special_tokens=True) -> List[str]:
        '''
        Return decoded sentences from token id sequences
        '''
        return [self.t.decode(s, skip_special_tokens=skip_special_tokens).encode('utf-8').strip() for s in encodings]

class SSTLoader():
    '''
    Data loading for sst data. Does encoding etc 

    Params:
    lim: Use lim of -1 to use all samples. Otherwise uses up to lim samples.  
    batch_size: Batch size for loaders
    tokenizer: Tokenizer

    Returns: 
    batches from TensorDataset with elements: input_ids, attention_mask, labels

    Dataset reference:
    DatasetDict({
        train: Dataset({
            features: ['sentence', 'label', 'idx'],
            num_rows: 67349
        })
        validation: Dataset({
            features: ['sentence', 'label', 'idx'],
            num_rows: 872
        })
        test: Dataset({
            features: ['sentence', 'label', 'idx'],
            num_rows: 1821
        })
    })
    '''
    def __init__(self, tokenizer: TokenizerWrapper, batch_size: int, lim: int = -1):
        self.lim = lim
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.__load_sst2()

    def __load_sst2(self):
        '''
        Return sst2 train and test data
        '''
        # Training data from glue (not tokenized) containing keys ('sentence', 'idx', 'label')  
        raw_datasets = datasets.load_dataset("glue", "sst2")
        train_dataset = raw_datasets['train']
        val_dataset = raw_datasets['validation']
        test_dataset = raw_datasets['test']
        if self.lim > 0:
            train_dataset = train_dataset.select(range(self.lim))
            val_dataset = val_dataset.select(range(self.lim))
            test_dataset = test_dataset.select(range(self.lim))

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def __create_torch_dataloader(self, sents, labels, shuffle) -> torch_data.DataLoader:
        encodings = self.tokenizer.encode(sents) 
        torch_ds = torch_data.TensorDataset(
                encodings['input_ids'], 
                encodings['attention_mask'], 
                labels
            )
        return torch_data.DataLoader(torch_ds, batch_size=self.batch_size, shuffle=shuffle)

    def __sst_to_loader(self, d, s):
        return self.__create_torch_dataloader(d['sentence'], torch.as_tensor(d['label']), s)
    
    def get_train_loader(self, shuffle=True):
        '''
        Encodes dataset and return train dataloader (data batches)
        '''
        return self.__sst_to_loader(self.train_dataset, shuffle)

    def get_val_loader(self, shuffle=True):
        '''
        Encodes dataset and return val dataloader (data batches)
        '''
        return self.__sst_to_loader(self.val_dataset, shuffle)

    def get_test_loader(self, shuffle=True):
        '''
        Encodes dataset and return test dataloader (data batches)
        '''
        return self.__sst_to_loader(self.test_dataset, shuffle)

