import torch
import torch.utils.data as torch_data
from typing import List, Dict, Union
import datasets

class TokenizerWrapper():
    '''
    Wrapper for tokenizer
    '''
    default_encode_config = dict(
        add_special_tokens = True, 
        padding=True, truncation=True, 
        return_tensors='pt'
    )
    def __init__(self, tokenizer, encode_config=default_encode_config):
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
        def decode_id_seq(s):
            decoded = self.t.decode(s, skip_special_tokens=skip_special_tokens)
            try:
                return bytes(decoded, 'utf8').decode('latin1', 'ignore')
            except UnicodeEncodeError as e:
                import pdb; pdb.set_trace()
                _ = 1
        return [decode_id_seq(e) for e in encodings]

class SSTLoader():
    '''
    Data loading for sst data. Does encoding etc 

    Params:
    lim:        Use lim of -1 to use all samples. Otherwise uses up to lim samples.  
    batch_size: Batch size for loaders
    tokenizer:  Tokenizer

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
        Features:
        {
            'sentence': Value(dtype='string', id=None), 
            'label': ClassLabel(num_classes=2, names=['negative', 'positive'], 
            'idx': Value(dtype='int32', id=None)
        }
    '''
    def __init__(self, tokenizer: TokenizerWrapper = None, batch_size: int = 8, lim: int = -1):
        self.lim = lim
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.__load_sst_binary()



    def __load_sst_binary(self):
        '''
        Set sst train, val and test data
        '''
        # Training data from glue (not tokenized) containing keys ('sentence', 'idx', 'label')  
        raw_dss = datasets.load_dataset("sst")
        dss = [raw_dss['train'], raw_dss['validation'], raw_dss['test']]
        for i, ds in enumerate(dss):
            dss[i] = self.preprocess_dataset(ds)
        self.train_dataset, self.val_dataset, self.test_dataset  = dss

    def __create_torch_dataloader(self, sents, labels, shuffle) -> torch_data.DataLoader:
        encodings = self.tokenizer.encode(sents) 
        torch_ds = torch_data.TensorDataset(
                encodings['input_ids'], 
                encodings['attention_mask'], 
                labels
            )
        return torch_data.DataLoader(torch_ds, batch_size=self.batch_size, shuffle=shuffle)

    def __sst_to_loader(self, d, s):
        return self.__create_torch_dataloader(d['sentence'], torch.as_tensor(d['label'], dtype=torch.long), s)

    @staticmethod    
    def __sentiment_to_binary(example):
        example['label'] = round(example['label'])
        return example

    def __preprocess_example_sents(self, example):
        example['sentence'] = self.preprocess_sentence(example['sentence'])
        example['sentence'] = example['sentence'] + " <s/> " + str(int(example['label']))
        return example

    @staticmethod 
    def preprocess_sentence(s: str):
        # remove_punc = "()-[]{};:\",<>/@#$%^&*_~`"
        # s = s.lower().strip()
        # s = ''.join([c for c in s if c not in remove_punc])
        return s

    def preprocess_dataset(self, ds: datasets.Dataset) -> datasets.Dataset:
        if self.lim > 0:
            ds = ds.select(range(self.lim))
        ds = ds.map(self.__sentiment_to_binary)
        ds = ds.map(self.__preprocess_example_sents)
        return ds

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

if __name__ == '__main__':
    from bart_rl import load_bart_tokenizer
    sst2 = SSTLoader(TokenizerWrapper(load_bart_tokenizer()), lim=100)
    # train_loader = sst2.get_train_loader()



