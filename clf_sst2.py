from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import utils
from data import SSTLoader, TokenizerWrapper
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from typing import Union, List

class DistilBertSST(nn.Module):
	'''
	Get finetuned checkpoint of distilbert on sst2
	'''
	model_key = 'distilbert-base-uncased-finetuned-sst-2-english'
	def __init__(self, device=utils.DEV):
		super().__init__()
		self.device = device
		self.model = DistilBertForSequenceClassification.from_pretrained(self.model_key).to(self.device)
		self.tokenizer = TokenizerWrapper(
			DistilBertTokenizerFast.from_pretrained(self.model_key),
			{
				'return_tensors': 'pt', 
				'padding': True,
				'truncation': True		
			}
		)

	def forward(self, *args, **kwargs):
		return self.model(*args, **kwargs)

	@torch.no_grad()
	def predict_on_text(self, text: Union[str, List[str]]) -> np.ndarray:
		'''
		Get predicted labels from applying model to text or texts.
		Let N := number of input texts and C := the number of classes

		Returns: 
			:np.ndarray of shape (N, ): 	Predicted labels
			:np.ndarray of shape (N, C):	Probabilities for each class
		'''
		self.model.eval()
		encodings = self.tokenizer.encode(text)
		outputs = self(encodings['input_ids'].to(self.device), encodings['attention_mask'].to(self.device))	
		return (
				np.argmax(utils.convert_to_numpy(outputs.logits), axis=1).flatten(),
				nn.functional.softmax(outputs.logits, dim=-1)
		)

def run_validate_model():
	model = DistilBertSST()
	model.to(self.device)
	sst2 = SSTLoader(model.tokenizer, batch_size=128, lim=-1)
	train_loader, val_loader, test_loader = sst2.get_train_loader(), sst2.get_val_loader(), sst2.get_test_loader()

	print(f"Testing on {len(val_loader)*val_loader.batch_size} inputs")
	model.eval()
	with tqdm(val_loader, unit="batch") as pbar:
		for batch in pbar:
			accs = []
			with torch.no_grad():
				batch = [data.to(self.device) for data in batch]
				outputs = model(batch[0], batch[1], labels=batch[2])
				accs.append(utils.flat_accuracy(outputs.logits, batch[2]))
			pbar.set_description(f"Mean Accuracy so far = {np.mean(accs)}")
	print("Final Accuracy = ", np.mean(accs))

if __name__ == '__main__':
	run_validate_model()



