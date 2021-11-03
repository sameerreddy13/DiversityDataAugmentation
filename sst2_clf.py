from transformers import BertTokenizerFast, BertForSequenceClassification
from utils import dev
from data import SSTLoader

class DistillBertSST():
	'''
	Get finetuned checkpoint of distillbert on sst2
	'''
	model_key = 'distilbert-base-uncased-finetuned-sst-2-english'
	def __init__(self):
		self.model = BertForSequenceClassification.from_pretrained(self.model_key)
		self.tokenizer = BertTokenizerFast.from_pretrained(self.model_key)


def main():
	model = DistillBertSST()
	

	# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
	# >>> labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
	# >>> outputs = model(**inputs, labels=labels)
	# >>> loss = outputs.loss
	# >>> logits = outputs.logits

if __name__ == '__main__':
	main()
