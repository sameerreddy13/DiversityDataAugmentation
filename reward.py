from abc import ABC, abstractmethod
from typing import Dict
import editdistance
import transformers
from nltk import ngrams
from sentence_transformers import SentenceTransformer
import clf_sst2
import torch
from utils import DEV
# class SentencePairReward(ABC):
# 	'''
# 	SentencePairReward is a base class for rewards that take in a pair of sentences and computes single or multiple reward between them e.g.
# 	diversity between input and generated sentences.
# 	'''
# 	def __init__(self, weight: float):
# 		'''
# 		Weight should be the weight for this reward
# 		'''
# 		pass

# 	@abstractmethod
# 	def reward(s1: str, s2: str) -> float:
# 		pass

# class EditDistanceReward(SentencePairReward):
# 	'''
# 	Edit distance reward
# 	'''
# 	def __init(self):


class RewardWrapper():
	'''
	RewardWrapper for reward functions and state.

	Assume for all inputs s1, s2 unless said otherwise:
		s1 (str): input sentence
		s2 (str): generated sentence
	'''
	embedder_key = 'all-distilroberta-v1'

	def __init__(self, clf = None):
		self.clf = clf_sst2.DistilBertSST()
		self.embedder = SentenceTransformer(self.embedder_key, device=DEV)


	def edit_distance(self, s1: str, s2: str) -> float:
		'''
		Return normalized edit distance in [0, 1].
		Note, this is Levenshtein distance.
		''' 

		N = len(s1) + len(s2)
		ed = editdistance.eval(s1, s2) / N
		r = 1 - ed
		return r

	def iou_ungrams(self, s1: str, s2: str, n = 1) -> float:
		'''
		Get ngram overlap between s1 and s2 as fraction in [0, 1]  
		IOU(ngrams(s1), ngrams(s2))
		'''
		def get_ngram_set(s, n=n):        
		    ngram_set= set()
		    for s in ngrams(s.split(),n=n):        
		        ngram_set.add(s)                
		    return ngram_set

		ng1, ng2 = get_ngram_set(s1), get_ngram_set(s2)
		print(len(ng1 & ng2), len(ng1 | ng2), len(ng1 & ng2) / len(ng1 | ng2), ng1, ng2)
		return len(ng1 & ng2) / len(ng1 | ng2)

	def embed_similarity(self, s1: str, s2: str):
		e1, e2 = self.embedder.encode([s1, s2], convert_to_tensor=True)
		return torch.nn.functional.cosine_similarity(e1, e2, dim=0).item()

	def clf_consistency(self, s1: str, s2: str, y: int) -> float:
		'''
		Consistency reward from classifier. 
		Inputs:
			y: label for s1
		'''
		yhat = self.clf.predict_on_text(s2)[0]
		return 1. if yhat == y else 0.

	def compute_rewards(self, s1: str, s2: str, y: int) -> Dict[str, float]:
		'''
		Inputs:
			y: label for s1
		'''
		return {
			"edit_distance": self.edit_distance(s1, s2),
			"iou_ungrams": self.iou_ungrams(s1, s2, n=1) + self.iou_ungrams(s1, s2, n=2),
			"embed_similarity": self.embed_similarity(s1, s2),
			"clf_consistency": self.clf_consistency(s1, s2, y)
		}
	

def main():
	reward_f = RewardWrapper()
	N = 2
	tests = [
		( "hi my name is sameer", "hi my name is bob", 0),
		("hi my name is sameer", "hi my name is sam", 0),
		("hi my name is sameer", "i hate bob hes a bad person", 0)
	]
	for t in tests:
		rewards = reward_f.compute_rewards(*t)
		print(rewards)

if __name__ == '__main__':
	main()