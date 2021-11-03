from abc import ABC, abstractmethod
import editdistance
import transformers
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
	def __init__(self, classifier):
		pass

	def edit_distance(self, s1: str, s2: str) -> float:
		'''
		Return (1 - normalized edit distance) - rewards similar sentences
		''' 

		N = max(len(s1), len(s2))
		ed = editdistance.eval(s1, s2) / N
		r = 1 - ed
		return r

	

def main():
	reward_f = RewardWrapper()
	test1, test2 = "hi my name is sameer", "hi my name is bob"
	print(reward_f.edit_distance(test1, test2))
	test1, test2 = "hi my name is sameer", "hi my name is sam"
	print(reward_f.edit_distance(test1, test2))
	test1, test2 = "hi my name is sameer", "blah blah blah blah"
	print(reward_f.edit_distance(test1, test2))


if __name__ == '__main__':
	main()