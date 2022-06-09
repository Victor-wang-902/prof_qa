import logging
from sentence_transformers.evaluation import SentenceEvaluator
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances
import numpy as np
from sentence_transformers.readers import InputExample

logger = logging.getLogger(__name__)


class TopNClassificationEvaluator(SentenceEvaluator):
	def __init__(self, sentence1=[], sentence2=[], labels=[], name="", batch_size=32, write_csv=True, top=[10]):
		self.sentences1 = sentence1
		self.sentences2 = sentence2
		self.labels = labels
		self.top = sorted(top,reverse=True)
		assert len(self.sentences1) == len(self.labels)
		
		self.write_csv = write_csv
		self.name = name
		self.batch_size = batch_size
		self.csv_file = "top_n_classification_evaluation" + ("_"+name if name else "") + "_results.csv"

		self.csv_headers = ["epoch", "steps"]
		for n in self.top:
			self.csv_headers.append("top_" + str(n) + "_accuracy")

	def __call__(self, model, output_path=None, epoch=-1, steps=-1):
		if epoch != -1:
			if steps == -1:
				out_txt = f" after epoch {epoch}:"
			else:
				out_txt = f" epoch {epoch} after {steps} steps:"
		else:
			out_txt = ":"
		
		logger.info("Top N Accuracy Evaluation of the model on " + self.name + " dataset" + out_txt)

		scores = self.compute_metrices(model)
		print("Final score for top 10, 5, 3, 1:",scores)
		main_score = scores[-1]
		
		file_output_data = [epoch, steps]
		
		for score in scores:
			file_output_data.append(score)

		if output_path is not None and self.write_csv:
			csv_path = os.path.join(output_path, self.csv_file)
			if not os.path.isfile(csv_path):
				with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
					writer = csv.writer(f)
					writer.writerow(self.csv_headers)
					writer.writerow(file_output_data)
			else:
				with open(csv_path, newline="",mode="a", encoding="utf-8") as f:
					writer = csv.writer(f)
					writer.writerow(file_output_data)

		return main_score

	def compute_metrices(self, model):
		sentences = list(set(self.sentences1 + self.sentences2))
		embeddings = model.encode(sentences, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True)
		emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
		embeddings1 = [emb_dict[sent] for sent in self.sentences1]
		embeddings2 = [emb_dict[sent] for sent in self.sentences2]
		correct = [0 for i in self.top]
		total = len(embeddings1)
		for i, embedding in enumerate(embeddings1):
			temp = [embedding for j in range(len(embeddings2))]
			temp_label = self.labels[i]
			cosine_scores = 1 - paired_cosine_distances(temp, embeddings2)
			#print("sentence1: ", self.sentences1[i])
			#print("scores: ", cosine_scores)
			target = self.labels[i]
			#print(target)
			#print("target: ", self.sentences2[target])
			indices = np.argsort(cosine_scores)
			indices = indices[::-1]
			#print("indices: ", indices)
			for j,k in enumerate(self.top):
				if target in indices[:k]:
					correct[j] += 1
				else:
					print("Mistake!!!\nsentences1:",self.sentences1[i],"\nsentences2:",self.sentences2[target],"\nscore:",cosine_scores,"\ntarget index:",target,"\nindices ranking:",indices)
					break
		correct = np.asarray(correct)
		return correct / total

				

