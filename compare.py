from sentence_transformers import SentenceTransformer
from fine_tune import get_train_eval_from_generator, CustomEvalDataset
from top_n_evaluator import TopNClassificationEvaluator
import pandas as pd

if __name__ == "__main__":
	df = pd.read_csv("prof_qa/professor_profile_all.txt", sep="\t")
	tit_idx = 0
	sub_idx = 1
	people = []
	num_prof = len(df)
	least_templates = float("inf")
	num_templates = 0
	for i in range(num_prof):
		tit = df.iloc[i, tit_idx]
		sub = df.iloc[i, sub_idx]
		person = [tit] + sub.split()
		people.append(person)
	template = dict()
	with open("prof_qa/data_all/templates.txt", "r") as f:
		for line in f.readlines():
			line = line.strip()
			sentences = line.split("\t")
			least_templates = min(least_templates, len(sentences) - 1)
			num_templates += len(sentences) - 1
			column_name = sentences[0]
			template[column_name] = sentences[1:]
	assert least_templates > 2
	assert 30 < len(people)
	num_category = len(template)
	eval_prof_list = people[:30]
	train_prof_list = people[30:]
	eval_templates = dict()
	train_templates = dict()
	for key, value in template.items():
		eval_templates[key] = value[:2]
		train_templates[key] = value[2:]
	eval_dataset = CustomEvalDataset(eval_prof_list, eval_templates, noise=True)
	sentences1, sentences2, labels = eval_dataset.generate_eval_data()

	evaluator = TopNClassificationEvaluator(sentences1, sentences2, labels,top = [10,5,3,1])
	things = ["model_v3", "model_v20", "model_v21", "model_v24", "model_v24", "model_v25", "model_v26", "model_v27", "model_v28", "model_v29", "model_v30", "model_v31", "model_v32"]
	for thingy in things:
		model1 = SentenceTransformer(thingy)
		print("#################################################################")
		model1.evaluate(evaluator, "eval_"+thingy + "_noise")

		print("#######################################################################")

