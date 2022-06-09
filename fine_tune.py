import os
import sys
import sentence_transformers
from top_n_evaluator import TopNClassificationEvaluator
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler, IterableDataset
import pandas as pd
import csv
import math
import random
from copy import deepcopy
from collections import OrderedDict

BATCH_SIZE = 256

def get_train_eval(train_csv_path, eval_csv_path):
	train_raw = pd.read_csv(train_csv_path)
	train_raw = train_raw.astype({"label": "float32"})
	eval_raw = pd.read_csv(eval_csv_path)
	eval_raw = eval_raw.astype({"label":"float32"})
	train_data = [InputExample(texts=[train_raw.loc[i,"first"], train_raw.loc[i,"second"]], label=train_raw.loc[i,"label"]) for i in list(train_raw.index.values)]
	train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
	temp = list(eval_raw.index.values)
	eval_data = [[eval_raw.loc[i,"first"] for i in temp], [eval_raw.loc[i,"second"] for i in temp], [eval_raw.loc[i,"label"] for i in temp]]
	print(len(train_data))
	#print(eval_data[0][:4], eval_data[1][:4], eval_data[2][:4])
	#eval_dataloader = DataLoader(eval_data, shuffle=True, batch_size=16)
	return train_dataloader, eval_data

class ShuffleDataset(IterableDataset):
	def __init__(self, dataset, buffer_size = 1000000):
		super().__init__()
		self.dataset = dataset
		self.buffer_size = buffer_size

	def __len__(self):
		return len(self.dataset)

	def __iter__(self):
		buf = []
		try:
			data_iter = iter(self.dataset)
			for i in range(self.buffer_size):
				buf.append(next(data_iter))
		except:
			self.buffer_size = len(buf)

		try:
			while True:
				try:
					item = next(data_iter)
					idx = random.randint(0, self.buffer_size - 1)
					yield buf[idx]
					buf[idx] = item
				except StopIteration:
					break
			random.shuffle(buf)
			while len(buf) > 0:
				yield buf.pop()
		except GeneratorExit:
			pass


class SamplerDataset(IterableDataset):
	def __init__(self, dataset, size=None):
		super().__init__()
		self.dataset = dataset
		self.data_len = len(self.dataset)
		if size is None:
			self.size = len(self.dataset)
		else:
			self.size = size
		self.distribution = 1/self.data_len * self.size
		print("distribution:", self.distribution)

	def __len__(self):
		return self.size

	def __iter__(self):
		dataset_iter = iter(self.dataset)
		try:
			i = 0
			while True:
				try:
					i+=1
					item = next(dataset_iter)
					odd = random.uniform(0.0, 1.0)
					if odd <= self.distribution:
						#if type(item) is not type(tuple()):
							#print("odd:", odd)
							#print(item.texts[0],"###", item.texts[1], "###", item.label)
						yield item
					else:
						continue
					#yield item
				except StopIteration:
					print("stop ieration")
					break
			print("total sampled: ", i)
		except GeneratorExit:
			print("what happened?")
			pass

class SingleSentenceGenerator(IterableDataset):
	def __init__(self, prof_list, templates):
		super().__init__()
		self.prof_list = prof_list
		self.num_prof = len(prof_list)
		self.templates = deepcopy(templates)
		self.num_templates = len(self.templates)

	def __len__(self):
		return self.num_prof * self.num_templates
	
	def __iter__(self):
		for prof in self.prof_list:
			for template in self.templates:
				yield template.format(person=" ".join(prof))


class SentencePairsDataset(IterableDataset):
	def __init__(self, prof_list, templates, num_template_per_category, neg_pos_ratio, train=True, intent_data=False, typo_threshold=None, ed_threshold=None):
		self.typo_dict = {"a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr", "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn", "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp", "p": "ol", "q": "aw", "r": "edft", "s": "awedxz", "t": "rfgy", "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu", "z": "asx"}	
		if ed_threshold is None:
			self.ed_threshold = OrderedDict([(1, 0.06), (2, 0.09), (3, 0.1), (0, 1.0)])
		else:
			self.ed_threshold = ed_threshold
		if typo_threshold is None:
			self.typo_threshold = OrderedDict([("swap", 0.1), ("replace", 0.3), ("add", 0.6), ("delete", 1.0)])
		else:
			self.typo_threshold = typo_threshold
		super().__init__()
		self.prof_list = prof_list
		self.num_prof = len(prof_list)
		self.ratio = neg_pos_ratio
		self.reverse_template_dict = dict()
		self.template_stream = []
		self.templates = deepcopy(templates)
		self.mode = train
		self._reverse_index(templates)
		self._create_template_stream()
		self.num_template = len(self.template_stream)
		self.num_template_per_category = num_template_per_category
		self.prof_table = dict()
		self.prof_distribution = dict()
		self._create_prof_names()
		self.intent = intent_data

	def __len__(self):
		return int(self.num_prof * self.num_template * (self.num_prof * self.num_template - self.num_template_per_category) * ((self.ratio + 1) / self.ratio))

	def _create_prof_names(self):
		for prof in self.prof_list:
			name = prof[1:]
			title = prof[0]
			prof_names = []
			distribution = []
			table = [[None for j in range(len(name))] for i in range(len(name))]
			for i in range(len(name)):
				self._name_combo(table, name, 0, i + 1)
			for item in table[0]:
				prof_names.extend(item)
				distribution.extend([1.0 for k in range(len(item))]) # for custom distribution of all names
				for j in range(len(item)):
					prof_names.append(" ".join([title, item[j]]))
					distribution.append(1.0)
			self.prof_table[" ".join(prof)] = prof_names
			self.prof_distribution[" ".join(prof)] = distribution

	def _reverse_index(self, templates):
		for key, value in templates.items():
			for sentence in value:
				self.reverse_template_dict[sentence] = key
	
	def _create_template_stream(self):
		for value in self.reverse_template_dict.keys():
			self.template_stream.append(value)
	
	def _name_combo(self, table, name, idx, r):
		length = len(name) - idx
		if idx > len(table) - 1:
			return
		if table[idx][r-1] is not None:
			return
		if r == 1 or length < r:
			table[idx][r-1] = name[idx:]
			return
		combo = []
		for i in range(idx, len(name) - r + 1):
			self._name_combo(table, name, i + 1, r - 1)
			for j in range(len(table[i+1][r-2])):
				combo.append(" ".join([name[i], table[i+1][r-2][j]]))
		table[idx][r-1] = combo
	
	def _noise_induce(self, item):
		new = []
		for word in item.split():
			if word.isdigit():
				continue
			word = word.lower()
			chance = random.uniform(0.0, 1.0)
			edit_distance = 0
			for ed, threshold in self.ed_threshold.items():
				if chance <= threshold:
					edit_distance = ed
					break
			while edit_distance > 0:
				operation = "delete"
				chance = random.uniform(0.0, 1.0)
				for method, threshold in self.typo_threshold.items():
					if chance <= threshold:
						operation = method
						break
				if operation == "delete":
					if len(word) <= 2:
						edit_distance -= 1
						continue
					roll = random.randint(0, len(word)-1)
					word = word[:roll] + word[roll+1:]
					edit_distance -= 1
				elif operation == "swap":
					if edit_distance <= 1:
						continue
					if len(word) <= 1:
						edit_distance -= 1
						continue
					roll = random.randint(0, len(word) - 2)
					word = word[:roll]+ word[roll+1] + word[roll] + word[roll+2:]
					edit_distance -= 2
				elif operation == "replace":
					roll = random.randint(0, len(word)-1)
					word = word[:roll] + random.choice(self.typo_dict[word[roll]]) + word[roll+1:]
					edit_distance -= 1
				elif operation == "add":
					roll = random.randint(0, len(word))
					if roll == 0:
						to_add = self.typo_dict[word[roll]]
					elif roll == len(word):
						to_add = self.typo_dict[word[roll-1]]
					else:
						to_add = self.typo_dict[word[roll-1]] + self.typo_dict[word[roll]]
					word = word[:roll] + random.choice(to_add) + word[roll:]
					edit_distance -= 1
			new.append(word)
		return " ".join(new)

				


			

	def _assemble_template(self, prof, template, noise=True):
		prof_names = self.prof_table[" ".join(prof)]
		choice = random.choice(prof_names)
		if noise:
			person = self._noise_induce(choice)
		else:
			person = choice
		return template.format(person=person)

	def __iter__(self):
		if self.intent:
			for i, template in enumerate(self.template_stream):
				category = self.reverse_template_dict[template]
				pos_templates = self.templates[category]
				for j, prof in enumerate(self.prof_list):
					neg_sum = 0
					pos_num = 0
					for k, target in enumerate(self.template_stream):
						for l, targ_prof in enumerate(self.prof_list):
							label = None
							if self.reverse_template_dict[target] == self.reverse_template_dict[template] and prof == targ_prof:
								continue
							elif self.reverse_template_dict[target] == self.reverse_template_dict[template]:
								continue
							elif prof == targ_prof:
								label = 0.0
							else:
								label = 0.0
							first = self._assemble_template(prof, template)
							second = target.format(person=" ".join(targ_prof))
							if self.mode:
								yield InputExample(texts=[first, second], label=label)
							else:
								yield (first, second, label)
							neg_sum += 1
					pos_num = neg_sum // self.ratio
					pos_sent_dataset = iter(SingleSentenceGenerator(self.prof_list, pos_templates))
					pos_sentences = []
					for pos in pos_sent_dataset:
						pos_sentences.append(pos)
					for i in range(pos_num):
						first = self._assemble_template(prof, template)
						second = random.choice(pos_sentences)
						if self.mode:
							yield InputExample(texts=[first, second], label=1.0)
						else:
							yield (first, second, 1.0)
		else:
			print(self.prof_list[27])
			for i, template in enumerate(self.template_stream):
				for j, prof in enumerate(self.prof_list):
					neg_sum = 0
					pos_num = 0
					for k, target in enumerate(self.template_stream):
						for l, targ_prof in enumerate(self.prof_list):
							label = None
							if self.reverse_template_dict[target] == self.reverse_template_dict[template] and prof == targ_prof:
								continue
							elif self.reverse_template_dict[target] == self.reverse_template_dict[template]:
								label = 0.0
							elif prof == targ_prof:
								label = 0.0
							else:
								label = 0.0
							first = self._assemble_template(prof, template)
							second = target.format(person=" ".join(targ_prof))
							if self.mode:
								yield InputExample(texts=[first, second], label=label)
							else:
								yield (first, second, label)
							neg_sum += 1
					print("neg_sum",neg_sum)
					pos_num = neg_sum // self.ratio
					temp_templates = self.templates[self.reverse_template_dict[template]]
					pos_sequence = RandomSampler(temp_templates, replacement=True, num_samples=pos_num)
					for item in pos_sequence:
						first = self._assemble_template(prof, template)
						if self.mode:
							yield InputExample(texts=[first, temp_templates[item].format(person=" ".join(prof))], label=1.0)
						else:
							yield (first, temp_templates[item].format(person=" ".join(prof)), 1.0)


class CustomEvalDataset(SentencePairsDataset):
	def __init__(self, prof, templates, ed_threshold=None, typo_threshold=None, noise=True):
		self.typo_dict = {"a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr", "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn", "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp", "p": "ol", "q": "aw", "r": "edft", "s": "awedxz", "t": "rfgy", "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu", "z": "asx"}	
		if ed_threshold is None:
			self.ed_threshold = OrderedDict([(1, 0.06), (2, 0.09), (3, 0.1), (0, 1.0)])
		else:
			self.ed_threshold = ed_threshold
		if typo_threshold is None:
			self.typo_threshold = OrderedDict([("swap", 0.1), ("replace", 0.3), ("add", 0.6), ("delete", 1.0)])
		else:
			self.typo_threshold = typo_threshold
		self.noise = noise
		self.prof_list = prof
		self.templates = deepcopy(templates)
		self.num_prof = len(self.prof_list)
		self.template_stream = []
		self.reverse_template_dict =dict()
		self._reverse_index(templates)
		self._create_template_stream()
		self.num_template = len(self.template_stream)
		self.prof_table = dict()
		self.prof_distribution = dict()
		self._create_prof_names()
		self.cat_dig_map = dict()
		for i, key in enumerate(self.templates):
			self.cat_dig_map[key] = i

	def __len__(self):
		raise NotImplementedError
	
	def __iter__(self):
		raise NotImplementedError
	
	def generate_eval_data(self):
		base_templates = []
		query_templates = []
		base_categories = []
		
		for key, value in self.templates.items():
			temp = value.pop()
			base_templates.append(temp)
			query_templates.append(temp)
			for item in value:
				query_templates.append(item)
		sentences2 = []
		base_categories = []
		for i, prof in enumerate(self.prof_list):
			for template in base_templates:
				base_categories.append((self.cat_dig_map[self.reverse_template_dict[template]], i))
				sentences2.append(template.format(person=" ".join(prof)))
		sentences1 = []
		labels = []
		for i, prof in enumerate(self.prof_list):
			for template in query_templates:
				for j, item in enumerate(base_categories):
				#	print(self.cat_dig_map[self.reverse_template_dict[template]],i,item)
					if (self.cat_dig_map[self.reverse_template_dict[template]], i) == item:
						labels.append(j)
						break	
				sentences1.append(self._assemble_template(prof, template, noise=self.noise))
		print(len(labels),len(sentences1))
		assert len(labels) == len(sentences1)
		return sentences1, sentences2, labels



def offline_train_eval_from_generator(eval_prof, eval_template_per_category, train_eval_ratio, train_neg_pos_ratio, val_neg_pos_ratio, train_path, eval_path, intent_data=False):
	print("reading files...")
	template = dict()
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
	with open("prof_qa/data_all/templates.txt", "r") as f:
		for line in f.readlines():
			line = line.strip()
			sentences = line.split("\t")
			least_templates = min(least_templates, len(sentences) - 1)
			num_templates += len(sentences) - 1
			column_name = sentences[0]
			template[column_name] = sentences[1:]
	assert least_templates > eval_template_per_category
	assert eval_prof < len(people)
	num_category = len(template)
	print("done")
	print("num_prof: ", num_prof)
	print("num_category: ", num_category)
	print("num_templates: ", num_templates)
	print("least_templates: ", least_templates)
	print("initializing datasets...")
	eval_prof_list = people[:eval_prof]
	train_prof_list = people[eval_prof:]
	eval_templates = dict()
	train_templates = dict()
	for key, value in template.items():
		eval_templates[key] = value[:eval_template_per_category]
		train_templates[key] = value[eval_template_per_category:]
	train_dataset = SentencePairsDataset(train_prof_list, train_templates, least_templates, train_neg_pos_ratio, False, intent_data=intent_data)
	eval_dataset = SentencePairsDataset(eval_prof_list, eval_templates, eval_template_per_category, val_neg_pos_ratio, False, intent_data=intent_data)
	print("done")
	print("writing data to files...")
	_batch_write(train_dataset, train_path)
	_batch_write(eval_dataset, eval_path)
	print("done")

def _batch_write(dataset, path, batch_size=1000):
	df = pd.DataFrame(columns=["first", "second", "label"])
	df = df.astype({"first": "string", "second": "string", "label": "float32"})
	df.to_csv(path, sep="\t")
	iterator = iter(dataset)
	end = False
	while True:
		df = pd.DataFrame(columns=["first", "second", "label"])
		df = df.astype({"first": "string", "second": "string", "label": "float32"})
		iteration = 0
		while iteration < batch_size:
			iteration += 1
			try:
				item = next(iterator)
				df =  df.append({"first": item[0], "second": item[1], "label": item[2]}, ignore_index=True)
			except StopIteration:
				end = True
				break
		print(df)
		df.to_csv(path, mode="a", index=False, header=False, sep="\t")
		if end:
			break

def get_train_eval_from_generator(cid, eval_prof, eval_template_per_category, train_eval_ratio, train_neg_pos_ratio, val_neg_pos_ratio, intent=False, custom_eval=False, size=50000):
	template = dict()
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
	with open("prof_qa/data_all/templates.txt", "r") as f:
		for line in f.readlines():
			line = line.strip()
			sentences = line.split("\t")
			least_templates = min(least_templates, len(sentences) - 1)
			num_templates += len(sentences) - 1
			column_name = sentences[0]
			template[column_name] = sentences[1:]
	assert least_templates > eval_template_per_category
	assert eval_prof < len(people)
	if custom_eval:
		assert intent == False
	num_category = len(template)
	#print("num_prof: ", num_prof)
	#print("num_category: ", num_category)
	#print("num_templates: ", num_templates)
	#print("least_templates: ", least_templates)
	eval_prof_list = people[:eval_prof]
	train_prof_list = people[eval_prof:]
	eval_templates = dict()
	train_templates = dict()
	for key, value in template.items():
		eval_templates[key] = value[:eval_template_per_category]
		train_templates[key] = value[eval_template_per_category:]
	train_dataset = ShuffleDataset(SamplerDataset(SentencePairsDataset(train_prof_list, train_templates, least_templates, train_neg_pos_ratio, train=False, intent_data=intent), size))
	print(len(train_dataset))	
	x = 0
	cwd = os.getcwd()
	train_cache_name = "cache_train_" + cid + "_" + str(x) + "_" + ".csv"
	directory = os.path.join(cwd, train_cache_name)
	while os.path.exists(directory):
		x += 1
		train_cache_name = "cache_train_" + cid + "_" + str(x) + "_" + ".csv"
		directory = os.path.join(cwd, train_cache_name)
		
	with open(directory, "w", encoding= "utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=["first","second", "label"], delimiter= "\t")
		writer.writeheader()
		for item in train_dataset:
			writer.writerow({"first": item[0], "second": item[1], "label": item[2]})

	train_raw = pd.read_csv(train_cache_name,sep="\t")
	print(train_raw)
	train_raw = train_raw.astype({"label": "float32"})
	train_data = [InputExample(texts=[train_raw.loc[i,"first"], train_raw.loc[i,"second"]], label=train_raw.loc[i,"label"]) for i in list(train_raw.index.values)]
	train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
	#print(eval_data[0][:4], eval_data[1][:4], eval_data[2][:4])
	#eval_dataloader = DataLoader(eval_data, shuffle=True, batch_size=16)
	x = 0
	cwd = os.getcwd()
	eval_cache_name = "cache_eval_" + cid + "_" + str(x) + "_" + ".csv"
	directory = os.path.join(cwd, eval_cache_name)
	while os.path.exists(directory):
		x += 1
		eval_cache_name = "cache_eval_" + cid + "_" + str(x) + "_" + ".csv"
		directory = os.path.join(cwd, eval_cache_name)

	if custom_eval:
		eval_dataset = CustomEvalDataset(eval_prof_list, eval_templates)
		sentences1, sentences2, labels = eval_dataset.generate_eval_data()
	else:
		eval_dataset = ShuffleDataset(SamplerDataset(SentencePairsDataset(eval_prof_list, eval_templates, eval_template_per_category, val_neg_pos_ratio, train=False, intent_data=intent), size))
		with open(directory, "w", encoding="utf-8") as f:
			writer = csv.DictWriter(f, fieldnames=["first","second", "label"], delimiter= "\t")
			writer.writeheader()
			for item in eval_dataset:
				writer.writerow({"first": item[0], "second": item[1], "label": item[2]})
		eval_raw = pd.read_csv(eval_cache_name, sep ="\t")
		eval_raw = eval_raw.astype({"label":"float32"})
		temp = list(eval_raw.index.values)
		eval_data = [[eval_raw.loc[i,"first"] for i in temp], [eval_raw.loc[i,"second"] for i in temp], [eval_raw.loc[i,"label"] for i in temp]]
	


	#x = 0
	#for item in train_dataset:
		#print(item.texts[0],"###", item.texts[1], "###", item.label)
		#x += 1
		#if x > 10000:
		#	break
	print("train size:", len(train_dataset))
	if custom_eval:
		return train_dataloader, (sentences1, sentences2, labels)
	else:
		eval_data = [[],[],[]]
		for item in eval_dataset:
			eval_data[0].append(item[0])
			eval_data[1].append(item[1])
			eval_data[2].append(item[2])
		return train_dataloader, eval_data


def get_model(model_path):
	model = SentenceTransformer(model_path, device="cuda", cache_folder="model_cache")
	train_loss = losses.CosineSimilarityLoss(model)
	return model, train_loss

def train_model(model, train_loss, train_dataloader, eval_data, model_storage, model_checkpoint, custom_evaluator=False):
	if custom_evaluator:
		evaluator = TopNClassificationEvaluator(eval_data[0], eval_data[1], eval_data[2], "custom", top=[10,5,3,1])
	else:
		evaluator = evaluation.BinaryClassificationEvaluator(eval_data[0], eval_data[1], eval_data[2], "default", show_progress_bar=True)
	model.fit(train_objectives=[(train_dataloader, train_loss)], evaluator=evaluator, epochs=100, warmup_steps=100, save_best_model=True, show_progress_bar=True, output_path=model_storage, checkpoint_path=model_checkpoint)

if __name__ == "__main__":
	dynamic_generation = int(sys.argv[1])
	if dynamic_generation == 1:
		model_path = sys.argv[2]
		model_storage = sys.argv[3]
		model_checkpoint = sys.argv[4]
		eval_prof = int(sys.argv[5])
		eval_template_per_category = int(sys.argv[6])
		train_eval_ratio = int(sys.argv[7])
		train_neg_pos_ratio = int(sys.argv[8])
		val_neg_pos_ratio = int(sys.argv[9])
		intent = bool(int(sys.argv[10]))
		custom_eval = bool(int(sys.argv[11]))
		size = int(sys.argv[12])
		train_dataloader, eval_data = get_train_eval_from_generator(str(size), eval_prof, eval_template_per_category, train_eval_ratio, train_neg_pos_ratio, val_neg_pos_ratio, intent, custom_eval, size)
		model, train_loss = get_model(model_path)
		train_model(model, train_loss, train_dataloader, eval_data, model_storage, model_checkpoint, custom_eval)
	elif dynamic_generation == 2:
		print("loading parameters")
		model_path = sys.argv[2]
		model_storage = sys.argv[3]
		model_checkpoint = sys.argv[4]
		train_csv_path = sys.argv[5]
		eval_csv_path = sys.argv[6]
		print("done")
		train_dataloader, eval_data = get_train_eval(train_csv_path, eval_csv_path)
		model, train_loss = get_model(model_path)
		train_model(model, train_loss, train_dataloader, eval_data, model_storage, model_checkpoint)
	elif dynamic_generation == 3:
		print("loading parameters")
		eval_prof = int(sys.argv[2])
		eval_template_per_category = int(sys.argv[3])
		train_eval_ratio = int(sys.argv[4])
		train_neg_pos_ratio = int(sys.argv[5])
		val_neg_pos_ratio = int(sys.argv[6])
		train_csv_path = sys.argv[7]
		eval_csv_path = sys.argv[8]
		intent = bool(int(sys.argv[9]))
		print("done")
		if not (os.path.exists(train_csv_path) and os.path.exists(eval_csv_path)):
			print("generating offline data")
			offline_train_eval_from_generator(eval_prof, eval_template_per_category, train_eval_ratio, train_neg_pos_ratio, val_neg_pos_ratio, train_csv_path, eval_csv_path, intent_data=intent)
		print("done")


