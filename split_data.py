import sys
import pandas as pd
from torch.utils.data import WeightedRandomSampler 
import numpy as np

TRAIN_SIZE = 64000
EVAL_SIZE = 1000

def get_train_eval(csv_path):
	df = pd.read_csv(csv_path, sep="\t")
	df = df.astype({"label": "float32"})
	print(df.dtypes)
	distribution = np.array([1.0] * len(df))
	distribution[df.index[df["label"] == 1.0].tolist()] = 6.25
	print(distribution[:100])
	sampler = list(WeightedRandomSampler(distribution,len(df), replacement=False))
	train_sampler = sampler[:TRAIN_SIZE]
	eval_sampler = sampler[TRAIN_SIZE:TRAIN_SIZE+EVAL_SIZE]
	test_sampler = sampler[TRAIN_SIZE+EVAL_SIZE:]
	train_raw = df.loc[train_sampler, ["first", "second", "label"]]
	eval_raw = df.loc[eval_sampler, ["first","second","label"]]
	test_raw = df.loc[test_sampler,["first","second", "label"]]
	print(train_raw)
	return train_raw, eval_raw, test_raw

if __name__ == "__main__":
	csv_path = sys.argv[1]
	train_raw, eval_raw, test_raw = get_train_eval(csv_path)
	train_raw.to_csv("prof_qa/data_all/train/train.csv", index=False)
	eval_raw.to_csv("prof_qa/data_all/eval/eval.csv", index=False)
	test_raw.to_csv("prof_qa/data_all/test/test.csv", index=False)



