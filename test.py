from fine_tune import SentencePairsDataset
import pandas as pd
#prof_list = [["John", "Doe"], ["Victor", "Wick"], ["Donald", "Trump"]]
#template = {"type 1": ["template 1 and type 1 and {person}", "template 2 and type 1 and {person}", "template 3 and type 1 and {person}"], "type 2": ["template 1 and type 2 and {person}", "template 2 and type 2 and {person}", "template 3 and type 2 and {person}"], "type 3": ["template 1 and type 3 and {person}", "template 2 and type 3 and {person}", "template 3 and type 3 and {person}"]}
#dataset_without = SentencePairsDataset(prof_list, template, 3, 2, False, False)
#dataset_with = SentencePairsDataset(prof_list,template,3,2,False, True)
#s = 0
#for data in iter(dataset_with):
#	print(data)
#	print(s)
#	s += 1
df = pd.read_csv("prof_qa/data_all/train_offline/train_intent.csv")
print(len(df))
