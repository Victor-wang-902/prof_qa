import sys
import pandas as pd

def generate_sentence_pairs(csv_path):
	df = pd.read_csv(csv_path, sep = "\t")
	#print(df)
	new_df = pd.DataFrame(data=None, columns=["first","second","label"])
	for i in range(370):
		for j in range(370):
			if df.loc[i,"type"] == df.loc[j,"type"]:
				new_df = new_df.append(pd.Series([df.loc[i,"question"],df.loc[j,"question"],1], index=new_df.columns),ignore_index=True)
			else:
				new_df = new_df.append(pd.Series([df.loc[i,"question"],df.loc[j,"question"],0], index=new_df.columns), ignore_index=True)
			print(new_df)
	new_df.to_csv("prof_qa/data_all/sentence_pairs.csv", sep="\t")


if __name__ == "__main__":
	csv_path = sys.argv[1]	
	generate_sentence_pairs(csv_path)
