from sentence_transformers import SentenceTransformer, util

model1 = SentenceTransformer("model_v21")
model2 = SentenceTransformer("model_v20")
sentence_pairs = [("what is wilsonn's email address", "What is the email address of Professor Yik Cheung Wilson Tam?"),
					("where is  Alicee Chuang's offic?", "what office is Professor Alice Chuang in?"),
					("What school does oliiver graduate from","which school was Dean Olivier marin graduated from?"),
					("Who is Professor Seeger Zou?", "what is the bio of Professor Almaz Zelleke?"),
					("show me a picture of Professor Michael Walfish.", "What is Professor Arif Ullah's phone number?"),
					("Where does Alice Chuang graduate from?", "What school did Professor Alice Chuang go to?"),
					("What subject does John Wick specialize in?", "what is Professor John Wick's research interest?"),
					("Where is Professor John Wick's office?", "where in the academic building is Professor John Doe's office located at?")]
embeddings1 = [(model1.encode(sentence_pairs[i][0]), model1.encode(sentence_pairs[i][1])) for i in range(len(sentence_pairs))]
embeddings2 = [(model2.encode(sentence_pairs[i][0]), model2.encode(sentence_pairs[i][1])) for i in range(len(sentence_pairs))]
cos_sim1 = [util.cos_sim(embeddings1[i][0],embeddings1[i][1]) for i in range(len(embeddings1))]
cos_sim2 = [util.cos_sim(embeddings2[i][0],embeddings2[i][1]) for i in range(len(embeddings2))]
print("Cos-similarity: label should be 0,0,0,0,0,1,1,0\npretrained model gives", cos_sim1,"\nfine tuned model gives", cos_sim2)
