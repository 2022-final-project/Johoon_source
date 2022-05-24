from lib2to3.pgen2 import token
from transformers import BertTokenizer
from transformers import BertModel
from transformers import BertConfig
from transformers import BertForMultipleChoice
from transformers import BertForPreTraining
from bert_ import BERTEmbedding
import linecache
import torch

configuration = BertConfig(500)
tokenizer = BertTokenizer.from_pretrained("./vocab.txt")

print("[tokenizer]")
print(tokenizer)

model1 = BertModel(configuration)
model2 = BertForMultipleChoice(configuration)
model3 = BertForPreTraining(configuration)
# model4 = position.PositionalEmbedding()

configuration1 = model1.config
configuration2 = model2.config

# model1.embeddings.position_embeddings()
# model1.embeddings = BERTEmbedding(vocab_size=len(tokenizer), embed_size=250)

# prompt = "select c1, c2 from t1, t2 where c2"
# choice0 = "1.0"
# choice1 = "2.0"
# choice2 = "4.0"
# choice3 = "8.0"
# choice4 = "16.0"
# choice5 = "32.0"
# labels = torch.tensor(0).unsqueeze(0)  

# encoding = tokenizer([prompt, prompt, prompt, prompt, prompt, prompt], [choice0, choice1, choice2, choice3, choice4, choice5], return_tensors="pt", padding=True, truncation=True)
# outputs2 = model2(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)

inputs1 = tokenizer("select c1 from t1", return_tensors="pt")
outputs1 = model1(**inputs1)
outputs3 = model3(**inputs1)

print("[inputs 1]\n", inputs1)
print("[outputs 1]\n", outputs1)
print("[outputs 3]\n", outputs3) 

# loss = outputs2.loss
# logits = outputs2.logits

# prediction_logits = outputs3.prediction_logits
# seq_relationship_logits = outputs3.seq_relationship_logits

# print("[prediction_logits]")
# print(prediction_logits)

# print("[seq_relationship_logits]")
# print(seq_relationship_logits)

# print("loss : ", loss)
# print("logits : ", logits)

'''

output1 : normal 하게 하는거
output2 : multipleChoice 로 하는거

'''