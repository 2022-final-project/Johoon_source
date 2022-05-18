from transformers import BertTokenizer
from transformers import BertModel
from transformers import BertConfig
from transformers import BertForMultipleChoice
import torch

configuration = BertConfig(500)
tokenizer = BertTokenizer.from_pretrained("./vocab.txt")
model1 = BertModel(configuration)
model2 =  BertForMultipleChoice(configuration)

configuration1 = model1.config
configuration2 = model2.config

prompt = "select c1, c2 from t1, t2 where c2"
choice0 = "1.0"
choice1 = "2.0"
choice2 = "4.0"
choice3 = "8.0"
choice4 = "16.0"
choice5 = "32.0"
labels = torch.tensor(0).unsqueeze(0)  

encoding = tokenizer([prompt, prompt, prompt, prompt, prompt, prompt], [choice0, choice1, choice2, choice3, choice4, choice5], return_tensors="pt", padding=True, truncation=True)
outputs2 = model2(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)

inputs1 = tokenizer("select c1 from t1", return_tensors="pt")
outputs1 = model1(**inputs1)

print("[input]\n", inputs1)
print("[outputs]\n", outputs1) 

loss = outputs2.loss
logits = outputs2.logits

print("loss : ", loss)
print("logits : ", logits)