# import torch
# from transformers import BertTokenizer, BertLMHeadModel

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertLMHeadModel.from_pretrained("bert-base-uncased")

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

# print("[input information]")
# print(inputs)

# v = open('./vocab.txt', 'r')

# inputs = {}

# for i, v in enumerate(v):
#     inputs[v[:len(v) - 1]] = i

# print(inputs)

# outputs = model(**inputs, labels=inputs["input_ids"])

# loss = outputs.loss
# logits = outputs.logits

# print("loss : ", loss)

import argparse
from tokenizers import BertWordPieceTokenizer
from tokenizers import 

parser = argparse.ArgumentParser()

parser.add_argument("--corpus_file", type=str)
parser.add_argument("--vocab_size", type=int, default=32000)
parser.add_argument("--limit_alphabet", type=int, default=6000)

args = parser.parse_args()

tokenizer = BertWordPieceTokenizer(
    vocab_file='./vocab.txt',
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=False, # Must be False if cased model
    lowercase=False,
    wordpieces_prefix="##"
)

tokenizer.train(
    files=[args.corpus_file],
    limit_alphabet=args.limit_alphabet,
    vocab_size=args.vocab_size
)

tokenizer.save("./", "ch-{}-wpm-{}".format(args.limit_alphabet, args.vocab_size))
