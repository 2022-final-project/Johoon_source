from unittest.util import strclass
from transformers import BertTokenizer, BertModel
import torch
from preProcessing import preProcessing

if __name__ == '__main__':
    # pre = preProcessing()

    # v = open('./vocab.txt', 'r')
    # ov = open('./one_line_vocab.txt', 'w')

    # while True:
    #     word = v.readline()
    #     if word == "":
    #         break
    #     ov.write(word[:len(word) - 1] + " ")

    rov = open('./one_line_vocab.txt', 'r')
    str = rov.readline()
    print("String is ", str)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print(" yeah dude")
    print(tokenizer.tokenize(str))
    model = BertModel.from_pretrained("bert-base-uncased")