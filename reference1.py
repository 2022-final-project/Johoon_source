# code by Tae Hwan Jung(Jeff Jung) @graykode
# Reference : https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/LSTM_Attn.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

class BiLSTM_Attention(nn.Module):
    def __init__(self):
        super(BiLSTM_Attention, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)                # vector 로 변환시켜주는 1 step
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)
        self.out = nn.Linear(n_hidden * 2, num_classes)

    # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, n_hidden * 2, 1)   # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2) # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy() # context : [batch_size, n_hidden * num_directions(=2)]

    def forward(self, X):
        input = self.embedding(X) # input : [batch_size, len_seq, embedding_dim]
        input = input.permute(1, 0, 2) # input : [len_seq, batch_size, embedding_dim]

        hidden_state = torch.zeros(1*2, len(X), n_hidden) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.zeros(1*2, len(X), n_hidden) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2) # output : [batch_size, len_seq, n_hidden]
        attn_output, attention = self.attention_net(output, final_hidden_state)
        return self.out(attn_output), attention # model : [batch_size, num_classes], attention : [batch_size, n_step]


'''
    1. AI 에서 Data Preprocessing 을 할 때 train_data 로 하는게 맞죠?
'''

class preProcessing():
    def __init__(self):
        self.sql_words = ["select", "from", "where", "join", "left", "right", "outer", "group", "order", "by", "limit",
                            "sum", "avg", "min", "max", "count", "in", "exists", "like", "as", "and", "or", "between", 
                            "*", ">", ">=", "<", "=<", "<=", "=>", "==", "/", "-", "+", "=",
                            "date", "asc", "desc"]
        self.before_then_ignore = ["as", "limit"]
        self.word_count = {}
        self.table_list = {}
        self.col_list = {}
        self.vocab = {}

        self.make_query_one_sentence()
        self.process_by_one_query()
        # self.table_preProcessing()
        # self.whitespace()
        # self.modify1()

    # query 들을 통해 vocab.txt 생성을 위한 정보들을 따오는 함수
    def table_preProcessing(self):
        q = open('./queries.txt', 'r')

        select_flag = False
        from_flag = False

        table_cnt = 0
        col_cnt = 0

        from_end_list = ["select", "where", "left", "right", "(", "group", "order"]

        while True:
            cur_str = q.readline()
            
            if (cur_str == ""):
                break
            
            str_list = cur_str.split()
            str_list_np = np.array(str_list)
            cur_size = str_list_np.size

            for val in str_list:
                val.strip()

                if val[-1] == ",":
                    val = val[0:len(val) - 1]

                if val.lower() == "from":     # from 이 나온 경우
                    from_flag = True            # from 이 나왔다는 변수를 True 로
                    select_flag = False         # select 가 나왔다는 변수는 False
                elif val.lower() in from_end_list:  # 현재 value 가 table 이 아닌 경우 
                    from_flag = False               # from 나왔다는 변수를 다시 False 로
                elif from_flag:                     # 현재 table 이 나오고 있는 경우
                    if cur_size == 1:
                        if val not in self.table_vocab:     # 아직 dictionary 에 없는 것 일 경우
                            table_cnt += 1                  # dictionary 에 추가한다.
                            self.table_vocab[val] = "t" + str(table_cnt)
                        else:                               # 이미 있는 경우에는 pass 한다.
                            continue
                    elif cur_size == 2:                     # alias 를 준 경우에는
                        if val == str_list[1]:              # alias 는 table_vocab 에 들어가지 않도록 조심한다.
                            continue

        for key in self.table_vocab:
            print(" ", key, " : ", self.table_vocab[key])

    def whitespace(self):
        q = open('./queries.txt', 'r')
        
        while True:
            cur_str = q.readline()

            if cur_str == "":       # 더 이상 단어가 없는 경우 반복문을 종료한다.
                break
        
            str_list = cur_str.split()

            for val in str_list:
                if val[-1] == ",":
                    val = val[0:len(val) - 1]

                val.strip()

                if val not in self.word_count:
                    self.word_count[val] = 0
                else:
                    self.word_count[val] += 1

        self.word_count = sorted(self.word_count.items(), reverse = True, key = lambda item: item[1])

        wc = open('./words_count.txt', 'w')

        for key, value in self.word_count:
            input = key + " : " + str(value) + '\n'
            wc.write(input)

    def modify1(self):
        q = open('./queries.txt', 'r')
        w = open('./modified_query1.txt', 'w')

        while True:
            cur_str = q.readline()

            if cur_str == "":       # 더 이상 단어가 없는 경우 반복문을 종료한다.
                break
        
            str_list = cur_str.split()
            bef_val = ""
            for val in str_list:
                if val[-1] == ",":
                    val = val[0:len(val) - 1]

                val = val.strip()
                val = val.lower()
                val = val.strip('(')
                val = val.strip(')')

                if bef_val in self.before_then_ignore:
                    continue
                elif val in self.sql_words:
                    continue

                bef_val = val

                if val[0:3] in ["sum", "avg", "min", "max"]:
                    continue
                elif val[0:5] == "count":
                    continue
                elif val != "" and (val[0] == "'" or val[-1] == "'"):
                    continue
                elif val != "" and 48 <= ord(val[0]) and ord(val[0]) <= 57:
                    continue
                elif val == "\n" or val == "":
                    continue

                print(val)
                
                w.write(val + '\n')

    def make_query_one_sentence(self):
        q = open('./queries.txt', 'r')
        w = open('./one_query_one_sentence.txt', 'w')

        while True:
            cur_str = q.readline()

            if cur_str == "":       # 더 이상 단어가 없는 경우 반복문을 종료한다.
                break
        
            str_list = cur_str.split()

            for val in str_list:
                val = val.strip()
                val = val.lower()
                
                if val[-1] != ";":
                    w.write(val + " ")
                else:
                    w.write(val + '\n')

    def process_by_one_query(self):
        q = open('./one_query_one_sentence.txt', 'r')

        while True:
            cur_str = q.readline()
            print(cur_str)

            if cur_str == "":
                break
'''
    [modify1 method 사용 원칙]
    1. talbe, column name 은 안에 하지말기
'''                
                        
if __name__ == '__main__':
    embedding_dim = 2 # embedding size
    n_hidden = 5  # number of hidden units in one cell
    num_classes = 2  # 0 or 1

    pre = preProcessing()   # 1. Set table name like [t1, t2, t3 ...]
                            # 2. Set collum name like [c1, c2, c3 ...]

    # 3 words sentences (=sequence_length is 3)
    sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
    labels = [1, 1, 1, 1, 0, 0]  # 1 is good, 0 is not good.

    # sentences = ["SELECT * FROM T1", "SELECT C1 FROM T1, T2", "SELECT C1, C2 FROM T1", "SELECT C1, C2, C3 FROM T1, T2, T3", "SELECT C1 FROM T1 WHERE C2 = S1", "SELECT C2 FROM T1"]
    # labels = [1, 0, 1, 1, 0, 1]  # 1 is good, 0 is not good.

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    # print(word_list)

    word_dict = {w: i for i, w in enumerate(word_list)}
    
    # print(word_dict)
    vocab_size = len(word_dict)

    model = BiLSTM_Attention()
    
    # print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    inputs = torch.LongTensor([np.asarray([word_dict[n] for n in sen.split()]) for sen in sentences])
    targets = torch.LongTensor([out for out in labels])  # To using Torch Softmax Loss function

    # Training
    for epoch in range(5000):
        optimizer.zero_grad()
        output, attention = model(inputs)
        loss = criterion(output, targets)

        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    # Test
    test_text = 'i hate baseball'
    tests = [np.asarray([word_dict[n] for n in test_text.split()])]
    test_batch = torch.LongTensor(tests)

    # Predict
    predict, _ = model(test_batch)
    predict = predict.data.max(1, keepdim=True)[1]
    if predict[0][0] == 0:
        print(test_text,"is Bad Mean...")
    else:
        print(test_text,"is Good Mean!!")

    fig = plt.figure(figsize=(6, 3)) # [batch_size, n_step]
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    ax.set_xticklabels(['']+['first_word', 'second_word', 'third_word'], fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels(['']+['batch_1', 'batch_2', 'batch_3', 'batch_4', 'batch_5', 'batch_6'], fontdict={'fontsize': 14})
    # plt.show()

    '''
    q = open('./queries.txt', 'r')
        mq = open('./modified_queries.txt', 'w')

        input_str = ""

        while True:
            str = q.readline()

            if str == "":
                break
            str = str.strip()

            if str[-1] == ";" or str[-2] == ";":
                colon_idx = str.find(";")
                str = str[0:colon_idx]

                input_str = input_str + " " + str + '\n'

                mq.write(input_str)
                input_str = ""
            elif str[-1] == '\n':
                str = str[:len(str) - 1]

                input_str = input_str + " " + str
            else:
                input_str = input_str + " " + str
            print(input_str)

        mq.close()
    '''

    '''
    [Query 수정]
        extract ... 이런거 없애고
        view 없애고

    '''