import numpy as np

class preProcessing():
    def __init__(self):
        self.sql_words = ["select", "from", "where", "join", "left", "right", "outer", "group", "order", "by", "limit",
                            "when", "then", "having",
                            "sum", "avg", "min", "max", "count", "in", "exists", "like", "as", "and", "or", "between", "not"
                            "*", ">", ">=", "<", "=<", "<=", "=>", "==", "/", "-", "+", "=",
                            "date", "month", "year", "asc", "desc", "<>"]
        self.before_then_ignore = ["as", "limit"]
        self.word_count = {}
        self.table_list = {}
        self.col_list = {}
        self.vocab = {}

        self.table_preProcessing()
        self.make_query_one_sentence()      # ";" 를 기준으로 한 행에 한 Query 가 들어가게 한다.
        self.process_by_one_query()         
        # self.make_query_one_sentence2()
        # self.whitespace()
        # self.modify1()

    def word_refine(self, cur_word):
        cur_word = cur_word.split()
        cur_word = cur_word[0]
        if cur_word[-1] == ",":
            cur_word = cur_word[0:len(cur_word) - 1]
        cur_word = cur_word.split(")")
        cur_word = cur_word[0]
        cur_word = cur_word.split("(")
        cur_word = cur_word[0]
        cur_word = cur_word.split("'")
        cur_word = cur_word[0]
        return cur_word

    def process_by_one_query(self):
        q = open('./one_query_one_sentence.txt', 'r')
        w = open('./modified_one_query_one_sentence.txt', 'w')
        wv1 = open('./temporary_vocab.txt', "w")

        while True:
            cur_query = q.readline()
            # print(cur_query)

            if cur_query == "":
                break

            word_list = cur_query.split()


            end_flag = False
            alias_list = []
            word_list_np = np.array(word_list)
            alias_idx = np.where(word_list_np == "as")[0]
            
            for idx in alias_idx:
                if word_list[idx + 1][-1] == ",":
                    word_list[idx + 1] = word_list[idx + 1][0:len(word_list[idx + 1]) - 1]
                alias_list.append(word_list[idx + 1])

            for word in word_list:
                word = self.word_refine(word)

                if word == "":
                    continue
                if word in self.sql_words or (48 <= ord(word[0]) and ord(word[0]) <= 57) or word in alias_list:
                    if word[-1] == ";":
                        end_flag = True
                    else:
                        continue
                
                if word in self.table_list or word in alias_list:
                    continue

                for alias in alias_list:
                    if len(alias) < len(word):
                        if word[0:len(alias)] == alias and word[len(alias)] == ".":
                            word = word[len(alias) + 1:]

                if end_flag == False:
                    w.write(word + " ")
                else:
                    end_flag = False
                    wlen = len(word)
                    word = word[0:wlen - 1]
                    word = self.word_refine(word)
                    print(" refined word : ", word)
                    if word in self.sql_words or (48 <= ord(word[0]) and ord(word[0]) <= 57) or word in alias_list:
                        word = ""
                    elif word in self.table_list or word in alias_list:
                        word = ""
                    w.write(word + '\n')

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
                val = val.strip()

                if val[-1] == ",":
                    val = val[0:len(val) - 1]

                if val.lower() == "from":     # from 이 나온 경우
                    from_flag = True            # from 이 나왔다는 변수를 True 로
                    select_flag = False         # select 가 나왔다는 변수는 False
                elif val.lower() in from_end_list:  # 현재 value 가 table 이 아닌 경우 
                    from_flag = False               # from 나왔다는 변수를 다시 False 로
                elif from_flag:                     # 현재 table 이 나오고 있는 경우
                    if cur_size == 1:
                        if val not in self.table_list:     # 아직 dictionary 에 없는 것 일 경우
                            table_cnt += 1                  # dictionary 에 추가한다.
                            self.table_list[val] = "t" + str(table_cnt)
                        else:                               # 이미 있는 경우에는 pass 한다.
                            continue
                    elif cur_size == 2:                     # alias 를 준 경우에는
                        if val == str_list[1]:              # alias 는 table_list 에 들어가지 않도록 조심한다.
                            continue

        # for key in self.table_list:
        #     print(" ", key, " : ", self.table_list[key])

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

                # print(val)
                
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

    def make_query_one_sentence2(self):
        q = open('./modified_one_query_one_sentence.txt', 'r')
        w = open('./sss.txt', 'w')

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