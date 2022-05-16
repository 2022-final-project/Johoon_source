import numpy as np
from pkg_resources import ensure_directory

class preProcessing():
    def __init__(self):
        self.sql_words = ["select", "from", "where", "join", "left", "right", "outer", "group", "order", "by", "limit",
                            "when", "then", "case", "having", "interval", 
                            "sum", "avg", "min", "max", "count", "in", "exists", "like", "as", "and", "or", "between", "not", 
                            "*", ">", ">=", "<", "=<", "<=", "=>", "==", "/", "-", "+", "=",
                            "date", "month", "year", "asc", "desc", "<>", "on", "end", "if", "else"]
        self.before_then_ignore = ["as", "limit"]
        self.operators = ["*", ">", ">=", "<", "=<", "<=", "=>", "==", "/", "-", "+", "=", "<>"]
        self.word_count = {}
        self.table_list = {}
        self.col_list = {}
        self.vocab = {}
        self.delete_state = False

        self.table_preProcessing()
        self.make_query_one_sentence()      # ";" 를 기준으로 한 행에 한 Query 가 들어가게 한다.
        self.process_by_one_query()
        self.column_preProcessing()       
        self.refine_words()
        self.make_vocab()
        # self.make_query_one_sentence2()
        # self.whitespace()
        # self.modify1()
    def make_vocab(self):
        q = open('./refined_query_to_make_vocab copy.txt', 'r')
        w = open('./vocab.txt', 'w')

        while True:
            words = q.readline()

            if words == "":
                break

            word_list = words.split()

            for word in word_list:
                if word not in self.vocab:
                    self.vocab[word] = 0
                else:
                    self.vocab[word] += 1

        self.vocab = sorted(self.vocab.items(), reverse = True, key = lambda item: item[1])

        print(" 보캅^^")
        print(self.vocab)

        for key, value in self.vocab:
            w.write(key + '\n')



    def refine_words(self):
        q = open('./one_query_one_sentence.txt', 'r')
        w = open('./refined_query_to_make_vocab.txt', 'w')

        while True:
            query = q.readline()

            if query == "":
                break
            
            word_list = query.split()
            # print(word_list)

            list_len = len(word_list)

            delete_flag = False
            alias_flag = False

            for i, word in enumerate(word_list):

                if len(word) == 0:
                    continue

                word = word.strip()


                if word == "as":
                    alias_flag = True
                    continue
                elif alias_flag:
                    alias_flag = False
                    continue

                if word[-1] == ",":
                    word = word[:len(word) - 1]

                while word != word.strip("(") or word != word.strip(")") or word != word.strip("*") or word != word.strip("+") or word != word.strip("-"):
                    word = word.strip("(")
                    word = word.strip(")")
                    word = word.strip("*")
                    word = word.strip("+")
                    word = word.strip("-")

                if word in self.table_list:
                    word = self.table_list[word]
                elif word in self.col_list:
                    word = self.col_list[word]

                if 0 < len(word) and (48 <= ord(word[0]) and ord(word[0]) <= 57):
                    word = ""

                if word in self.operators:
                    word = ""

                if word.startswith("'"):
                    if word.endswith("'"):
                        word = ""
                    else:
                        delete_flag = True
                elif word.endswith("'"):
                    word = ""
                    delete_flag = False

                if 4 < len(word) and word[0:4] in ["sum(", "avg(", "min(", "max("]:
                    word = word[:3] + " "

                    nxt_word = word[4:]

                    if nxt_word in self.table_list:
                        nxt_word = self.table_list[word]
                    elif nxt_word in self.col_list:
                        nxt_word = self.col_list[word]

                    word = word + nxt_word

                if 6 < len(word) and word[0:6] in ["count("]:
                    word = word[:5] + " "

                    nxt_word = word[6:]

                    if nxt_word in self.table_list:
                        nxt_word = self.table_list[word]
                    elif nxt_word in self.col_list:
                        nxt_word = self.col_list[word]

                    word = word + nxt_word

                if delete_flag:
                    word = ""

                if i < list_len - 1:
                    w.write(word + " ")
                else:
                    w.write(word + '\n')

    def column_preProcessing(self):
        c = open('./column_list.txt', 'r')
        w = open('./column_vocab.txt', 'w')
        
        col_num = 0;

        while True:
            query_cols = c.readline()

            if query_cols == "":       # 더 이상 단어가 없는 경우 반복문을 종료한다.
                break
        
            col_list = query_cols.split()

            for col in col_list:
                if col[-1] == "*":
                    col = col[0:len(col) - 1]

                if col not in self.col_list:
                    col_num += 1                  # dictionary 에 추가한다.
                    self.col_list[col] = "c" + str(col_num)

        print(" Column lists")
        for key, value in self.col_list.items():
            w.write(key + " : " + value + '\n')
        

    def word_refine(self, cur_word):
        cur_word = cur_word.strip()
        if cur_word[-1] == ",":
            cur_word = cur_word[0:len(cur_word) - 1]
        cur_word = cur_word.strip(")")
        cur_word = cur_word.strip("(")
        if (cur_word.find("'") != -1):
            if cur_word.count("'") == 2:
                return ""
            if self.delete_state == False:
                self.delete_state = True
            elif self.delete_state == True:
                self.delete_state = False
                return ""

        cur_word = cur_word.strip("'")
        if self.delete_state:
            cur_word = ""
        return cur_word

    def process_by_one_query(self):
        q = open('./one_query_one_sentence.txt', 'r')
        w = open('./column_list.txt', 'w')
        wv1 = open('./temporary_vocab.txt', "w")

        cntt = 0

        while True:
            cur_query = q.readline()
            # print(cur_query)

            if cur_query == "":
                break

            cntt += 1
            # print(" ocunt is ", cntt)
            word_list = cur_query.split()


            end_flag = False
            alias_list = []
            word_list_np = np.array(word_list)
            alias_idx = np.where(word_list_np == "as")[0]

            for idx in alias_idx:
                # print("alias : ", word_list[idx + 1])
                if word_list[idx + 1][-1] == ",":
                    word_list[idx + 1] = word_list[idx + 1][0:len(word_list[idx + 1]) - 1]
                # print("   -----> ", word_list[idx + 1])
                alias_list.append(word_list[idx + 1])

            for key, value in enumerate(alias_list):
                print(key, ":", value)
            print("word list : ", word_list)
    # 이후 쿼리의 마지막은 word == word_list[len(word_list) - 1] 로 수정
            for word in word_list:
                if word == word_list[len(word_list) - 1]:
                    w.write('\n')
                    break
                if 4 < len(word) and word[0:4] in ["sum(", "avg(", "min(", "max("]:
                    word = word[4:]
                    # ("w fwe afe ;", word)
                
                print("before : ", word)
                word = self.word_refine(word)
                print("after : ", word)
                if word == "":
                    continue
                if word in self.sql_words or (48 <= ord(word[0]) and ord(word[0]) <= 57) or word in alias_list:
                    if word[-1] == ";":
                        end_flag = True
                    else:
                        continue
                
                if word in self.table_list or word in alias_list:
                    if end_flag:
                        end_flag = False
                        w.write('\n')
                    continue

                for alias in alias_list:
                    print(" alias : [", alias, "] : word is ", word)
                    if len(alias) < len(word):
                        if word[0:len(alias)] == alias and word[len(alias)] == ".":
                            print(" word ", word, " 가 작업하러 들어옴")
                            word = word[len(alias) + 1:]
                            print("   after 작업 : ", word)
                    if end_flag:
                        end_flag = False
                        w.write('\n')

                if end_flag == False:
                    w.write(word + " ")
                else:
                    end_flag = False
                    wlen = len(word)
                    word = word[0:wlen - 1]
                    word = self.word_refine(word)
                    # print(" refined word : ", word)
                    if word in self.sql_words or (48 <= ord(word[0]) and ord(word[0]) <= 57) or word in alias_list:
                        word = ""
                    elif word in self.table_list or word in alias_list:
                        word = ""
                    w.write(word + '\n')

    # query 들을 통해 vocab.txt 생성을 위한 정보들을 따오는 함수
    def table_preProcessing(self):
        q = open('./queries.txt', 'r')
        w = open('./table_vocab.txt', 'w')

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

        for key, value in self.table_list.items():
            w.write(key + " : " + value + '\n')

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

        query_cnt = 0
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
                    query_cnt += 1
                    # print(" query count is ", query_cnt)
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
