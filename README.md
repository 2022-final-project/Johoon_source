#reference1.py

### [Using Bi LSTM]

`220512`
1. Attemping to make vocab.txt by using queries.txt file

`220513`
1. Simple query 간단하게 한줄로 표현되게끔 구현

`220514`
1. Word counting 성공

`220515`
1. query 조건
   - column 이름을 `desc`, `date` 등과 같이 함수로 쓰이는 단어와 겹치지 않게 한다.
   - alias 부여시 `as` 를 사용한다.
   - `substring` 을 사용하지 않는다.
   - 모두 LIMIT 1; 부여

`220516`
1. 이후 tokenizing 및 BERT 과정을 위한 vocab.txt 베타(?) 버전 제작 완료