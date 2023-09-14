# coding: utf-8
import sys
sys.path.append('..')
import os
import numpy as np


id_to_char = {}
char_to_id = {}


def _update_vocab(txt):
    chars = list(txt)

    for i, char in enumerate(chars):
        if char not in char_to_id:
            tmp_id = len(char_to_id)
            char_to_id[char] = tmp_id
            id_to_char[tmp_id] = char


def load_data(file_name='addition.txt', seed=1984):
    # file_path = os.path.dirname(os.path.abspath(__file__)) + '/' + file_name
    file_path = file_name

    if not os.path.exists(file_path):
        print('No file: %s' % file_name)
        return None

    questions, answers = [], []

    for line in open(file_path, 'r'):
        idx = line.find('_') # _을 기준으로 Question과 Answer 구분
        questions.append(line[:idx])
        answers.append(line[idx:-1])

    # 어휘 사전 생성
    for i in range(len(questions)):
        q, a = questions[i], answers[i]
        _update_vocab(q)
        _update_vocab(a)


    # 넘파이 배열 생성
    # x: 총 학습데이터의 크기 x 학습데이터 1개길이(padding 되어서 고정된 크기)
    x = np.zeros((len(questions), len(questions[0])), dtype=np.int32)
    t = np.zeros((len(questions), len(answers[0])), dtype=np.int32)

    # padding 비어있는 numpy 배열에 값을 채워 넣는 형식으로 Padding을 구현
    for i, sentence in enumerate(questions):
        x[i] = [char_to_id[c] for c in list(sentence)]
    for i, sentence in enumerate(answers):
        t[i] = [char_to_id[c] for c in list(sentence)]

    # 뒤섞기
    indices = np.arange(len(x))
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(indices)
    x = x[indices]
    t = t[indices]

    # 검증 데이터셋으로 10% 할당
    split_at = len(x) - len(x) // 10
    (x_train, x_test) = x[:split_at], x[split_at:]
    (t_train, t_test) = t[:split_at], t[split_at:]

    return (x_train, t_train), (x_test, t_test)


def get_vocab():
    return char_to_id, id_to_char