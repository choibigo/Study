from collections import defaultdict

def solution(genres, plays):

    info = defaultdict(lambda : {'sum':0, 'id_list':[]})
    for idx, (genre, play) in enumerate(zip(genres, plays)):
        info[genre]['sum'] += play
        info[genre]['id_list'] += [[idx, play]]
    info = sorted(info.items(), key=lambda x : -x[1]['sum'])
    
    answer = []
    for _, value in info:
        temp = sorted(value['id_list'], key=lambda x : -x[1])
        answer += [value[0] for value in temp[:2]]

    return answer