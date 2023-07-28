def solution(word):

    info = dict()
    result = list()
    for idx, w in enumerate(word):
        if w not in info:
            result.append(-1)
        else:
            result.append(idx-info[w])
            
        info[w] = idx
    return result