from itertools import combinations

def solution(relation):
    
    keys = [i for i in range(len(relation[0]))]
    keys_combination=list()
    for i in range(1, len(relation[0])+1):
        keys_combination += list(combinations(keys, i))

    answer = list()
    for keys in keys_combination:
        keys = set(keys)
        search = set()
        for r in relation:
            colum = ""
            for k in keys:
                colum += r[k]
            
            search.add(colum)
            
        if len(search) == len(relation):
            flag = True
            for a in answer:
                if a <= keys:
                    flag = False
                    
            if flag:
                answer.append(keys)
                
    return len(answer)