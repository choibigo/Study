def solution(clothes):
    
    info = dict()
    for c, key in clothes:
        info[key] = info.get(key, []) + [c]
    
    count = 1
    for value in info.values():
        count *= (len(value)+1)
        
    return count-1