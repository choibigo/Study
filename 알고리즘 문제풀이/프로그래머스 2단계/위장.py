def solution(clothes):
    
    clothes_dict = dict()
    
    for cloth in clothes:
        value, key = cloth
        
        if key in clothes_dict:
            clothes_dict[key].append(value)
        else:
            clothes_dict[key] = list()
            clothes_dict[key].append(value)
    
    answer = 1
    for v in clothes_dict.values():
        answer *= (len(v)+1)
    
    answer -=1
    
    return answer