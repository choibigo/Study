def solution(target_list):
    
    target_list = target_list[1:-1]
    target_list = target_list.replace(",{", "")
    target_list = target_list.replace("{", "")
    target_list = target_list.split("}")
    target_list = sorted(target_list, key = lambda x : len(x))
    
    answer = list()
    
    for target in target_list[1:]:
        temp = target.split(",")
        
        for t in temp:
            if int(t) not in answer:
                answer.append(int(t))
        
    
    return answer
