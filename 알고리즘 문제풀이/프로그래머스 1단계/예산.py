def solution(d_list, budget):
    
    d_list.sort()
    money = budget
    count = 0
    
    for d in d_list:
        if money >= d:
            money -= d
            count +=1
        else:
            break
    
    return count