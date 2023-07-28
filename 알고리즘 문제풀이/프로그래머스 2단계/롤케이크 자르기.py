def solution(topping):
    len_topping = len(topping)
    left = [0]
    left_set = set()
    for t in topping:
        left_set.add(t)
        left.append(len(left_set))
    
    answer = 0
    right_set = set()
    for i, t in enumerate(topping[::-1]):
        right_set.add(t)
        
        if len(right_set) == left[len_topping-i-1]:
            answer +=1
            
    return answer