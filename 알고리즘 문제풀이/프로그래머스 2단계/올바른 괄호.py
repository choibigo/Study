def solution(s):
    
    stack_list = list()
    
    for bracket in s:
        if bracket =="(":
            stack_list.append(bracket)
        
        elif bracket == ")":
            
            try:
                stack_list.pop()
            except:
                return False
                

    return len(stack_list) == 0
    