def solution(s):
    
    stack_list = list()
    
    for c in s:
        if len(stack_list) == 0:
            stack_list.append(c)
        else:
            if stack_list[-1] == c:
                stack_list.pop()
            else:
                stack_list.append(c)

    return 1 if len(stack_list) == 0 else 0