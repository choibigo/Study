def solution(bracket_list):

    stack = list()
    
    for b in bracket_list:
        if b == "(":
            stack.append("(")
        else:
            if len(stack) == 0:
                return False
            else:
                if stack[-1] =="(":
                    stack.pop()
                else:
                    return False
                
    if len(stack) == 0:
        return True
    else:
        return False