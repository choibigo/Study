def correct_bracket(bracket_list):
    
    stack = list()
    
    for bracket in bracket_list:
        if bracket == "(":
            stack.append("(")
        else:
            if len(stack) == 0:
                return False
            else:
                if stack[-1] == "(":
                    stack.pop()
    
    return True if len(stack) == 0 else False

def reverse_bracket(bracket_list):
    temp = ""
    for bracket in bracket_list:
        if bracket == ")":
            temp += "("
        else:
            temp += ")"
    return temp

def logic(bracket_list):
        if len(bracket_list) == 0:
            return ""
    
        bracket_check = [0,0]
        index = 0
        for i, bracket in enumerate(bracket_list):
            if bracket == ")":
                bracket_check[0] += 1
            else:
                bracket_check[1] += 1
            
            if bracket_check[0] == bracket_check[1]:
                index = i
                break
        
        u = bracket_list[:index+1]
        v = bracket_list[index+1:]
        
        if correct_bracket(u):
            return u + logic(v)
        else:
            temp = "(" + logic(v) + ")" + reverse_bracket(u[1:-1])
            return temp


def solution(p):
    return logic(p)