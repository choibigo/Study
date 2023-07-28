def check(bracket_list):
    
    stack = list()
    
    for b in bracket_list:
        
        if len(stack) == 0:
            if b in ["]", "", "}"]:
                return 0
            
            stack.append(b)
        elif b in ["[", "(", "{"]:
            stack.append(b)
        else:
            if b == "]":
                if stack[-1] == "[":
                    stack.pop()
                else:
                    return 0
            elif b == ")":
                if stack[-1] == "(":
                    stack.pop()
                else:
                    return 0
            elif b == "}":
                if stack[-1] == "{":
                    stack.pop()
                else:
                    return 0
        
    return 1 if len(stack) == 0 else 0 

def solution(s):

    count = 0
    for _ in range(len(s)):
        s = s[1:]+s[0]
        count += check(s)
            
    return count
def check(bracket_list):
    
    stack = list()
    
    for b in bracket_list:
        
        if len(stack) == 0:
            if b in ["]", "", "}"]:
                return 0
            
            stack.append(b)
        elif b in ["[", "(", "{"]:
            stack.append(b)
        else:
            if b == "]":
                if stack[-1] == "[":
                    stack.pop()
                else:
                    return 0
            elif b == ")":
                if stack[-1] == "(":
                    stack.pop()
                else:
                    return 0
            elif b == "}":
                if stack[-1] == "{":
                    stack.pop()
                else:
                    return 0
        
    return 1 if len(stack) == 0 else 0 

def solution(s):

    count = 0
    for _ in range(len(s)):
        s = s[1:]+s[0]
        count += check(s)
            
    return count
