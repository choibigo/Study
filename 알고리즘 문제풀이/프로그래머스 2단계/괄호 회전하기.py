def rotate(str_list):
    temp = str_list[1:]
    temp.append(str_list[0])
    return temp

def check_str(target_str):
    stack_list = list()
    
    for i in range(len(target_str)):
        char = target_str[i]
        
        if char.isalpha():
            pass
        else:
            if char == "[" or char == "(" or char == "{":
                stack_list.append(char)
            else:
                if len(stack_list) == 0:
                    return False
                else:
                    if char =="]":
                        if stack_list[-1] == "[":
                            stack_list.pop()
                    elif char ==")":
                        if stack_list[-1] == "(":
                            stack_list.pop()
                    elif char =="}":
                        if stack_list[-1] == "{":
                            stack_list.pop()
    return len(stack_list) == 0

def solution(s):
    
    target_str = list(s)
    
    answer = 0
    for _ in range(len(target_str)):
        target_str = rotate(target_str)
        if check_str(target_str):
            answer += 1
    
    return answer