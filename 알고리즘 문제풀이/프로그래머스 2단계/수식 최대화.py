from itertools import permutations

def operation(num1, num2, exp):
    if exp == "*":
        return str(int(num1) * int(num2))
    if exp == "+":
        return str(int(num1) + int(num2))
    if exp == "-":
        return str(int(num1) - int(num2))

def solution(expression):
    answer = 0
    
    t_express = list()
    temp = ""
    for exp in str(expression):
        if exp.isdigit():
            temp+=exp
        else:
            t_express.append(temp)
            t_express.append(exp)
            temp=""
    
    t_express.append(temp)
    
    for operations in list(permutations(["*","+","-"], 3)):
        express = t_express[:]
        for o in operations:
            stack_list = list()
            while len(express) != 0:
                pop_v = express.pop(0)
                
                if pop_v == o:
                    left_num = stack_list.pop()
                    right_num = express.pop(0)
                    stack_list.append(operation(left_num, right_num, o))
                    
                else:
                    stack_list.append(pop_v)
            express = stack_list[:]
        
        answer = max(abs(int(express[0])), answer)
            
    return answer
