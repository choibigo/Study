import input_setting

from collections import deque

def func():
    op_list = ['*','+','-','/']
    start, end = map(int, input().split())
    check = set()

    MAX_SIZE = 10e+9

    if start == end:
        return 0
    else:
       nodes = deque()
       nodes.append([start, ""])

       while nodes:
        pop_num, exp = nodes.popleft()

        if pop_num == end:
            return exp
        
        next_num = pop_num * pop_num
        if 0<=next_num<=MAX_SIZE and next_num not in check:
            check.add(next_num)
            nodes.append([next_num, exp+"*"])
        
        next_num = pop_num + pop_num
        if 0<=next_num<=MAX_SIZE and next_num not in check:
            check.add(next_num)
            nodes.append([next_num, exp+"+"])

        if 1 not in check:
            check.add(1)
            nodes.append([1, exp+"/"])

    return -1


print(func())