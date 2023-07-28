import input_setting

from collections import deque

case = int(input())
for _ in range(case):
    check = [False] * 10000
    num, target = map(int, input().split())
    nodes = deque()
    nodes.append([num, ""])
    check[num] = True

    while nodes:
        pop_num, op_list = nodes.popleft()
        
        if pop_num == target:
            print(op_list)
            break

        #1
        num2 = (2*pop_num)%10000
        if not check[num2]:
            check[num2] = True
            nodes.append([num2, op_list+"D"]) 

        #2
        num2 = (pop_num-1)%10000
        if not check[num2]:
            check[num2] = True
            nodes.append([num2, op_list+"S"]) 

        #3
        num2 = (10*pop_num+(pop_num//1000))%10000
        if not check[num2]:
            check[num2] = True
            nodes.append([num2, op_list+"L"]) 

        #4
        num2 = (pop_num//10+(pop_num%10)*1000)%10000
        if not check[num2]:
            check[num2] = True
            nodes.append([num2, op_list+"R"]) 