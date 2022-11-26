import input_setting

import sys
from collections import deque
case = int(input())

for _ in range(case):
    count, idx = map(int, input().split(" "))
    num_list = deque(list(map(int, sys.stdin.readline().split(" "))))
    answer = 0

    while num_list:

        if len(num_list) == 1:
            answer+=1
            break

        pop_num = num_list.popleft()

        if pop_num >= max(num_list):
            if idx == 0:
                answer+=1
                break
            idx -=1
            answer +=1
        else:
            if idx == 0:
                idx = len(num_list)
            else:
                idx -=1
            
            num_list.append(pop_num)

    print(answer)
    