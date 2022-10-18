import input_setting

from collections import deque

def set_prime_num():
    max_num = 9999
    result = set(range(2, max_num+1))

    for i in range(2, max_num):
        if i in result:
            result -= set(range(i*2, max_num+1, i))

    check = [False] * (max_num+1)

    for p in result:
        check[p] = True

    return check

case = int(input())
prime_nums = set_prime_num()
for _ in range(case):
    check = prime_nums[:]
    start, end = map(int, input().split())

    nodes = deque()
    nodes.append([start, 0])
    
    while nodes:
        num, count = nodes.popleft()
        
        if num == end:
            print(count)
            break
        
        for idx in range(4):
            for j in range(10):
                temp = list(str(num))
                temp[idx] = str(j)
                temp = int("".join(temp))

                if temp > 1000 and check[temp]:
                    check[temp] = False
                    nodes.append([temp, count+1])

