import input_setting

from collections import deque

def get_prime_numbers(n=9999):

    prime_nums = set(range(2, n+1))
    for i in range(2, n):
        if i in prime_nums:
            prime_nums -= set(range(i*2, n+1, i))

    origin_check_list = [False] * (n+1)
    for p in prime_nums:
        origin_check_list[p] = True

    return origin_check_list

def BFS():
    first_num, target = map(int, input().split())
    check_list = origin_check_list[:]
    nodes = deque()
    nodes.append([first_num, 0])
    
    while nodes:
        num, count = nodes.popleft()
        if num == target:
            return count

        num = list(str(num))

        for d in range(len(num)):
            for i in range(0, 10):
                if num[d] != i:
                    next_num = num[:]
                    next_num[d] = str(i)
                    next_num = "".join(next_num)
                    next_num = int(next_num)

                    if next_num >= 1000 and check_list[next_num]:
                        check_list[next_num] = False
                        nodes.append([next_num, count+1])

    return "Impossible"

origin_check_list = get_prime_numbers()
case = int(input())
for _ in range(case):
    print(BFS())
