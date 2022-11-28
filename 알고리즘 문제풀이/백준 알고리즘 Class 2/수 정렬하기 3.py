import input_setting


import sys
res = [0] * 10001

for _ in range(int(input())):
    num = int(sys.stdin.readline())
    res[num] +=1

for i in range(10001):
    if res[i]:
        for _ in range(res[i]):
            print(i)