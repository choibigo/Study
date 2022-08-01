import sys

import input_setting

def dp(num):
    if num <= 3:
        return [0, 1, 2, 4]
    else:
        res = [0] * (num + 1)
        res[1] = 1
        res[2] = 2
        res[3] = 4

        for i in range(4, num+1):
            res[i] = (res[i-1] + res[i-2] + res[i-3]) % 1000000009

        return res

case = int(input())
num_list = list()

max_num = 0
for _ in range(case):
    temp = int(input())
    num_list.append(temp)
    if max_num < temp:
        max_num = temp

result = dp(max_num)

for num in num_list:
    print(result[num])



