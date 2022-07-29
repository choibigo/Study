import sys
import copy 

import input_setting # Need To Delete

n = int(input())
num_list = list(map(int, sys.stdin.readline().split()))

res = [1] * (n)
res_list = list()

for num in num_list:
    res_list.append([num])

for index in range(1, n):
    for j in range(0, index):
        if num_list[index] > num_list[j] and res[index] < res[j] + 1:
            res[index] = res[j] + 1
            temp = copy.deepcopy(res_list[j])
            temp.append(num_list[index])
            res_list[index] = temp


max_value = max(res)
max_index = res.index(max_value)

print(max_value)
print(*res_list[max_index])

