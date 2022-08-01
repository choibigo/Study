import sys

import input_setting # Need To Delete

# 1     2      3      4     5      6      7      8     9     10      11      12      13    14      15    16
# 1     2      3      1     2      3      4      2     1     2       3       4       2     3       4      1

n = int(input())
res = [0] * (n+1)

square = 1
for i in range(1, n+1):
    if i == square * square:
        square +=1
        res[i] = 1
    else:
        temp_list = list()
        for s in range(1, square):
            temp_list.append(res[i - (s*s)] + 1)
        res[i] = min(temp_list)

print(res[n])
