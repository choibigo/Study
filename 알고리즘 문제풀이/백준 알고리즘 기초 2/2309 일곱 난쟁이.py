import input_setting

import sys

def DFS(v, s):
    if v==2:
        res_sum = num_list[res[0]] + num_list[res[1]]
        if (sum_num_list - res_sum) == 100:
            val_1 = num_list[res[0]]
            val_2 = num_list[res[1]]

            num_list.remove(val_1)
            num_list.remove(val_2)
            num_list.sort()

            for num in num_list:
                print(num)

            sys.exit()
    else:
        for i in range(s, 9):
            res.append(i)
            DFS(v+1, i+1)
            res.pop()


num_list = list()
for _ in range(9):
    num_list.append(int(input()))

sum_num_list = sum(num_list)

res = list()
DFS(0,0)