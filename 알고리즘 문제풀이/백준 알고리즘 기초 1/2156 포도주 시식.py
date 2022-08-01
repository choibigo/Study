import input_setting

import sys

n = int(input())
wine_list = list()

for _ in range(n):
    wine_list.append(int(input()))

if n == 1:
    print(wine_list[0])
elif n == 2:
    print(wine_list[0] + wine_list[1])
elif n == 3:
    print(max(wine_list[0] + wine_list[2], wine_list[1] + wine_list[2], wine_list[0] + wine_list[1]))

else:
    res = [0] * n
    res[0] = wine_list[0]
    res[1] = wine_list[0] + wine_list[1]
    res[2] = max(wine_list[0] + wine_list[2], wine_list[1] + wine_list[2], wine_list[0] + wine_list[1])

    for i in range(3, n):
        val_1 = res[i-2] + wine_list[i]
        val_2 = res[i-3] + wine_list[i-1] + wine_list[i]
        val_3 = res[i-1]
        res[i] = max(val_1, val_2, val_3) 

    print(max(res))