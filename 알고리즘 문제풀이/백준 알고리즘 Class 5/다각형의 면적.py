import input_setting


import sys
n = int(input())
point_list = [list(map(int, sys.stdin.readline().split())) for _ in range(n)]
point_list+=[point_list[0]]

a = 0
b = 0
for i in range(n+1):
    if i == 0:
        a += (point_list[i][0]*point_list[i+1][1])

    elif i == n:
        b += (point_list[i][0]*point_list[i-1][1])

    else:
        a += (point_list[i][0]*point_list[i+1][1])
        b += (point_list[i][0]*point_list[i-1][1])

print(round(abs(a-b) / 2, 2))