import input_setting



import sys
n = int(input())
w_h_list = [list(map(int, sys.stdin.readline().split())) for _ in range(n)]

for i in w_h_list:
    rank = 1
    for j in w_h_list:
        if i[0]<j[0] and i[1]<j[1]:
            rank+=1
    print(rank, end=' ')