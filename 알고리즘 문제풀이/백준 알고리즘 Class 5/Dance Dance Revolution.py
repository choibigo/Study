import input_setting


from collections import deque
pos_list = list(map(int, input().split()))
len_pos = len(pos_list)
info = dict()
info[0] = {0:1, 1:2, 2:2, 3:2, 4:2}
info[1] = {0:3, 1:1, 2:3, 3:4, 4:3}
info[2] = {0:3, 1:3, 2:1, 3:3, 4:4}
info[3] = {0:3, 1:4, 2:3, 3:1, 4:3}
info[4] = {0:3, 1:3, 2:4, 3:3, 4:1}

dp = [[[0,0],[0,0]] for _ in range(len_pos-1)]
dp[0] = [[pos_list[0],info[0][pos_list[0]]], [pos_list[0], info[0][pos_list[0]]]]

for i in range(1, len_pos-1):
    pass
