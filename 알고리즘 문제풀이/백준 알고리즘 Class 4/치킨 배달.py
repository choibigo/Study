import input_setting


import sys
from itertools import combinations
n, chicken_count = map(int, input().split())
board = [sys.stdin.readline().strip().split(' ') for _ in range(n)]

house_list = list()
chicken_list = list()

for row in range(n):
    for col in range(n):
        if board[row][col]=='1':
            house_list.append([col, row])
        elif board[row][col]=='2':
            chicken_list.append([col, row])

result = float('inf')
for candidate in combinations(chicken_list, chicken_count):
    temp_result = 0
    for house in house_list:
        min_value = float('inf')
        for c in candidate:
            min_value = min(min_value, (abs(house[0]-c[0])+abs(house[1]-c[1])))
        temp_result += min_value
    result = min(temp_result, result)

print(result)