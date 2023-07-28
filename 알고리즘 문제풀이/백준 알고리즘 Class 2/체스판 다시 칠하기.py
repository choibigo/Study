import input_setting


import sys
rows, cols = map(int, input().split())
board = [list(sys.stdin.readline().strip()) for _ in range(rows)]
b_w = ['B', 'W']

def check(row, col, min_value, idx):
    res = 0
    for r in range(row, row+8):
        idx ^= 1
        for c in range(col, col+8):
            if board[r][c] != b_w[idx]:
                res+=1
            if res >= min_value:
                return min_value
            idx ^= 1
    return res

min_value = 64
for row in range(rows-7):
    for col in range(cols-7):
        min_value = min(min_value, check(row, col, min_value, 0), check(row, col, min_value, 1))

print(min_value)