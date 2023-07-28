import input_setting


import sys
for r in sorted([sys.stdin.readline().strip().split(' ') for _ in range(int(input()))], key=lambda x : int(x[0])):
    print(*r)