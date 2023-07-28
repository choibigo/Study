import input_setting


import sys
from heapq import heappush, heappop

n = int(input())
heap = list()
for _ in range(n):
    num = int(sys.stdin.readline())
    heappush(heap, num)

for _ in range(n):
    print(heappop(heap))