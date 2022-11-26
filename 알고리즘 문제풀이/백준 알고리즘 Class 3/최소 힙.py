
import sys

import heapq
heap = list()
for _ in range(int(sys.stdin.readline())):
    num = int(sys.stdin.readline())
    if num:
        heapq.heappush(heap, num)
    else:
        if len(heap):
            print(heapq.heappop(heap))
        else:
            print(0)
