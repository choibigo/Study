import input_setting

from heapq import heappush, heappop
import sys

heap = list()
for _ in range(int(sys.stdin.readline())):
    num = int(sys.stdin.readline())

    if num == 0:
        if len(heap):
            print(heappop(heap)[1])
        else:
            print(0)
    else:
        heappush(heap, (abs(num), num))