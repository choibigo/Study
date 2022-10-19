import input_setting

import heapq

n = int(input())
class_list = list()

for _ in range(n):
    start, end = map(int, input().split())
    heapq.heappush(class_list, [end, start])


end_time = 0
count = 0

while class_list:
    end, start = heapq.heappop(class_list)

    if start>=end_time:
        count +=1
        end_time = end

print(count)
