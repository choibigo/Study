import heapq

n = int(input())
plus_list = list()
minus_list = list()
one_count = 0

for _ in range(n):
    temp = int(input())

    if temp == 1:
        one_count+=1
    elif temp > 0:
        heapq.heappush(plus_list, temp * -1)
    elif temp <= 0:
        heapq.heappush(minus_list, temp)
    

total = one_count
while len(plus_list) > 1:
    pop_num1 = heapq.heappop(plus_list)
    pop_num2 = heapq.heappop(plus_list)
    total += pop_num1 * pop_num2

if plus_list:
    total += plus_list[0] * -1


while len(minus_list) > 1:
    pop_num1 = heapq.heappop(minus_list)
    pop_num2 = heapq.heappop(minus_list)

    if pop_num1 == 0 and pop_num2 == 0:
        break

    total += pop_num1 * pop_num2

if minus_list:
    total += minus_list[0]

print(total)

