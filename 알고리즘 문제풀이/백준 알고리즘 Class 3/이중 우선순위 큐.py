import input_setting


from heapq import heappush, heappop
case = int(input())

for _ in range(case):
    k = int(input())
    min_heap = []
    max_heap = []
    visited = [False]*k

    for key in range(k):
        op, num = input().split()
        num = int(num)
        if op =="I":
            heappush(min_heap, (num, key))
            heappush(max_heap, (-num, key))
            visited[key] = True

        elif op=="D":
            if len(min_heap):
                if num == 1:
                    while max_heap and not visited[max_heap[0][1]]:
                        heappop(max_heap)
                    if max_heap:
                        pop_num = heappop(max_heap)
                        visited[pop_num[1]]=False
                elif num ==-1:
                    while min_heap and not visited[min_heap[0][1]]:
                        heappop(min_heap)
                    if min_heap:
                        pop_num = heappop(min_heap)
                        visited[pop_num[1]] = False

        while max_heap and not visited[max_heap[0][1]]:
            heappop(max_heap)
        while min_heap and not visited[min_heap[0][1]]:
            heappop(min_heap)

    if len(min_heap):
        print(-heappop(max_heap)[0], heappop(min_heap)[0])
    else:
        print("EMPTY")

# Need TO Check