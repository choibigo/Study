import input_setting

import sys

# region Heap 구현

def heap_insert(heap, num):
    heap.append(num)
    child = len(heap)-1
    while child >1:
        parent = child//2
        if heap[parent] < heap[child]:
            heap[parent], heap[child] = heap[child], heap[parent]
            child = parent
        else:
            break

def heap_delete(heap):
    heap[1], heap[-1] = heap[-1], heap[1]
    result = heap.pop()
    parent = 1
    while 2*parent <= len(heap)-1:
        left = parent*2
        if parent*2+1 <= len(heap)-1:
            right = parent*2+1 
            if heap[left]>heap[right]:
                if heap[parent]<heap[left]:
                    heap[left], heap[parent] = heap[parent], heap[left]
                    parent = left
                else:
                    break
            else:
                if heap[parent]<heap[right]:
                    heap[right], heap[parent] = heap[parent], heap[right]
                    parent = right
                else:
                    break
        else:
            if heap[parent]<heap[left]:
                heap[left], heap[parent] = heap[parent], heap[left]
                parent = left
            else:
                break
    return result

heap = [-1]
for _ in range(int(sys.stdin.readline())):
    num = int(sys.stdin.readline())

    if num:
        heap_insert(heap, num)
    else:
        if len(heap) > 1:
            print(heap_delete(heap))
        else:
            print(0)
# endregion


# region heapq Lib 사용

# import heapq
# heap = list()
# for _ in range(int(sys.stdin.readline())):
#     num = int(sys.stdin.readline())
#     if num:
#         heapq.heappush(heap, -num)
#     else:
#         if len(heap):
#             print(-heapq.heappop(heap))
#         else:
#             print(0)

# endregion


