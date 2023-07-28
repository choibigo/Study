import input_setting

from collections import deque

n = sum(list(map(int, input().split())))
laber_snake = dict()

# region BFS
# check = [True] + [False for _ in range(100)]

# for _ in range(n):
#     start, end = map(int, input().split())
#     laber_snake[start] = end

# nodes = deque()
# nodes.append([1,0])

# while nodes:
#     pos, count = nodes.popleft()

#     if pos == 100:
#         print(count)
#         break

#     for move in range(1, 7):
#         n_pos = pos + move
        
#         if n_pos<=100 and not check[n_pos]:
#             if n_pos in laber_snake:
#                 n_pos = laber_snake[n_pos]

#             if not check[n_pos]:
#                 check[n_pos] = True
#                 nodes.append([n_pos, count+1])
# endregion
