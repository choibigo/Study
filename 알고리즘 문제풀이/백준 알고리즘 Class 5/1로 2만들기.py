import input_setting


from collections import deque
n = int(input())
res = [-1 for _ in range(n+1)]
nodes = deque([n])

while nodes:
    x = nodes.popleft()
    if x == 1:
        break

    if x%3 == 0 and res[x//3]==-1:
        res[x//3] = x
        nodes.append(x//3)

    if x%2 == 0 and res[x//2]==-1:
        res[x//2] = x
        nodes.append(x//2)

    if x-1 > 0 and res[x-1]==-1:
        res[x-1] = x
        nodes.append(x-1)

v = 1
result = list()
while v != -1:
    result.append(v)
    v = res[v]

print(len(result)-1)
print(*result[::-1])