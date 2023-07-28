import input_setting


# Need To Check
import sys
sys.setrecursionlimit(200000)
n, m = map(int, input().split())
parent = [i for i in range(n)]

def find_parent(x):
    if x == parent[x]:
        return x
    else:
        parent[x] = find_parent(parent[x])
        return parent[x]

def union(x, y):
    x = find_parent(x)
    y = find_parent(y)

    if x<y:
        parent[y] = x
    else:
        parent[x] = y

for i in range(m):
    a, b = map(int, sys.stdin.readline().split())
    if find_parent(a) == find_parent(b):
        print(i+1)
        sys.exit(0)
    union(a,b)

print(0)