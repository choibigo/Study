import input_setting

def DFS(v):
    if v == count:
        print(*res)
        return

    for i in range(1, num+1):
        res.append(i)
        DFS(v+1)
        res.pop()

num, count = map(int, input().split())
res = list()
ch = [0] * (num+1)

DFS(0)