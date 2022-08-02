import input_setting

def DFS(s):
    if len(res) == count:
        print(*res)
        return

    for i in range(s, num+1):
        res.append(i)
        DFS(i)
        res.pop()

num, count = map(int, input().split())
res = list()
ch = [0] * (num+1)

DFS(1)
