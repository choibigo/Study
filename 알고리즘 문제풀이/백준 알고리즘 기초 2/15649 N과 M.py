import input_setting

def DFS(v):
    if v == count:
        print(*res)
    else:
        for i in range(1, num+1):
            if ch[i] == 0:
                ch[i] = 1
                res.append(i)
                DFS(v+1)
                ch[i] = 0
                res.pop()

num, count = map(int, input().split())

res = list()
ch = [0] * (num+1)

DFS(0)