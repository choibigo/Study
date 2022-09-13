import input_setting

num, count = map(int, input().split())
flag = 2

if flag == 1:
    
    def DFS(v, s, num, count):

        if v == count:
            print(*res)
    
        for i in range(s, num):
            res.append(i+1)
            DFS(v+1, i+1, num, count)
            res.pop()

    res = list()
    DFS(0, 0, num, count)

elif flag == 2:
    from itertools import combinations

    data = [i for i in range(1, num+1)]

    for c in list(combinations(data, count)):
        print(*c)
