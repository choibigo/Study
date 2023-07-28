import input_setting

num, count = map(int, input().split())
flag = 1

if flag == 1:
    
    def DFS(v, s, res):
        if v == count:
            print(res)
            return
        
        for i in range(s, num+1):
            DFS(v+1, i+1, res+[i])

    DFS(0,1,[])

elif flag == 2:
    from itertools import combinations

    data = [i for i in range(1, num+1)]

    for c in list(combinations(data, count)):
        print(*c)
