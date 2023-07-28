import input_setting

flag = 1
num, count = map(int, input().split())

if flag == 1:
    pass
    check = [False]*(num+1)
    def DFS(v, res):
        if v == count:
            print(*res)
            return 
        
        for i in range(1, num+1):
            if not check[i]:
                check[i] = True
                DFS(v+1, res+[i])
                check[i] = False

    DFS(0, [])

elif flag == 2:
    from itertools import permutations

    data = [i for i in range(1, num+1)]
    
    for l in list(permutations(data, count)):
        print(*l)