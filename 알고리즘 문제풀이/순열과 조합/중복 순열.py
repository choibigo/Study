import input_setting

num, count = map(int, input().split())

flag = 1

if flag == 1:
    
    def DFS(v, res):
        if v == count:
            print(res)
            return 
        
        for i in range(1, num+1):
            DFS(v+1, res+[i])

    DFS(0, [])

elif flag == 2:
    
    from itertools import product

    data = [i for i in range(1, num+1)]

    for d in list(product(data, repeat = count)):
        print(*d)