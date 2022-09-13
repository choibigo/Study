import input_setting

num, count = map(int, input().split())

flag = 2

if flag == 1:
    
    def DFS(v, num, count):
        
        if v == count:
            print(*res)
            return 
        
        for i in range(num):
            res.append(i+1)
            DFS(v+1, num, count)
            res.pop()

    res = list()
    DFS(0, num, count)

elif flag == 2:
    
    from itertools import product

    data = [i for i in range(1, num+1)]

    for d in list(product(data, repeat = count)):
        print(*d)