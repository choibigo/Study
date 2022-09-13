import input_setting

flag = 2
num, count = map(int, input().split())

if flag == 1:
    def DFS(v, num, count):

        if v == count:
            print(*res)
            return
        
        for i in range(num):
            if check[i] == False:
                check[i] = True
                res.append(i+1)
                
                DFS(v+1, num, count)

                check[i] = False
                res.pop()

    res = list()
    check = [False] * num
    DFS(0, num, count)
elif flag == 2:
    from itertools import permutations

    data = [i for i in range(1, num+1)]
    
    for l in list(permutations(data, count)):
        print(*l)