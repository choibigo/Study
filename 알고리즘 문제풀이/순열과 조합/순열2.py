import input_setting

num, count = map(int, input().split())
target_list = list(map(int, input().split()))

flag = 2

if flag == 1:
    def DFS(v, num, count, target_list):
        if v == count:
            print(*res) 
            return 

        for i in range(num):
            if ch[i] == False:
                ch[i] = True
                res.append(target_list[i])

                DFS(v+1, num, count, target_list)

                ch[i] = False
                res.pop()

    res = list()
    ch = [False] * num
    target_list.sort()
    DFS(0, num, count, target_list)

elif flag == 2:
    from itertools import permutations
    target_list.sort()
    for p in list(permutations(target_list, count)):
        print(*p)

