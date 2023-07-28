import input_setting

num, count = map(int, input().split())
target_list = list(map(int, input().split()))

def DFS(v, num, count, target_list):
    if v == count:
        print(*res)
        return 

    for i in range(num):
        res.append(target_list[i])
        DFS(v+1, num, count, target_list)
        res.pop()

target_list.sort()
res = list()
DFS(0, num, count, target_list)