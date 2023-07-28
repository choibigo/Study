import input_setting

num, count = map(int, input().split())
target_list = list(map(int, input().split()))


def DFS(v, s, num, count, target_list):
    if v == count:
        print(*res)
        return 

    for i in range(s, num):
        res.append(target_list[i])
        DFS(v+1, i, num, count, target_list)
        res.pop()

target_list.sort()
res = list()
DFS(0, 0, num, count, target_list)