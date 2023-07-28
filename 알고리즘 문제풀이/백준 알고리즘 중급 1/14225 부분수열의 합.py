import input_setting

n = int(input())
num_list = list(map(int, input().split()))
check = [False] * (sum(num_list) + 2)

def DFS(v, sum_val):
    if v == n:
        check[sum_val] = True
        return

    DFS(v+1, sum_val + num_list[v])
    DFS(v+1, sum_val)


DFS(0, 0)
print(check.index(False))    