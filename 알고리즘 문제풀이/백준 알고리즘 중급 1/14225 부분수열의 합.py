import input_setting

n = int(input())
num_list = list(map(int, input().split()))
check = [False] * (sum(num_list)+2)

def DFS(index, res):
    if index == n:
        check[sum(res)] = True
        return 

    DFS(index+1, res)
    DFS(index+1, res+[num_list[index]])

DFS(0, [])

print(check[1:].index(False)+1)