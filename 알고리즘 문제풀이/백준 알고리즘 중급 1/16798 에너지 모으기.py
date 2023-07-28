import input_setting

n = int(input())
num_list = list(map(int, input().split()))
max_val = 0

def DFS(sum_val):
    global max_val
    if len(num_list) == 2:
        max_val = max(max_val, sum_val)
        return 
        
    for i in range(1, len(num_list)-1):
        val = num_list[i-1] * num_list[i+1]
        origin = num_list[i]
        del num_list[i]
        DFS(sum_val + val)
        num_list.insert(i, origin)

DFS(0)
print(max_val)