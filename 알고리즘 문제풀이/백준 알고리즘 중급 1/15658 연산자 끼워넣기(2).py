import input_setting

n = int(input())
num_list = list(map(int,input().split()))
op_count = list(map(int,input().split()))
max_ans = -1000000000
min_ans = 1000000000

def DFS(v, cal_num):

    global max_ans, min_ans

    if v == n:
        max_ans = max(max_ans, cal_num)
        min_ans = min(min_ans, cal_num)
        return 

    if op_count[0] > 0:
        op_count[0] -= 1
        DFS(v+1, cal_num+num_list[v])
        op_count[0] += 1

    if op_count[1] > 0:
        op_count[1] -= 1
        DFS(v+1, cal_num-num_list[v])
        op_count[1] += 1
    
    if op_count[2] > 0:
        op_count[2] -= 1
        DFS(v+1, cal_num*num_list[v])
        op_count[2] += 1
    
    if op_count[3] > 0:
        op_count[3] -= 1
        DFS(v+1, int(cal_num/num_list[v]))
        op_count[3] += 1


DFS(1, num_list[0])

print(max_ans)
print(min_ans)
