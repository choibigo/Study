import sys
sys.setrecursionlimit(100000)

def preorder(arr, answer):
    node = arr[0]
    arr_left = list()
    arr_right= list()
    
    for i in range(1, len(arr)):
        if arr[i][0] < node[0]:
            arr_left.append(arr[i])
        else:
            arr_right.append(arr[i])
    
    answer.append(node[2])
    if len(arr_left):
        preorder(arr_left, answer)
    if len(arr_right):
        preorder(arr_right, answer)
    return 

def postorder(arr, answer):
    node = arr[0]
    arr_left = list()
    arr_right = list()
    
    for i in range(1, len(arr)):
        if arr[i][0] < node[0]:
            arr_left.append(arr[i])
        else:
            arr_right.append(arr[i])
            
    if len(arr_left):
        postorder(arr_left, answer)
    if len(arr_right):
        postorder(arr_right, answer)
    
    answer.append(node[2])
    return
    
def solution(node_info):

    for idx in range(len(node_info)):
        node_info[idx].append(idx+1)
        
    arr_y = sorted(node_info, key=lambda x : (-x[1], x[0]))
    arr_x = sorted(node_info)
    
    pre_answer = list()
    preorder(arr_y, pre_answer)
    
    post_answer = list()
    postorder(arr_y, post_answer)

    return [pre_answer, post_answer]