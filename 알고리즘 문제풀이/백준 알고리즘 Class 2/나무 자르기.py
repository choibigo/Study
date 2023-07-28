import input_setting

import sys
tree_count, need_len = map(int, sys.stdin.readline().split(" "))
tree_list = list(map(int, sys.stdin.readline().split(" ")))

if need_len == 0:
    print(max(tree_list))
else:

    left = 1
    right = max(tree_list)
    answer = 0
    while left<=right:
        mid = (left+right)//2

        tree_len=0
        for t in tree_list:
            tree_len += max(0, t-mid)

        if tree_len < need_len:
            right = mid-1
        else:
            answer = mid
            left = mid+1
    
    print(answer)