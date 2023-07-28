import input_setting

import sys
n = int(input())
num_list = list(map(int, sys.stdin.readline().split( )))
num_list.sort()

search_n = int(input())
search_list = list(map(int, sys.stdin.readline().split( )))

def binary_search(num, num_list):
    left = 0
    right = n-1

    while left <= right:
        mid = (left+right)//2

        if num == num_list[mid]:
            return True
        elif num < num_list[mid]:
            right = mid-1
        elif num > num_list[mid]:
            left = mid+1
    return False

for search in search_list:
    if binary_search(search, num_list):
        print(1)
    else:
        print(0)

