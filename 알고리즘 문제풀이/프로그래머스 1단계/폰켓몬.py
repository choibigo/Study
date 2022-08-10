def solution(nums):

    set_nums = len(list(set(nums)))
    pick_num = len(nums) // 2
    
    answer = min(set_nums, pick_num)
    
    return answer