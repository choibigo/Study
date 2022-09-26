def solution(stones, k):
    
    left = 1
    right = 200000000
    
    while left<= right:
        temp_stones = stones[:]
        mid = (left+right) // 2
        
        count = 0
        for t in temp_stones:
            if t-mid <= 0:
                count+=1
            else:
                count = 0
                
            if count >=k:
                break
        
        if count >= k:
            right = mid-1
        else:
            left = mid+1
        
    return left