def solution(n):
    
    # 1 2 3 4 5 6 7 8 9 10
    # 1 2 3 5 

    if n<=3:
        return n
    
    dp = [0] *(n+1)
    
    dp[1] = 1
    dp[2] = 2
    dp[3] = 3
    
    for i in range(4, n+1):
        dp[i] = (dp[i-1] + dp[i-2]) % 1234567
    
    return dp[-1]