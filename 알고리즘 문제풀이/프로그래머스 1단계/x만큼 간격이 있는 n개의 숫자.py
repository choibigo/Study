def solution(x, n):
    
#     if x == 0:
#         return [0] * n
    
#     min_v = min(x, x*n)
#     max_v = max(x, x*n)
    
#     ans = list(range(min_v, max_v+1, abs(x)))
#     ans.sort(key = lambda x : abs(x))
    
    
    
#     return ans

    return [i*x for i in range(1, n+1)]