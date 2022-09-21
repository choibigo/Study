def dp(sticker):
    res = [0] * len(sticker)
    res[0] = sticker[0]
    res[1] = max(sticker[0], sticker[1])

    for i in range(2, len(sticker)-1):
        res[i] = max(res[i-1], res[i-2]+sticker[i])
    
    return max(res)
    

def solution(sticker):

    if len(sticker) == 1:
        return sticker[0]
    
    a = dp(sticker)
    
    sticker = sticker[1:] + [sticker[0]]
    b = dp(sticker)
    
    return max(a, b)
    
    