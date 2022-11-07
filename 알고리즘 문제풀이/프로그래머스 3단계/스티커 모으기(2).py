def solution(sticker):
    
    len_sticker = len(sticker)
    if len_sticker == 1:
        return sticker[0]

    res1 = [0] * len_sticker
    res1[0] = sticker[0]
    res1[1] = sticker[0]
    for i in range(2, len_sticker-1):
        res1[i] = max(res1[i-1], res1[i-2]+sticker[i])

    res2 = [0] * len_sticker
    res2[1] = sticker[1]

    for i in range(2, len_sticker):
        res2[i] = max(res2[i-1], res2[i-2]+sticker[i])

    return max(max(res1), max(res2))