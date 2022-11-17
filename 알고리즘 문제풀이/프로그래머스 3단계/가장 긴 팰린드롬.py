def solution(s):
    for size in range(len(s), 0, -1):
        for i in range(len(s)-size+1):
            if s[i:i+size] == s[i:i+size][::-1]:
                return size
        