def solution(s, n):
    
    # a : 97
    # z : 122
    
    # A : 65
    # Z : 90
    
    # space : 32
    
    answer = ""
    for ch in s:
        if ch.islower():
            ch_num = (ord(ch) + n) % 123 + int((ord(ch) + n) // 123) * 97
            answer += chr(ch_num)
        elif ch.isupper():
            ch_num = (ord(ch) + n) % 91 + int((ord(ch) + n) // 91) * 65
            answer += chr(ch_num)
        else:
            answer += ch
    
    return answer