def solution(s):
    
    for i in range(1, len(s)//2):
        result = ""
        temp = s[:i]
        count = 1
        
        for j in range(i, len(s), i):
            if s[j:i+j] == temp:
                count +=1
            else:
                if count != 1:
                    result += (str(count) + temp)                    
                else:
                    result += temp
                
                temp = s[j:i+j]
                count = 1
            
        if count != 1:
            result += (str(count) + temp)                    
        else:
            result += temp
    
    
    return result


if __name__ == "__main__":
    solution("abcabcabcabc6de")