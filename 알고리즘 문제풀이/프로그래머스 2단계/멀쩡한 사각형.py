def gcd(num1, num2):
    while(num2):
        num1, num2 = num2, num1%num2
    
    return num1

def solution(w,h):
    
    return w*h - (w+h - gcd(w, h))