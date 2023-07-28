def solution(brown, yellow):
    
    brown = brown-4
    for i in range(1, brown//2):
        if (brown//2-i)*i == yellow:
            return [brown//2-i+2, i+2]
            