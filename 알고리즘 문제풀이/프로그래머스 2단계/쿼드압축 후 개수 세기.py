def solution(arr):
    result = [0,0]
    
    def comporession(a, b, l):
        start = arr[a][b]
        
        for i in range(a, a+l):
            for j in range(b, b+l):
                if start != arr[i][j]:
                    comporession(a, b, l//2)
                    comporession(a+l//2, b, l//2)
                    comporession(a, b+l//2, l//2)
                    comporession(a+l//2, b+l//2, l//2)
                    return
                
        result[start] +=1
    
    comporession(0,0,len(arr))
    
    return result