def nd_num(num):
    
    global res
    
    if num <2 :
        res+=str(num)
        return 
    
    nd_num(num//2)
    res += str(num%2)

res = ""

def solution(n, arr1, arr2):
    
    global res
    
    board = [0] * n
    result = list()
    
    for i in range(n):
        num1 = arr1[i]
        num2 = arr2[i]
        res = ""
        nd_num(num1|num2)
        res = res.rjust(n, "0")
        res = res.replace("1", "#")
        res = res.replace("0", " ")
        result.append(res)
    
    return result