def divisor_count(num, limit, power):
    if num ==1:
        return 1
    result = 0
    for i in range(1, int(num**0.5)+1):
        if num%i==0:
            if i == num//i:
                result +=1
            else:
                result +=2
                
    return result if result<=limit else power

def solution(number, limit, power):
    result = [divisor_count(i, limit, power) for i in range(1, number+1)]
    return sum(result)