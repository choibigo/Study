def solution(price, money, count):
    
    total_money = 0
    for i in range (1, count+1):
        total_money += (i * price)
        
    return max(0, (total_money - money))