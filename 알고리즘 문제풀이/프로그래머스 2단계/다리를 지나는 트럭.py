from collections import deque

def solution(bridge_length, weight, truck_weights):
    
    truck_weights = deque(truck_weights)
    ing_trucks = deque()
    
    ttt =1
    weight_sum = 0
    
    while len(truck_weights) !=0 or len(ing_trucks) != 0:
        
        if len(ing_trucks) != 0:
            for i in range(len(ing_trucks)):
                ing_trucks[i][0] += 1
                
            if ing_trucks[0][0] == bridge_length:
                temp = ing_trucks.popleft()
                weight_sum -= temp[1]
        
        if len(truck_weights) != 0:
            if weight_sum + truck_weights[0] <= weight and len(ing_trucks)< bridge_length:
                temp = truck_weights.popleft()
                ing_trucks.append([0, temp])
                weight_sum += temp
        
        ttt+=1
    
    return ttt-1