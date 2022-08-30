from collections import deque

def solution(bridge_length, weight, truck_weights):
    
    truck_weights = deque(truck_weights)
    ing_trucks = deque()
    truck_sum = 0

    print(truck_weights[0])
    
    # Debug
    break_i = 1
    
    while len(truck_weights) != 0 or len(ing_trucks) != 0:

        if len(truck_weights) != 0:
            if truck_sum + truck_weights[0] <= weight and len(ing_trucks) < bridge_length:
                truck_sum += truck_weights[0]
                ing_trucks.append([0, truck_weights.popleft()])
        
        if len(ing_trucks) != 0:
            if ing_trucks[0][0] == bridge_length:
                truck_sum-=ing_trucks[0][1]
                ing_trucks.popleft()

            for i in range(len(ing_trucks)):
                ing_trucks[i][0] += 1
                
                
        print(f"#### {break_i} ####")
        print(truck_weights)    
        print(ing_trucks)  

        if break_i > 10:
            break
            
        break_i+=1

    
    return 1

solution(2, 10, [7,4,5,6])