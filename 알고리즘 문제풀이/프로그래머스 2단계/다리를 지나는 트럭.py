from collections import deque 

def solution(bridge_length, weight, wait_truck):

    ing_truck = list()
    sum_weight = 0
    time = 0
    while wait_truck or ing_truck:
        if ing_truck and ing_truck[0][1] == bridge_length:
            sum_weight-= ing_truck[0][0]
            ing_truck.pop(0)
        
        if wait_truck and sum_weight+wait_truck[0] <= weight:
            truck = wait_truck.pop(0)
            ing_truck.append([truck, 0])
            sum_weight += truck
        
        for i in range(len(ing_truck)):
            ing_truck[i][1] += 1
            
        time+=1
        
    return time