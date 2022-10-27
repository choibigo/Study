from collections import deque 

def solution(bridge_length, weight, wait_truck):

    ing_truck = list()
    sum_weight = 0
    time = 0
    while wait_truck or ing_truck:
        if ing_truck and time - ing_truck[0][1] == bridge_length:
            sum_weight-= ing_truck[0][0]
            ing_truck.pop(0)
        
        if wait_truck and sum_weight+wait_truck[0] <= weight:
            truck = wait_truck.pop(0)
            ing_truck.append([truck, time])
            sum_weight += truck
        
        time+=1
        
    return time