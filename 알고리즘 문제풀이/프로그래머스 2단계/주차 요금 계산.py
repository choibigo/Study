from collections import defaultdict
import math

def time_to_minute(time_string):
    hour, minute = time_string.split(":")
    hour = int(hour)
    minute = int(minute)
    
    return hour*60 + minute

def solution(fees, records):
    
    fees = list(map(int, fees))
    time_basic, fee_basic, time_per, fee_per = fees
    
    car_in_time_list = dict()
    car_use_time_list = defaultdict(int)
    
    for record in records:
        time, car_num, op = record.split(" ")

        if op == "IN":
            car_in_time_list[car_num] = time_to_minute(time)
            
        elif op == "OUT":
            car_in_time = car_in_time_list.pop(car_num)
            car_use_time_list[car_num] += time_to_minute(time) - car_in_time

    for car_num, in_time in car_in_time_list.items():
        car_use_time_list[car_num] += time_to_minute("23:59") - in_time
    
    car_use_time_list = sorted(car_use_time_list.items(), key = lambda item : item[0])
    
    answer = list()
    
    for car_num, time in car_use_time_list:
        time = int(time)
        
        if time <= time_basic:
            answer.append(fee_basic)
        else:
            temp = fee_basic + math.ceil((time - time_basic)/time_per) * fee_per 
            answer.append(int(temp))
        
    return answer