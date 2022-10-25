import math

def time_to_miniute(time):
    hour, miniute = time.split(":")
    return int(hour)*60+int(miniute)

def solution(fees, records):
    basic_time, basic_fee, per_time, per_fee = fees
    
    time_info = dict()
    status_info = dict()
    
    for r in records:
        time, number, act = r.split(" ")
        
        if act == "IN":
            status_info[number] = time_to_miniute(time)
        elif act == "OUT":
            time_info[number] = time_info.get(number, 0) + time_to_miniute(time) - status_info[number]
            del status_info[number]

    for number in status_info.keys():
        time_info[number] = time_info.get(number, 0) + time_to_miniute("23:59") - status_info[number]
    
    answer = list()
    time_info = sorted(time_info.items(), key = lambda x: x[0])
    for _, time in time_info:
        if time<=basic_time:
            fee = basic_fee
        else:
            fee = basic_fee + math.ceil((time-basic_time)/per_time) * per_fee
        answer.append(fee)
        
    return answer

