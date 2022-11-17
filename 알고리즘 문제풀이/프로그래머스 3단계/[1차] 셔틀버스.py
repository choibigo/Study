import heapq

def num_to_hour(num):
    hour, miniute = divmod(num, 60)
    if hour<10:
        hour = f"0{hour}"
    if miniute<10:
        miniute = f"0{miniute}"
    return f"{hour}:{miniute}"

def solution(n, t, m, timetable):
    heap=[int(i[:2])*60+int(i[3:]) for i in timetable]
    heapq.heapify(heap)
    
    for i, bus_time in enumerate([540+(i*t) for i in range(n)]):
        people_count = 0
        while heap and people_count < m:
            pop_time = heapq.heappop(heap)
            if pop_time <= bus_time:
                last_time = pop_time
                people_count += 1
            else:
                heapq.heappush(heap, pop_time)
                break
        
        if i == n-1:
            if people_count == m:
                return num_to_hour(last_time-1)
            else:
                return num_to_hour(bus_time)
    