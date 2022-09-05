def solution(cacheSize, cities):
    
    memory = list()
    time = 0
    
    if cacheSize == 0:
        return len(cities) * 5
    
    for city in cities:
        city = city.lower()
        if city in memory:
            memory.remove(city)
            memory.append(city)
            time += 1
        else:
            if len(memory) == cacheSize:
                memory.pop(0)
            memory.append(city)
            time += 5
            
    return time