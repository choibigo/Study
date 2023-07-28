def solution(cacheSize, cities):

    if cacheSize == 0: return len(cities)*5
    
    cache = list()
    answer = 0
    for city in cities:
        city = city.lower()
        
        if city in cache:
            cache.remove(city)
            cache.append(city)
            answer +=1
        else:
            if len(cache) == cacheSize:
                del cache[0]
                
            cache.append(city)
            answer += 5
            
    return answer