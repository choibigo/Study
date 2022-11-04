def solution(genres, plays):

    genre_info = dict()
    for idx, (genre, play) in enumerate(zip(genres, plays)):
        if genre in genre_info:
            genre_info[genre]['sum'] += play
            genre_info[genre]['play'].append([idx, play])
        else:
            genre_info[genre] = {"sum":play, "play":[[idx, play]]}
    
    genre_info = sorted(genre_info.items(), key = lambda x : -x[1]['sum'] )
    
    answer = list()
    for key, genre_play in genre_info:
        plays = sorted(genre_play['play'], key = lambda x : (-x[1], x[0]))
        answer += [p for p, _ in plays[:2]]
        
    return answer