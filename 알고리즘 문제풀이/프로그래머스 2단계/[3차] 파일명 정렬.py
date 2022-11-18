def solution(files):

    info = list()
    for file in files:
        number = ""
        head = ""
        tail = ""
        for i in range(len(file)):
            if len(number) and not file[i].isdigit():
                tail = file[i:]
                break
            elif file[i].isdigit():
                number += file[i]
            else:
                head += file[i]
    
        info.append([head, number, tail])
        
    info = sorted(info, key=lambda x: (x[0].lower(), int(x[1])))
    return list(map(lambda x : "".join(x), info))
