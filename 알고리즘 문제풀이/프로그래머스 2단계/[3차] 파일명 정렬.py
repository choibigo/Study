def solution(files):
    
    file_dict = list()
    
    for file in files:
        
        number = ""
        head = ""
        for i, c in enumerate(file):
            if c.isdigit():
                number+=c
            else:
                if number == "":
                    head += c
                else:
                    file_dict.append([head, number, file])
                    break

            if i== len(file)-1:
                file_dict.append([head, number, file])

    file_dict = sorted(file_dict, key = lambda x : (x[0].lower(), int(x[1]) ))
    
    return [file_name for _,_,file_name in file_dict]