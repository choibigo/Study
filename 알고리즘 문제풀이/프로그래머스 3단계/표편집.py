def solution(n, select, cmd_list):

    answer = ["O" for _ in range(n)]
    info = {i:[i-1, i+1] for i in range(n)}
    info[0] = [None, 1]
    info[n-1] = [n-2, None]
    delete_list = list()
    
    for cmd in cmd_list:
        cmd = cmd.split(" ")
        
        if cmd[0] == "C":
            answer[select] = "X"
            prev, nnext = info[select]
            delete_list.append([select, prev, nnext])
            
            if prev == None:
                info[nnext][0] = None
            elif nnext == None:
                info[prev][1] = None
            else:
                info[nnext][0] = prev
                info[prev][1] = nnext
                
            if nnext == None:
                select = info[select][0]
            else:
                select = nnext
        
        elif cmd[0] == "Z":
            idx, prev, nnext = delete_list.pop()
            answer[idx] = "O"
            
            if prev == None:
                info[nnext][0] = idx
            elif nnext == None:
                info[prev][1] = idx
            else:
                info[nnext][0] = idx
                info[prev][1] = idx
        
        elif cmd[0] == "D":
            for _ in range(int(cmd[1])):
                select = info[select][1]
            
        elif cmd[0] == "U":
            for _ in range(int(cmd[1])):
                select = info[select][0]
                
    return "".join(answer)