import input_setting

start = input()
end = input()

flag = False
while len(end) >= len(start):
    if end==start:
        flag = True
        break
    
    end = end[:-1] if end[-1] == "A" else end[:-1][::-1]
        

print(1) if flag else print(0)
    


