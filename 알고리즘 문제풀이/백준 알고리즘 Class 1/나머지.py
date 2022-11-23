import input_setting

answer = set()
while True:
    try:
        answer.add(int(input())%42)
    except:
        print(len(answer))
        break