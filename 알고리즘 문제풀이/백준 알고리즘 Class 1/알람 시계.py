hour, miniute = map(int, input().split())
hour, miniute = divmod(hour*60+miniute-45, 60)

if hour <0:
    hour += 24

print(hour, miniute)