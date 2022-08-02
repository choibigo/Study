import input_setting

esm = list(map(int, input().split()))

if esm[0] == esm[1] == esm[2]:
    print(esm[0])

else:
    day = 16

    while True:

        e_mod = (day-esm[0]) % 15
        s_mod = (day-esm[1]) % 28
        m_mod = (day-esm[2]) % 19

        if e_mod == 0 and s_mod == 0 and m_mod == 0:
            print(day)
            break

        day+=1
