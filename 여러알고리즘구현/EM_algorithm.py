if __name__ == "__main__":
    observation = [[5, 5], [9, 1], [8, 2], [4, 6], [7, 3]]  # Number of head and tail

    param_a = 0.6
    param_b = 0.5

    for _ in range(10):
        responsibility = [[0, 0] for _ in range(5)]  # [a, b] * 전체 차수

        # E-step
        for i, (h_num, t_num) in enumerate(observation):
            a = param_a**h_num * (1 - param_a)**t_num
            b = param_b**h_num * (1 - param_b)**t_num

            responsibility[i][0] = a / (a + b)
            responsibility[i][1] = b / (a + b)

        # M-step
        a_sum = [0, 0]  # head, tail
        b_sum = [0, 0]
        for i, (h_num, t_num) in enumerate(observation):
            a_sum[0] += responsibility[i][0] * h_num
            a_sum[1] += (1 - responsibility[i][0]) * t_num

            b_sum[0] += responsibility[i][1] * h_num
            b_sum[1] += (1 - responsibility[i][1]) * t_num

        param_a = round(a_sum[0] / sum(a_sum), 2)
        param_b = round(b_sum[0] / sum(b_sum), 2)

        print(param_a, param_b)