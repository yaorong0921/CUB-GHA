def compute_window_nums(ratios, stride, input_size):
    window_nums = []
    for _, ratio in enumerate(ratios):
        window_nums.append(int((input_size - ratio[0])/stride + 1) * int((input_size - ratio[1])/stride + 1))
    return window_nums