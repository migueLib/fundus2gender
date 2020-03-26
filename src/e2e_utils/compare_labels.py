def compare_with_true(data, true):
    """"Uses a series for the true"""
    return data.apply(lambda x: x == true, axis=0)
