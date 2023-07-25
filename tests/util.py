def check_arr(a0, a1, tol=1e-5):
    if len(a0) != len(a1):
        return False
    flag = all([abs(v0 - v1) <= tol and sign(v0) == sign(v1) for v0, v1 in zip(a0, a1)])
    return flag


def sign(value):
    value = 0.0 if str(value) == '-0.0' else value
    return 1 if value >= 0.0 else -1
