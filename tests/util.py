def check_arr(a0, a1, tol=1e-5, show_diff=False):
    if len(a0) != len(a1):
        return False
    abs_flags = [abs(v0 - v1) <= tol for v0, v1 in zip(a0, a1)]
    sign_flags = [sign(v0, tol=tol) == sign(v1, tol=tol) for v0, v1 in zip(a0, a1)]
    flag = all([af and sf for af, sf in zip(abs_flags, sign_flags)])
    if show_diff and not flag:
        for i in range(len(a0)):
            if not abs_flags[i] or not sign_flags[i]:
                a0v = a0[i]
                a1v = a1[i]
                print(f'---> v0={a0v} : v1={a1v} :: d={abs(a0v - a1v)} : s0={sign(a0v)} : s1={sign(a1v)}')
    return flag


def sign(value, tol=1e-5):
    value = 0.0 if abs(value) <= tol else value
    value = 0.0 if str(value) == '-0.0' else value
    return 1 if value >= 0.0 else -1
