try:
    import numpy as np
except ImportError:
    raise Exception('NumPy required for testing.')


def require_torch():
    try:
        import torch
        torch.manual_seed(0)
        return torch
    except ImportError:
        raise ImportError('PyTorch required to run tests')


def check_arr(a0, a1, tol=1e-5, show_diff=False):
    flag = np.allclose(a0, a1, rtol=tol, atol=tol)
    if show_diff and not flag:
        for v0, v1 in zip(a0, a1):
            print(f'---> v0={v0} : v1={v1} :: d={abs(v0 - v1)}')
    return flag
