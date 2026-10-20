
import numpy as np

def conv1d(x: np.ndarray, kernel: np.ndarray, mode: str) -> np.ndarray:
    n, k = len(x), len(kernel)

    if mode == "valid":
        padded_x = x
    elif mode == "same":
        p = k // 2
        padded_x = np.pad(x, (p, p), mode="constant")
    else:
        raise ValueError("mode must be 'valid' or 'same'")

    ans = np.zeros(len(padded_x) - k + 1)

    for i in range(len(ans)):
        ans[i] = np.sum(padded_x[i:i+k] * kernel)

    return ans
