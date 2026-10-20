import numpy as np


def conv1d(x: np.ndarray, kernel: np.ndarray, mode: str) -> np.ndarray:
    """
    1D cross-correlation ("convolution" in the deep-learning sense: the
    kernel is NOT flipped) with "valid" or "same" zero padding.

    x: shape (n,). kernel: shape (k,), k odd. mode: "valid" or "same".

    "valid": no padding, output length n - k + 1.
    "same": zero-pad x by k // 2 on each side first, output length n.

    Slide the kernel across the (possibly padded) signal without flipping
    it, taking sum(window * kernel) at each position.
    """
    # TODO: build the padded (or unpadded) signal first, then one sliding-window loop.
    # pass
    n,k = len(x),len(kernel)
    if mode == 'valid':
        updated_kernel = np.concatenate((kernel,[0]*(n-k)))
        ans = np.zeros(n-k+1)
        for i in range(n-k+1):
            # print(np.roll(updated_kernel,i))
            ans[i] = (x*np.roll(updated_kernel,i)).sum()
    else:
        x = np.concatenate(([0]*(k//2),x,[0]*(k//2)))
        n = len(x)
        updated_kernel = np.concatenate((kernel,[0]*(n-k)))
        ans = np.zeros(n-k+1)
        for i in range(n-k+1):
            # print(np.roll(updated_kernel,i))
            ans[i] = (x*np.roll(updated_kernel,i)).sum()
        

    return ans
