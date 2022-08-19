import numpy as np

def im2col_cython(x, field_height, field_width, padding, stride):
    N = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    HH = (H + 2 * padding - field_height) / stride + 1
    WW = (W + 2 * padding - field_width) / stride + 1
    p = padding
    x_padded = np.pad(x,((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    cols = np.zeros((C * field_height * field_width, N * HH * WW),dtype=x.dtype)
    im2col_cython_inner(cols, x_padded, N, C, H, W, HH, WW, field_height, field_width, padding, stride)
    return cols

def im2col_cython_inner(cols, x_padded, N, C, H, W, HH, WW,
                             field_height, field_width, padding, stride):
    for c in range(C):
        for yy in range(HH):
            for xx in range(WW):
                for ii in range(field_height):
                    for jj in range(field_width):
                        row = c * field_width * field_height + ii * field_height + jj
                        for i in range(N):
                            col = yy * WW * N + xx * N + i
                            cols[row, col] = x_padded[i, c, stride * yy + ii, stride * xx + jj]

def col2im_cython(cols, N, C, H, W,
                            field_height, field_width, padding, stride):
    x = np.empty((N, C, H, W), dtype=cols.dtype)
    HH = (H + 2 * padding - field_height) / stride + 1
    WW = (W + 2 * padding - field_width) / stride + 1
    x_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding), dtype=cols.dtype)

    # Moving the inner loop to a C-function with no bounds checking improves
    # performance quite a bit for col2im.
    col2im_cython_inner(cols, x_padded, N, C, H, W, HH, WW, field_height, field_width, padding, stride)
    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded

def col2im_cython_inner(cols, x_padded, N, C, H, W, HH, WW,
                             field_height, field_width, padding, stride):
    for c in range(C):
        for ii in range(field_height):
            for jj in range(field_width):
                row = c * field_width * field_height + ii * field_height + jj
                for yy in range(HH):
                    for xx in range(WW):
                        for i in range(N):
                            col = yy * WW * N + xx * N + i
                            x_padded[i, c, stride * yy + ii, stride * xx + jj] += cols[row, col]

def col2im_6d_cython_inner(cols, x_padded, N, C, H, W, HH, WW,
                                out_h, out_w, pad, stride):

    for n in range(N):
        for c in range(C):
            for hh in range(HH):
                for ww in range(WW):
                    for h in range(out_h):
                        for w in range(out_w):
                            x_padded[n, c, stride * h + hh, stride * w + ww] += cols[c, hh, ww, n, h, w]

def col2im_6d_cython(cols, N, C, H, W, HH, WW, pad, stride):
    x = np.empty((N, C, H, W), dtype=cols.dtype)
    out_h = int((H + 2 * pad - HH) / stride + 1)
    out_w = int((W + 2 * pad - WW) / stride + 1)
    x_padded = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=cols.dtype)
    col2im_6d_cython_inner(cols, x_padded, N, C, H, W, HH, WW, out_h, out_w, pad, stride)

    if pad > 0:
        return x_padded[:, :, pad:-pad, pad:-pad]
    return x_padded