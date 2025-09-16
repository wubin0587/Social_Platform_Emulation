import numpy

# 默认后端为 NumPy (CPU)
_xp_backend = numpy
_is_gpu_available = False

try:
    import cupy
    if cupy.cuda.is_available():
        _xp_backend = cupy
        _is_gpu_available = True
except ImportError:
    pass # 静默处理，不打印任何信息

# --- 提供外部可以访问的变量和函数 ---

# 最终确定的计算后端
xp = _xp_backend

def is_gpu_available():
    """返回一个布尔值，表示GPU是否可用"""
    return _is_gpu_available

def to_cpu(array):
    """将数组从GPU移至CPU (如果它在GPU上)"""
    if _is_gpu_available and isinstance(array, cupy.ndarray):
        return array.get()
    return array

# to_gpu 函数可以保留，以备后用
def to_gpu(array):
    """将数组从CPU移至GPU (如果GPU可用)"""
    if _is_gpu_available:
        return cupy.asarray(array)
    return array