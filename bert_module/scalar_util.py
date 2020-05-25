# from https://github.com/allenai/allennlp/blob/c09833c3a2b2fe66f10ffd18761f90d0912c5ea2/allennlp/nn/util.py
import torch

def info_value_of_dtype(dtype: torch.dtype):
    """
    Returns the `finfo` or `iinfo` object of a given PyTorch data type. Does not allow torch.bool.
    """
    if dtype == torch.bool:
        raise TypeError("Does not support torch.bool")
    elif dtype.is_floating_point:
        return torch.finfo(dtype)
    else:
        return torch.iinfo(dtype)


def max_value_of_dtype(dtype: torch.dtype):
    """
    Returns the maximum value of a given PyTorch data type. Does not allow torch.bool.
    """
    print(dtype==torch.float16 ,"max")
    print(dtype==torch.half ,"max")
    a = info_value_of_dtype(dtype)
    print(a, "haha")
    print(a.max, "haha")
    return info_value_of_dtype(dtype).max


def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half or dtype == torch.float16 :
        print("haha3")
        #return 1e-4
        return torch.tensor(1e-4,dtype=torch.half)
    else:
        print("haha4")
        raise TypeError("Does not support dtype " + str(dtype))
