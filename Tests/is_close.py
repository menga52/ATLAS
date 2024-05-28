import numbers
from collections.abc import Sequence
import numpy as np
from jax import numpy as jnp
from jaxlib.xla_extension import ArrayImpl
from oct2py.io import Cell

tolerance = 0.001
relative_tolerance = 0.01
array_types = (np.ndarray,ArrayImpl)
def is_number(a):
  return isinstance(a, numbers.Number)
def is_len_1(a):
  return isinstance(a, Sequence) and len(a)==1 or type(a)==np.ndarray and len(a)==1
def is_close_base_case(a,b):
  return abs(a-b)<=tolerance or abs(a-b)<=relative_tolerance*max(a,b)
def isempty(ls):
  if type(ls) is not list:
    return False
  if not ls:
    return True
  return False

def is_close(a,b,verbose=False):
  if type(a) == ArrayImpl:
    a = np.asarray(a)
  if type(b) == ArrayImpl:
    b = np.asarray(b)
  if id(a) == id(b):
    return True
  if type(a) is float and type(b) is float and jnp.isnan(a) and jnp.isnan(b):
    return True
  if isempty(a) and type(b) in (np.ndarray,ArrayImpl):
    return b.size == 0
  if type(a)==Cell:
    a = a.tolist()
    a = np.asarray(a)
  if type(b)==Cell:
    b = b.tolist()
    b = np.asarray(b)
  if verbose:
    if type(a) == np.ndarray:
      print("a.shape:",a.shape)
    if type(b) == np.ndarray:
      print("b.shape:",b.shape)
    print("is_close")
    print("a:",a)
    print("b:",b)
    print("type a:",str(type(a)))
    print("type b:",type(b))
  

  if is_number(a) and is_number(b):
    return is_close_base_case(a,b)
  if not (type(a) == np.ndarray and type(b) == np.ndarray):
    return False
  if a.shape != b.shape:
    return False
  if a.shape == ():
    return is_close_base_case(a,b)
  # if len(a.shape)==1:
  #   for i in range(len(a)):
  #     if not is_close_base_case(a,b):
  #       return False
    return True
  for i in range(len(a)):
    if not is_close(a[i],b[i]):
      return False
  return True

def is_close_forgiving(a,b,verbose=False):
  if verbose:
    print("a:",a,"; b:",b)
    print("type a:",type(a))
    print("type b:",type(b))
  if is_number(a) and is_number(b):
    return is_close_base_case(a,b)
  if is_number(a) and is_len_1(b):
    return is_close_forgiving(a,b[0],verbose)
  if is_number(b) and is_len_1(a):
    return is_close_forgiving(a[0],b,verbose)
  if is_len_1(a) and is_len_1(b):
    return is_close_forgiving(a[0],b[0],verbose)
  if (is_number(a) or is_len_1(a)) and type(b)==list:
    return False
  if (is_number(b) or is_len_1(b)) and type(a)==list:
    return False
  # at this point, both a and b are nontrivial lists
  if len(a) != len(b):
    return False
  # ... of the same length
  for i in range(len(a)):
    if not is_close_forgiving(a[i],b[i],verbose):
      return False
  return True