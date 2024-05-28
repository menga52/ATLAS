import numpy as np
from jax import numpy as jnp
from jaxlib.xla_extension import ArrayImpl
from numbers import Number
from ATLAS.ATLAS.ChartEntry import ChartEntry

def python_array_to_cell(arr):
  ls = arr.tolist()
  return "{"+" ".join([python_to_matlab(jnp.array(item)) for item in ls])+"}"

def python_list_to_cell(ls):
  return "{"+" ".join([python_to_matlab(item) for item in ls])+"}"

def python_number_list_to_matrix(ls):
  return "["+" ".join([str(a) for a in ls]) + "]"
def is_number_list(obj):
  if not type(obj) is list: return False
  for a in obj:
    if not isinstance(a,Number): return False
  return True

object_list = [ChartEntry]
def python_to_matlab(obj):
  if isinstance(obj,Number):
    ret = str(obj)
  elif is_number_list(obj):
    ret = python_number_list_to_matrix(obj)
  elif type(obj) is np.ndarray:
    if len(obj.shape)>2:
      ret = python_array_to_cell(obj)
    else: ret = array_to_matrix(obj)
  elif type(obj) is ArrayImpl:
    if len(obj.shape)>2:
      ret = python_array_to_cell(obj)
    else: ret = array_to_matrix(obj)
  elif type(obj) is dict:
    ret = dict_to_struct(obj)
  elif type(obj) in object_list:
    ret = dict_to_struct(obj.__dict__)
  elif type(obj) is list:
    return python_list_to_cell(obj)
  else:
    print(type(obj))
    print(obj)
  if ret == None:
    print(type(obj))
    print(obj)
  return ret

def array_to_matrix(arr):
  x=str(arr.tolist()).replace('[', '').replace(']', ';').replace(',', '')[:-1]
  return "["+x+"]"

def struct_to_dict(matlab_matrix_str):
  rows = matlab_matrix_str.strip('[]').split(';')
  python_list = []
  for row in rows:
    elements = row.split()
    python_row = [float(element) for element in elements]
    python_list.append(python_row)
  return python_list


def dict_to_struct(d,verbose=False):
  parts = []
  for key, value in d.items():
    # Convert the key to a valid Octave field name
    key_str = str(key)
    # Convert the value to an Octave-compatible string representation
    if isinstance(value, (int, float)):
      value_str = str(value)
    elif isinstance(value, str):
      value_str = f"'{value}'"
    elif isinstance(value, (list, tuple)):
      value_str = f"[{' '.join(map(str, value))}]"
    elif isinstance(value, dict):
      value_str = dict_to_struct(value)
    elif isinstance(value, np.ndarray):
      value_str = array_to_matrix(value)
    elif isinstance(value, ArrayImpl):
      value_str = array_to_matrix(value)
    else:
      continue
      # raise ValueError(f"Unsupported value type: {type(value)} for key: {key}")
    parts.append(f'"{key_str}", {value_str}')
  return f"struct({', '.join(parts)})"