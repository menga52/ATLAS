import numpy as np

class DummyRNG:
  def __init__(self, value=1):
    self.val = value
  
  def standard_normal(self, size):
    temp = np.zeros(size)
    temp.fill(self.val)
    return temp
  
  def random(self):
    return self.val