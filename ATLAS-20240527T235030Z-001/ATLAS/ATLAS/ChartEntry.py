class ChartEntry:
  def __init__(self):
    self.LearningTime = None
    self.U = None # D by d matrix
    self.V = None # D by el matrix. See Learning_Slow_Manifold
    self.sigma = None # 1 by d row
    self.sigma_fast = None # D by 1 column??? not sure
    self.Lambda = None # d by d matrix
    self.b = None # temp_mean/norm(t_span), D by 1 column
    self.X_int = None # 1 by D row
    self.WU = None # matrix, D by d
  
  def clone(self):
    other = ChartEntry()
    other.LearningTime = self.LearningTime
    copy_list = ["U","V","sigma","sigma_fast","Lambda","b","X_int","WU"]
    for item in copy_list:
      to_copy = getattr(self,item)
      if type(to_copy) is not type(None):
        setattr(other,item,to_copy.copy())
    return other
    # other.U = self.U.copy() # unnecessary once switched to jnp
    # other.V = self.V.copy()
    # other.sigma = self.sigma.copy()
    # other.sigma_fast = self.sigma_fast.copy()
    # other.Lambda = self.Lambda.copy()
    # other.b = self.b.copy()
    # other.X_int = self.X_int.copy()
    # other.WU = self.WU.copy()
    # return other
  def equals(self,other):
    if type(other)!=type(self):
      return False
    return self.LearningTime==other.LearningTime\
      and self.U==other.U and self.V==other.V\
      and self.sigma==other.sigma\
      and self.sigma_fast==other.sigma_fast\
      and self.Lambda==other.Lambda and self.b==other.b\
      and self.X_int==other.X_Int and self.WU==other.WU
  copy = clone # alias the class method