from ATLAS.Models import Butane
from ATLAS.Models import Halfmoon
from ATLAS.Models import Peanut
import numpy as np

def run(arg=0, verbose=False, seed=1234567):
  if seed != -1:
    np.random.seed(seed)
  model = None
  if arg==0:
    print("Run with 1 for Butane, 2 for Halfmoon, 3 for Peanut")
    return
  if arg==1:
    print("4")
    model = Butane.Butane(verbose)
  elif arg==2:
    model = Halfmoon(verbose)
  elif arg==3:
    model = Peanut(verbose)
  else:
    print("invalid argument: ",arg)
    print("Run with 1 for Butane, 2 for Halfmoon, 3 for Peanut")
    return

  model.initial_learning()
  model.exploration_learning()
  model.butane_simulation_learning()
  model.relearning()
  model.MSM_learning()
	