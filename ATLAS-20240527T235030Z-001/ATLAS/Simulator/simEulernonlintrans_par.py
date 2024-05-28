# should be able to simply add par parameter to simEulernonlintrans.py function
from ATLAS.Simulator.simEulernonlintrans import simEulernonlintrans
def simEulernonlintrans_par(RHS_parameter,Sim_parameter,\
  verbose=False,parallel=False,rng=-1):
  return simEulernonlintrans(RHS_parameter,Sim_parameter,\
    verbose=verbose,parallel=parallel,rng=rng)