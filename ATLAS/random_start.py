from networkx import number_connected_components
import numpy as np

seed = 1
default_rng = np.random.default_rng(seed)

# randomly pick an element in the smallest connected component in the graph
"""
chart - list of ChartEntry objects
connected_component_list - list of components.
    component is a list of indices corresponding to entries in chart
chart_sim_parameter - dictionary
rng - a random number generator
"""
def random_start(chart, connected_component_list, chart_sim_parameter, rng=-1):
  if rng == -1:
    rng = default_rng
  num_connected_components = len(connected_component_list)
  min_size = np.infty # infinity
  index_of_min = 0

  # find the smallest connected component
  for connected_component_index in range(num_connected_components):
    component_size = len(connected_component_list[connected_component_index])
    if component_size < min_size:
      index_of_min = connected_component_index
      min_size = component_size
  
  min_size_component = connected_component_list[index_of_min]
  nearest = min_size_component[int(rng.random()*len(min_size_component))]
  chart_sim_parameter["nearest"] = nearest
  # chart_sim_parameter.connectivity = connectivity;
  # ^ appears in original code, but connectivity isn't mutated
  chart_sim_parameter["X_int"] = chart[nearest].X_int
  return # no return necessary because the function mutates chart_sim_parameter
