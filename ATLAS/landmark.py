import numpy as np
import networkx as nx
from ATLAS.ATLAS.check_x_in_chart import check_x_in_chart
from ATLAS.ATLAS.distance_chart_to_x import distance_chart_to_x

"""
chart_ori - original chart, list of ChartEntry objects
t0 - number, the relaxation time. 0.1/
chi_p - test statistic for 95% confidence interval. 5.991
threshold - This is R_max determined by the diameter of the 
      slow manifold when setting the value of R or the
      maximum possible layer [1,0.001]/
connectivity_threshold - 0.3/

returns amended chart
"""
def landmark(chart_ori, t0, chi_p, threshold, connectivity_threshold, verbose=False):
  # This function finds the landmark among the sample points and builds
  # the connectivity matrix for these landmarks.
  # Distance is diffusion distance with unit of sqrt(t0)
  K = len(chart_ori)
  indicator = np.ones(K)


  check = check_x_in_chart
  # if dist(i,j)<1-sqrt(1/2) and dist(j,i)<1-sqrt(1/2), remove j
  for i in range(K):
    for j in range(i+1,K):
      if indicator[i] and indicator[j] and check(chart_ori[i].X_int, chart_ori[j], t0, 0.3, chi_p, threshold) \
          and check(chart_ori[j].X_int, chart_ori[i], t0, 0.3, chi_p, threshold):
        indicator[j] = 0
  
  chart = [chart_ori[k] for k in range(K) if indicator[k]>0]
  # i and j are connected if dist(i,j)<connectivity_threshold or dist(j,i)<connectivity_threshold
  chart_length = len(chart)
  connectivity = np.zeros((chart_length, chart_length))

  for i in range(chart_length):
    connectivity[i][i] = 1
    for j in range(i+1, chart_length):
      if check(chart[i].X_int, chart[j], t0, connectivity_threshold, chi_p, threshold) \
          and check(chart[j].X_int, chart[i], t0, connectivity_threshold, chi_p, threshold):
        dist1 = distance_chart_to_x(chart[i].X_int, chart[j], t0, chi_p, threshold)
        dist2 = distance_chart_to_x(chart[j].X_int, chart[i], t0, chi_p, threshold)
        connectivity[i][j] = max(dist1, dist2)
  
  P = nx.Graph()
  for i in range(len(connectivity)):
    for j in range(len(connectivity[i])):
      if connectivity[i][j] != 0:
        P.add_edge(i, j, weight=connectivity[i][j]) 
    
  # replaces P = graph(connectivity, 'upper')
  connected_component_generator_object = nx.connected_components(P)
  connected_component_list = list(connected_component_generator_object)
  num_connected_components = len(connected_component_list)
  if num_connected_components == 1 and verbose:
    print("All landmarks are connected.")
  elif verbose:
    print("Number of Regions =", num_connected_components)
  
  connectivity = connectivity + np.transpose(connectivity)
  K = len(chart)

  for i in range(K):
    connectivity[i][i] = 1
    # denote each node is connected with itself. Distance is set as 1 for convention
  
  return chart, connectivity, P, num_connected_components, connected_component_list



