import numpy as np
import networkx as nx
from ATLAS.ATLAS.check_x_in_chart import check_x_in_chart
from ATLAS.ATLAS.distance_chart_to_x import distance_chart_to_x

"""
chart - original chart. list of ChartEntry
chart_add - additions to the chart. Unknown type
t0 - 
chi_p - 
threshold - 
connectivity_threshold - 

returns: 
"""
def landmark_add(chart, connectivity, chart_add, t0, chi_p, threshold, connectivity_threshold):
  K = len(chart)
  connectivity_new = np.zeros((K+1,K+1))
  for i in range(K):
    for j in range(K):
      connectivity_new[i][j] = connectivity[i][j]
  
  chart = np.append(chart, chart_add, axis=1)
  
  for i in range(K+1):
    if check_x_in_chart(chart[i].X_int, chart[K], t0, connectivity_threshold, chi_p, threshold) \
        or check_x_in_chart(chart[K].X_int, chart[i], t0, connectivity_threshold, chi_p, threshold):
      dist1 = distance_chart_to_x(chart[i].X_int, chart[K], t0, chi_p, threshold)
      dist2 = distance_chart_to_x(chart[K].X_int, chart[i], t0, chi_p, threshold)
      connectivity_new[K][i] = max(dist1, dist2)
  
  connectivity_new[K][K] = 1
  connectivity_new[:][K] = np.transpose(connectivity_new)[:][K]
  connectivity = connectivity_new
  P = nx.Graph() # not sure what to do with 'upper'
  for i in range(len(connectivity)):
    for j in range(len(connectivity[i])):
      if connectivity[i][j] != 0:
        P.add_edge(i, j, weight=connectivity[i][j]) 
  
  return chart, connectivity, P
