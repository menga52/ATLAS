import numpy as np
from sklearn.neighbors import NearestNeighbors
pi = 3.14159

def meshgrid(xi,yi):
  # translation of matlab function
  xlen = len(xi)
  ylen = len(yi)
  A = np.zeros((ylen,xlen))
  B = np.zeros((ylen,xlen))
  for i in range(ylen):
    for j in range(xlen):
      B[i][j] = i
      A[i][j] = j
  return A,B

def set_well(chart, TranM, chart_angle):
  # store chart_angle
  K = len(chart)
  chart_angle = np.zeros((d, K))
  for i in range(K):
    X_int = np.transpose(chart[i].X_int)
    theta_X_int = np.arctan2(np.sqrt(X_int[0]**2+X_int[1]**2), X_int[2])
    phi_X_int = np.arctan2(X_int(2), X_int(1)) % (2*pi)
    chart_angle[:,i] = np.asarray([[phi_X_int], [theta_X_int]])
  V,_ = np.linalg.eig(np.transpose(TranM))
  eigenvalues_absolute_vals = np.abs(V)
  # sort the eigenvalues, output indices sorted accordingly
  sorted_indices = np.argsort(eigenvalues_absolute_vals)[::-1]
  largest_indices = sorted_indices[:6]
  V = [V[i] for i in largest_indices]
  # _,VD = eig(np.transpose(TranM)) # commented because unused

  if V[0] < 0:
    V = [-v for v in V]
    V[:,0] = -V[:,0]
  # if V(1,1) < 0:
  #   V(:,1)=-V(:,1);
  # end

  angle = np.asarray([[pi/6, 5/6*pi]])
  neighbors_obj = NearestNeighbors(n_neighbors=1,algorithm='auto').fit(np.transpose(chart_angle))
  distances, indices = neighbors_obj.kneighbors(angle)
  index_of_nearest = indices[0][0]
  nearest = chart_angle[index_of_nearest]   

  if V[index_of_nearest,1] < 0:
      V[:,1] = -V[:,2]
  z2   = np.transpose(V[:,2])

  # periodic extension
  for i in range(K):
    if chart_angle[0,i] > 2*pi-0.2:
      temp_phi = chart_angle[0,i]-2*pi
    if chart_angle(1,i) < 0.2:
      temp_phi = chart_angle(1,i)+2*pi;
    temp = np.asarray([[temp_phi],[chart_angle[1,i]]])
    chart_angle = np.hstack((chart_angle, temp))
    z2  = np.hstack((z2, z2(i)))
    x   = chart_angle[0,:]
    y   = chart_angle[1,:]
    xi  = range(0,2*pi,0.005)
    yi  = range(min(y), max(y), 0.002)
    """
    TODO: Plot data
    from here forward is not translated.
    Could be run without translating using oct2py.octave
    """
    xq,yq   = meshgrid(xi,yi);
    zi        = griddata(x,y,z2,xq,yq,'natural');
    [c,h]     = contourf(xi,yi,zi,[-0.06:0.02:0.06],'ShowText','on');
    s         = contourdata(c);

    for i = 1:length(s)
      if s(i).level==0
          phi_zero  = s(i).xdata;
          theta_zero = s(i).ydata;
          break
      end
    end

    well1 = find(V(:,2)>  0.050); %define cyan state 
    well2 = find(V(:,2)< -0.050); %define red  state 

    index1                = find(ismember(nearest_store, well1));
    L_IC                  = length(index1);
    index1                = index1(round(linspace(1, L_IC, N_IC)));
    index2                = find(ismember(nearest_store, well2));
    L_IC                  = length(index2);
    index2                = index2(round(linspace(1, L_IC, N_IC)));
    X_int_store1          = X(index1,:);
    nearest_store1        = nearest_store(index1);
    X_int_store2          = X(index2,:);
    nearest_store2        = nearest_store(index2);
 

    well_threshold1 =find(V(:,2)>  0.020);
    well_threshold2 =find(V(:,2)< -0.020);


    chart_angle       = zeros(d, K);
    for i = 1:K
      X_int                 = chart{i}.X_int';
      theta_X_int           = atan2( sqrt( X_int(1)^2 + X_int(2)^2 ), X_int(3) );
      phi_X_int             = mod(atan2(X_int(2), X_int(1)),2*pi);
      chart_angle(:,i)      = [phi_X_int;  theta_X_int ];