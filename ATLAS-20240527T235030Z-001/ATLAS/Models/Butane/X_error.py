import numpy as np
from scipy.linalg import subspace_angles

def X_error(chart_entry, parameter):
  X = chart_entry.X_int
  sigma = chart_entry.sigma
  U = chart_entry.sigma
  b = chart_entry.sigma
  H = chart_entry.sigma
  x1 = X[0]
  y1 = X[1]
  y3 = X[2]
  x4 = X[3]
  y4 = X[4]
  z4 = X[5]
  l = parameter.l
  theta = parameter.theta
  c1 = parameter.c1
  c2 = parameter.c2
  c3 = parameter.c3
  manifold_error = np.sqrt(x1+l*np.sin(theta))**2 + (y1-l*np.cos(theta))**2 + (y3-l)**2 \
      + (x4+l*np.cos(theta)-l)**2 + (np.sqrt(x4**2+z4**2)) - l*np.sin(theta)**2
  # note that this is not the 2-norm error. true invariant manifold is[l*sin(theta), -l*cos(theta), -l, x,
    # l*cos(theta)-l, z] with x^2+z^2=(l*sin(theta))^2
  sigma_true = parameter.sigma/(l*np.sin(theta))

  H_true = np.array([0,0,0,-z4*sigma_true,0,x4*sigma_true]) # transpose?
  sigma_abs_error = np.norm(np.norm(H_true,'fro')-abs(sigma),2)
  sigma_error = sigma_abs_error/np.norm(H_true,'fro') # norm(H_true, 'fro')
  angle = np.subspace_angles(H_true,H)[0]
  
  c_term =  z4*((c1*(x4**2+z4**2)+3*c3*x4**2)+2*c2*x4*np.sqrt(x4**2+z4**2))/(x4**2+z4**2)**(5/2)
  # c_term =  z4.*   (  (c1 .*(l*sin(theta))^2  + 3 * c3 *x4.^2 ) + 2* c2*x4.*(l*sin(theta)) )./(l*sin(theta)).^5

  b4 = -c_term * z4 - sigma_true**2/2*x4
  b6 =  c_term * x4 - sigma_true**2/2*z4

  # c_term = -c1*z4./sqrt(x4.^2 + z4.^2) - 2*c2 * x4.*z4./(x4.^2 + z4.^2) - 3*c3 * x4.^2.*z4./(x4.^2 + z4.^2).^(3/2);
  # b4    =   c_term * z4 - sigma_true^2/2*x4;
  # b6    =   - c_term * x4 - sigma_true^2/2*z4;
  
  b_true = np.array([0,0,0,b4,0,b6]) # transpose?
  b_abs_error = np.norm(b-b_true,2)
  b_error = b_abs_error/np.norm(b_true,2)

  return manifold_error, b_error, b_abs_error, sigma_error, sigma_abs_error, angle
