import numpy as np

def get_diffusion(X,epsilon,c2,c4,c6,verbose=False):
  x  = X[0][0]
  y  = X[1][0]
  z  = X[2][0]

  r           = np.sqrt(x**2+y**2+z**2)
  r_sq        = r**2
  rsin        = np.sqrt(x**2 + y**2);
  rsin_sq     = rsin**2 
  sqrt_eps    = np.sqrt(epsilon);

  d1          = c2/(sqrt_eps*r_sq)
  d2          = c4/r_sq
  d3          = c6/r

  # diffusion = [ c2 ./(sqrt_eps.* r ).* x./r,  c4 .* rsin./r_sq.* x.*z./rsin, -y.*c6./ r;
  #               c2 ./(sqrt_eps.* r ).* y./r,  c4 .* rsin./r_sq.* y.*z./rsin,  x.*c6./ r;
  #               c2 ./(sqrt_eps.* r ).* z./r, -c4 .* rsin./r_sq.* rsin,        0 ];

  diffusion = [ [d1*x, d2*x*z, -y*d3],
                [d1*y, d2* y*z, x*d3],
                [d1*z, -c4*rsin_sq/r_sq, 0]]
  return np.asarray(diffusion)


  