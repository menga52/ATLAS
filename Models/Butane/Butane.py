from ATLAS.Models.Model import *
from ATLAS.Models.ButaneParameters import *
from math import *
from numpy import np

class Butane(Model): 
  def __init__(self):
    self.init()
	
  def createParameters(self):
    self.parameters = ButaneParameters()
		
  def get_diffusion(self,X,parameter):
    sigma = parameter["sigma"]
    # diffusion = sigma + lambda**0.5*(y-tau*sin(omega*x))**2
    return sigma

  def get_drift(self,X,parameter):
    x1 = X[0]
    y1 = X[1]
    y3 = X[2]
    x4 = X[3]
    y4 = X[4]
    z4 = X[5]
    drift = np.zeros((6,1))
    l = parameter["l"]
    k2 = parameter["k2"]
    k3 = parameter["k3"]
    theta = parameter["theta"]
    c1 = parameter["c1"]
    c2 = parameter["c2"]
    c3 = parameter["c3"]

    b1 = 1 - l/sqrt(x1**2+y1**2)
    b3 = 1 - l/(x4**2+z4**2+(y3-y4)**2)

    theta1 = theta - acos(y1*np.sign(y3)/sqrt(x1**2+y1**2))
    theta2 = theta - acos((y3-y4)*np.sign(y3)/sqrt(x4**2+z4**2-(y3-y4)**2))

    # c_term = - z4*(np.sign(x1)*(c1*(x4**2+z4**2)+3*c3*x4**2)+2*c2*x4*(x4.^2 + z4.^2)**0.5)/(x4**2+z4**2)**(5/2)
    c_term = z4*((c1*(x4**2+z4**2)+3*c3*x4**2)+2*c2*x4*sqrt(x4**2+z4**2))/(x4**2+z4**2)**2.5
    drift[1] = k2*x1*b1-k3*y1*np.sign(x1*y3)/(x1**2+y1**2)*theta1
    drift[2] = k2*y1*b1+k3*abs(x1)*np.sign(y3)/(x1**2+y1**2)*theta1
    drift[3] = k2*((y3-y4)*b3+(-l+abs(y3))*np.sign(y3))+k3*(x4**2+z4**2)**0.5*np.sign(y3)/(x4**2+z4**2+(y3-y4)**2)*theta2
    drift[4] = k2*x4*b3-k3*x4*(y3-y4)/sqrt(x4**2+z4**2)*np.sign(y3)/(x4**2+z4**2+(y3-y4)**2)*theta2+z4*c_term                
    drift[5] = -k2*(y3-y4)*b3-k3*sqrt(x4**2+z4**2)*np.sign(y3)/(x4**2+z4**2+(y3-y4)**2)*theta2
    drift[6] = k2*z4*b3-k3*z4*(y3-y4)/sqrt(x4**2+z4**2)*np.sign(y3)/(x4**2+z4**2+(y3-y4)**2)*theta2-x4*c_term
    return -drift
	
  def potential(self,x1,y1,y3,x4,y4,z4,parameter):
    l = parameter["l"]
    k2 = parameter["k2"]
    k3 = parameter["k3"]
    theta = parameter["theta"]
    c1 = parameter["c1"]
    c2 = parameter["c2"]
    c3 = parameter["c3"]

    b1 = sqrt(x1**2+y1**2)-l
    b2 = abs(y3)-l
    b3 = sqrt(x4**2+z4**2+(y3-y4)**2)-l

    theta1 = theta - acos(y1*np.sign(y3)/sqrt(x1**2+y1**2))
    theta2 = theta - acos((y3-y4)*np.sign(y3)/sqrt(x4**2+ z4**2+(y3-y4)**2))
    c1_term = c1*x4*np.sign(x1)/sqrt(x4**2+z4**2)
    c2_term = c2*x4**2/(x4**2+z4**2)
    c3_term = c3*x4**3*np.sign(x1)/(x4**2+z4**2)**(3/2)
    # return V
    return c1_term + c2_term + c3_term + 1/2*k2*(b1**2+b2**2+b3**2)+1/2*k3*(theta1**2+theta2**2)
	
	