import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

## Functions
def molarmass(frac, mass):
	return np.dot(frac,mass)

def mbartopa(p):
	return (p*100.)
	
def patombar(p):
	return (p/100.)

## Planet arrays
planet_data = {}
planet_data["Jupiter"]  = {"g": 22.31, "R": 8314./molarmass(np.asarray([0.9,0.1]), np.asarray([2.,4.]))}
planet_data["Saturn"] = {"g":10.5} 
						
## Species arrays
spec_data = {}
spec_data["H2O"] = {"mass":18., "ant_liq": [11.079, -2261.1], "ant_ice": [12.61,-2681.18], "q": 0.00142, "rho_ice": 917., "rho_liq": 1000.}
spec_data["NH3"] = {"mass":17., "ant_liq": [10.201, -1248.], "ant_ice": [11.9,-1588.], "q": 0.000187, "rho_ice": 786.8,"rho_liq": 733.}

## T-P profiles
## tp_profile["planet"]["pressure mbar"][variable sought] 
tp_profile = {}
tp_profile["Jupiter"]={}
tp_profile["Jupiter"]["100"]={"T": 111.5, "Rho": 0.0244, "Dynvisc": 5.71E-06}
tp_profile["Jupiter"]["200"]={"T": 113.5, "Rho": 0.0478, "Dynvisc": 5.78E-06}
tp_profile["Jupiter"]["300"]={"T": 119.5, "Rho": 0.0682, "Dynvisc": 5.99E-06}