import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

## Functions
def molarmass(frac, mass):
	return np.dot(frac,mass)

def mbartopa(p):
	return (p*100.)
	
def patombar(p):
	return (p/100.)
	
def least_squares(x, y):
	## Calculate linear least square (needed for dyn. viscosity)
	k = 1

	X = np.zeros((k+1,len(x)))

	for i in range(X.shape[0]):
		X[i,:] = x**i
		
	a = np.zeros(k)

	a = np.dot(np.linalg.inv(np.dot(X,X.T)),np.dot(X,y.T))
	print(np.sum((y - np.dot(X.T,a))**2))
	return a
	
def dynvisc_jupiter():		## WORK IN PROGRESS
	## C Palotai's code from EPIC model (epic_microphysics_funcs.c)
	## Ref - Jupiter: C. F. Hansen (1979)
	x1 = 0.864
	x2 = 0.136
	
	n = 64
	viscx = np.zeros(n+1)
	viscy = np.zeros(n+1)
	for i in range(n+1):
		temperature = 100. + i*(500.-100.)/n;

		n1  = 90.6*np.power(temperature/300.,.6658);
		n1  = n1/(1.+4./temperature);                # viscosity of H_2 in micropoise
		n2  = 191.6*np.power(temperature/300.,.7176);
		n2  = n2/(1.-11.4/temperature);              # viscosity of He in micropoise

		q1  = 32.3*(1.+4./temperature)*np.power(300./temperature,.1658);
		q2  = 21.5*(1.-11.4/temperature)*np.power(300./temperature,.2176);
		q3  = (np.sqrt(q1)+np.sqrt(q2))/2.;
		q3  = q3*q3;

		r1  = 1.+.7967*(x2/x1)*(q3/q1);
		r2  = 1.+.5634*(x1/x2)*(q3/q2);
		nu  = n1/r1 + n2/r2 ;                  # viscosity of atmosphere in micropoise
		nu  = nu*1.e-7;                        # viscosity of atmosphere in kg/m/s

		viscx[i] = temperature;
		viscy[i] = nu;
	return(least_squares(viscx,viscy))

## Planet arrays
planet_data = {}
planet_data["Jupiter"]  = {"g": 22.31, "xH2": 0.864, "xHe": 0.136,"datfile": "jupiter_data.csv"}
planet_data["Saturn"] = {"g":10.5, "xH2": 0.96,"xHe": 0.04, "datfile": "saturn_data.csv"} 
						
## Species arrays
spec_data = {}
spec_data["H2O"] = {"mass":18., "ant_liq": [11.079, -2261.1], "ant_ice": [12.61,-2681.18], "q": 0.001420, "rho_ice": 917.0, "rho_liq": 1000.}
spec_data["NH3"] = {"mass":17., "ant_liq": [10.201, -1248.0], "ant_ice": [11.90,-1588.00], "q": 0.000187, "rho_ice": 786.8, "rho_liq":  733.}

## T-P profiles
## tp_profile["planet"]["pressure mbar"][variable sought] 
'''tp_profile = {}
tp_profile["Jupiter"]={}
tp_profile["Jupiter"]["100"]={"T": 111.5, "Rho": 0.0244, "Dynvisc": 5.71E-06}
tp_profile["Jupiter"]["200"]={"T": 113.5, "Rho": 0.0478, "Dynvisc": 5.78E-06}
tp_profile["Jupiter"]["300"]={"T": 119.5, "Rho": 0.0682, "Dynvisc": 5.99E-06}
'''

## Setup the T-P profile, dyn. visc. data and atmospheric data

for planet in planet_data.keys():
	print("Setting up %s"%(planet))
	## Calculate molar mass of atmosphere from H2 and He concentrations
	Matmo = molarmass([planet_data[planet]["xH2"],planet_data[planet]["xHe"]],[2.,4.])
	Ratmo = 8314./Matmo
	planet_data[planet]["R"] = Ratmo
	
	## Open the T-P and viscosity file
	file = planet_data[planet]["datfile"]
	
	data = np.genfromtxt(file,skip_header = 1, delimiter = ",",missing_values = '', filling_values = None)
	
	## Clean up missing values
	data = data[np.isfinite(data[:,1]) & np.isfinite(data[:,2])]
	
	# Set up P with better resolution
	P = 10.**np.arange(np.log10(data[0,0]), np.log10(data[-1,0]), 0.01) ## P from 10 mbar to 10 bar
	
	## Interpolate T values so we can get better resolution
	## NOTE: interpolation is done in logP-space for better accuracy
	fT = interp1d(np.log10(data[:,0]), data[:,1], kind='cubic')
	planet_data[planet]["fT"] = fT
	
	## Do the same for dyn. viscosity
	fmu = interp1d(np.log10(data[:,0]), data[:,2], kind='cubic')
	planet_data[planet]["fmu"] = fmu

	## Save minimum and maximum values of P used in interpolation
	planet_data[planet]["Prange"] = np.asarray([data[0,0],data[-1,0]])
	
	''' Plot the data
	plt.figure()
	plt.plot(data[:,1],data[:,0],'ro')
	plt.plot(fT(np.log10(P)),P,'r-')
	plt.axes().set_yscale('log')
	plt.axes().set_ylim((np.max(P),np.min(P)))
	
	plt.figure()
	plt.plot(data[:,2],data[:,0],'ko')
	plt.plot(fmu(np.log10(P)),P,'k-')
	
	plt.axes().set_yscale('log')
	plt.axes().set_ylim((np.max(P),np.min(P)))
	'''
