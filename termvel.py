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

	return a
	
def dynvisc(planet):
	## C Palotai's code from EPIC model (epic_microphysics_funcs.c)
	## Ref - Jupiter: C. F. Hansen (1979)
	## planet_data is global dictionary of planet parameters
	## planet is string with the name of the planet
	
	global planet_data
	## Fetch H2 and He concentrations
	x1 = planet_data[planet]["xH2"]
	x2 = planet_data[planet]["xHe"]
	
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

def vT_calc(planet, species, phase, p, D):
	## Terminal velocity calculation
	## From Palotai, Cs. and Dowling. T. E. 2008, Icarus 194, 303-326
	## Variables: 
	##  planet [string] - name of the planet
	## 	species [string] - chemical formula of species (as in spec_data array)
	## 	phase [string] - either rain, snow or ice
	##	p [float] - Pressure in mbar
	## 	D [float] - diameter of particle
	
	global planet_data, spec_data
	
	g = planet_data[planet]["g"]
	Ratmo = planet_data[planet]["R"]
	T = planet_data[planet]["fT"](np.log10(p))
	rho_air = p/(Ratmo*T)
	
	dynvisc_corr = planet_data[planet]["mucorr"]
	dynvisc = dynvisc_corr[0] + dynvisc_corr[1]*T
		
	if(phase == "rain"):
		rho_liq = spec_data[species]["rho_liq"]
		W = np.log10((4. * (D**3.)*rho_air*g*(rho_liq - rho_air))/(3.*dynvisc**2.))
		Re = -1.81391 + 1.34671*W - 0.12427*W**2. + 0.006344*W**3.
		
		return (dynvisc*Re)/(D*rho_air)
		
	elif(phase in ["ice","snow"]):
		A_Ae = spec_data[species]["A_Ae_"+phase]
		X = ((8./np.pi)*(A_Ae*0.333*(D**2.4)*rho_air*g)/(dynvisc**2))
		Re = 8.5*(np.sqrt(1. + 0.1519*np.sqrt(X)) - 1.)**2.

		return ((1./D)*(dynvisc*Re)/(4.*rho_air))
	
global planet_data, spec_data
## Planet arrays
planet_data = {}
planet_data["Jupiter"]  = {"g": 22.31, "xH2": 0.864, "xHe": 0.136,"datfile": "jupiter_data.csv", "Pref": 1000.}
#planet_data["Saturn"] = {"g":10.5, "xH2": 0.96,"xHe": 0.04, "datfile": "saturn_data.csv", "Pref": 1000.} 		

## Species arrays
spec_data = {}
spec_data["H2O"] = {"mass":18., "ant_liq": [11.079, -2261.1], "ant_ice": [12.61,-2681.18], "q": 0.001420, "rho_ice": 917.0, "rho_liq": 1000., "A_Ae_snow": 1.05, "A_Ae_ice": 1.}
spec_data["NH3"] = {"mass":17., "ant_liq": [10.201, -1248.0], "ant_ice": [11.90,-1588.00], "q": 0.000187, "rho_ice": 786.8, "rho_liq":  733., "A_Ae_snow": 1.05, "A_Ae_ice": 1.}

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
	
	## Find correlation between dynamic viscosity and temperature
	dynvisc_corr = dynvisc(planet)
	mu = dynvisc_corr[0] + dynvisc_corr[1]*fT(np.log10(P))
	
	## Store correlation parameters
	planet_data[planet]["mucorr"] = np.asarray(dynvisc_corr)

	## Save minimum and maximum values of P used in interpolation
	planet_data[planet]["Prange"] = np.asarray([data[0,0],data[-1,0]])
	
	'''
	#Plot the data
	plt.figure()
	plt.plot(data[:,1],data[:,0],'ro')
	plt.plot(fT(np.log10(P)),P,'r-')
	plt.axes().set_yscale('log')
	plt.axes().set_ylim((np.max(P),np.min(P)))
	
	plt.figure()
	plt.plot(mu,P,'k--')
	plt.axes().set_yscale('log')
	plt.axes().set_ylim((np.max(P),np.min(P)))'''
	
## Setup pressure intervals for vT calculation
P = {}
P["H2O"] = np.asarray([4750., 4500., 4000., 3500.,3000., 2500., 2000., 1000.])
P["NH3"] = np.asarray([1000., 850., 700., 600., 500., 400.])

## Set up terminal velocity dictionary
vT = {}

## Run terminal velocity for each species for each planet
for species in spec_data.keys():
	vT[species] = {}
	
	for planet in planet_data.keys():
		vT[species][planet] = {}
		
		for phase in ["rain","ice","snow"]:
			## setup particle sizes
			## different sizes for each phase
			
			if(phase == "rain"):
				D = np.linspace(200.e-6, 5.e-3, 1000.)
			elif(phase == "ice"):
				D = np.linspace(1.e-6, 500.e-6, 1000.)
			elif(phase == "snow"):
				D = np.linspace(0.5e-3, 5.e-3, 1000.)
				
			vT[species][planet][phase] = {}
			vT[species][planet][phase]["D"] = D
			for p in P[species]:
				vT[species][planet][phase][p] = np.zeros(len(D))
				
				for i, d in enumerate(D):
					vT_d = vT_calc(planet, species, phase, p, d)
					vT[species][planet][phase][p][i] = vT_d