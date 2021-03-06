import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

## Functions
def molarmass(frac, mass):
	return np.dot(frac,mass)

def mbartopa(p):
	return (p*100.)
	
def patombar(p):
	return (p/100.)
	
def least_squares(x, y, k = 1):
	## Calculate linear least square (needed for dyn. viscosity)
	X = np.zeros((k+1,len(x)))
	for i in range(X.shape[0]):
		X[i,:] = x**i
	a = np.zeros(k)
	a = np.dot(np.linalg.inv(np.dot(X,X.T)),np.dot(X,y.T))

	return a
	
def newtons(func, val, x0, thresh = 0.1, h = 0.0001):
	## Root finding algorithm based on first order derivatives (Newton-Rhapsons' Method)
	diff = 100.
	while diff > thresh:
		grad = (func(x0 + h) - func(x0))/h
		x0 = x0 - (func(x0)-val)/grad
		diff = func(x0) - val
	return x0
	
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

def vT_calc(planet, species, phase, p, D, planet_data, spec_data):
	## Terminal velocity calculation
	## From Palotai, Cs. and Dowling. T. E. 2008, Icarus 194, 303-326
	## Variables: 
	##  planet [string] - name of the planet
	## 	species [string] - chemical formula of species (as in spec_data array)
	## 	phase [string] - either rain, snow or ice
	##	p [float] - Pressure in mbar
	## 	D [float] - diameter of particle

	g = planet_data[planet]["g"]
	Ratmo = planet_data[planet]["R"]
	T = planet_data[planet]["fT"](np.log10(p))
	pPa = mbartopa(p)
	rho_air = pPa/(Ratmo*T)
	
	dynvisc_corr = planet_data[planet]["mucorr"]
	dynvisc = dynvisc_corr[0] + dynvisc_corr[1]*T
		
	if(phase == "rain"):
		rho_liq = spec_data[species]["rho_liq"]
		W = np.log10((4. * (D**3.)*rho_air*g*(rho_liq - rho_air))/(3.*dynvisc**2.))
		Re = 10.**(-1.81391 + 1.34671*W - 0.12427*W**2. + 0.006344*W**3.)
		
		return (dynvisc*Re)/(D*rho_air)
		
	elif(phase == "snow"):
		A_Ae = spec_data[species]["A_Ae_"+phase]
		rho_scale = spec_data[species]["rho_ice"]/spec_data["H2O"]["rho_ice"]
		m = 0.333*rho_scale*(D**2.4)
		X = ((8.*A_Ae*m*rho_air*g)/(np.pi*dynvisc**2))
		Re = 8.5*(np.sqrt(1. + 0.1519*np.sqrt(X)) - 1.)**2.

		return ((1./D)*(dynvisc*Re)/(rho_air))
	elif(phase == "ice"):
		A_Ae = spec_data[species]["A_Ae_"+phase]
		rho_scale = spec_data["H2O"]["rho_ice"]/spec_data[species]["rho_ice"]
		m = np.power((D/11.9)/rho_scale,2.)
		X = ((8.*A_Ae*m*rho_air*g)/(np.pi*dynvisc**2))
		Re = 8.5*(np.sqrt(1. + 0.1519*np.sqrt(X)) - 1.)**2.

		return ((1./D)*(dynvisc*Re)/(rho_air))

def vT_fit (x, a, b):
	return (a + b*x)
		
def vT_fit_old(x, a, b, c):
	return( a + b*x[:,0] + c*x[:,1])
	
global planet_data, spec_data
## Planet arrays
planet_data = {}
planet_data["Jupiter"]  = {"g": 22.31, "xH2": 0.864, "xHe": 0.136,"datfile": "jupiter_data.csv", "Pref": 1000.}
planet_data["Saturn"] = {"g":10.5, "xH2": 0.96,"xHe": 0.04, "datfile": "saturn_data.csv", "Pref": 1000.} 		

## Species arrays
## ant_ice and ant_liq are unnecessary and exist in case we need them for another calculation
spec_data = {}
spec_data["H2O"] = {"mass":18., "ant_liq": [11.079, -2261.1], "ant_ice": [12.61,-2681.18], "q": 0.001420, "rho_ice": 917.0, "rho_liq": 1000., "A_Ae_snow": 1.05, "A_Ae_ice": 1., "tp": 273.16}
spec_data["NH3"] = {"mass":17., "ant_liq": [10.201, -1248.0], "ant_ice": [11.90,-1588.00], "q": 0.000187, "rho_ice": 786.8, "rho_liq":  733., "A_Ae_snow": 1.05, "A_Ae_ice": 1., "tp": 195.20}

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
	Ratmo = 8314.4598/Matmo
	planet_data[planet]["R"] = Ratmo
	
	## Open the T-P and viscosity file
	file = planet_data[planet]["datfile"]
	
	data = np.genfromtxt(file,skip_header = 1, delimiter = ",",missing_values = '', filling_values = None)
	
	## Clean up missing values
	data = data[np.isfinite(data[:,1])]
	
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
	
	planet_data[planet]["tpPressure"] = {}
	
## Setup pressure intervals for vT calculation
P = {}
Pref = 1000.

''' Old pressure values #P["H2O"] = np.asarray([5000., 4750., 4500., 4000., 3500.,3000., 2500., 2000., 1000.])
#P["NH3"] = np.asarray([1000., 850., 700., 600., 500., 400.])'''

P["H2O"] = np.linspace(1000.,5000.,10)
P["NH3"] = np.linspace(400.,1000.,10)

## x, y, gamma, vT and chi are dictionaries which are organized as follows:
## [Variable] - either x, y, gamma or chi^2 of the fit
## 		First level - Species
##			Second level - Planet
##				Third level - Phase
## 					Fourth level - value of variable of Phase of Species at Planet.
## So to get the value of x for NH3 ice on Jupiter, you use:
## 		x["NH3"]["Jupiter"]["ice"]

## Set up terminal velocity dictionary
vT = {}

## Fit parameters

x = {}
y = {}
gamma = {}
chi = {}

'''DEPRECATED
x2 = {}
y2 = {}
gamma2 = {}
chi2 = {}'''

## Run terminal velocity for each species for each planet
for species in spec_data.keys():
	print("\nSpecies: %s"%(species))
	vT[species] = {}
	
	x[species] = {}
	y[species] = {}
	gamma[species] = {}
	chi[species] = {}
	
	'''DEPRECATED
	x2[species] = {}
	y2[species] = {}
	gamma2[species] = {}
	chi2[species] = {}'''
	
	for planet in planet_data.keys():
		print("\tPlanet: %s"%(planet))
		vT[species][planet] = {}
		x[species][planet] = {}
		y[species][planet] = {}
		gamma[species][planet] = {}
		chi[species][planet] = {}
		
		'''DEPRECATED
		x2[species][planet] = {}
		y2[species][planet] = {}
		gamma2[species][planet] = {}
		chi2[species][planet] = {}'''
		
		for phase in ["rain","ice","snow"]:
			print("\t\tPhase: %s"%(phase))
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
			
			print("\t\t\tCalculating terminal velocity profile")
			
			## Initiaze the vT array with empty array
			for p in P[species]:
				vT[species][planet][phase][p] = np.zeros(len(D))
			
			
			## This is formatting the vT data to save as .csv
			## In case we need to see the raw data
			vT_dat = np.zeros((len(D),2*len(P[species])+2))
			header = ""
			for i,p in enumerate(P[species]):
				vT_d = vT_calc(planet,species, phase, p, D, planet_data, spec_data)
				vT[species][planet][phase][P[species][i]] = vT_d
				header = header + "P=%i,vT,"%(int(p))				
				vT_dat[:,2*i] = D
				vT_dat[:,2*i+1] = vT_d
			
			## save vT data to csv
			outname = "%s_%s_%s.csv"%(planet,species,phase)
			np.savetxt(outname,vT_dat,header=header,delimiter=",")
			
			''' DEPRECATED
			## Calculate the terminal velocity and save it to the dictionary at the triple-point pressure
			vT_d = vT_calc(planet,species, phase, planet_data[planet]["tpPressure"][species], D, planet_data, spec_data)
			vT[species][planet][phase][planet_data[planet]["tpPressure"][species]] = vT_d'''

			gamma_vals = []			
						
			## Loop through each pressure to calculate the gamma fit for (P0/P)
			## log(vT(P)/vT(P0)) = gamma*np.log(P0/P)
			
			for p in P[species]:
				if(p != Pref):
					gamma_p = np.average(np.log(vT[species][planet][phase][p]/vT[species][planet][phase][Pref])/np.log(Pref/p))
					gamma_vals.append(gamma_p)
			
			gamma[species][planet][phase] = np.average(gamma_vals)
			
			gamma_vals = []
			
			'''DEPRECATED
			Pref2 = planet_data[planet]["tpPressure"][species]

			for p in P[species]:
				if(p != Pref2):
					gamma_p = np.average(np.log(vT[species][planet][phase][p]/vT[species][planet][phase][Pref2])/np.log(Pref2/p))
					gamma_vals.append(gamma_p)
			
			gamma2[species][planet][phase] = np.average(gamma_vals)
			'''
			
			## Convert 2D array of pressure and particle size to 1D (xData)
			## Also convert vT to 1D array (zData)
			## For fitting
			xData = []
			zData = []
			
			print("\t\tCreating array to fit")
			for i in range(len(P[species])):
				for j in range(len(D)):
					#index = (i*len(P[species])) + j
					xData.append(D[j])
					zData.append(vT[species][planet][phase][P[species][i]][j]/((Pref/P[species][i])**gamma[species][planet][phase]))
			xData = np.asarray(xData)
			zData = np.asarray(zData)

			print("\t\tFitting")
			par, cov = curve_fit(vT_fit, np.log10(xData),np.log10(zData))
			
			## Save the parameters
			x[species][planet][phase] = 10.**par[0]
			y[species][planet][phase] = par[1]
			
			
			'''DEPRECATED
			## Run with new triple-point reference pressure
			xData2 = []
			zData2 = []
			

				Pref2 = planet_data[planet]["tpPressure"][species]
				print("\t\tCreating array to fit")
				for i in range(len(P[species])):
					for j in range(len(D)):
						#index = (i*len(P[species])) + j
						xData2.append(D[j])
						zData2.append(vT[species][planet][phase][P[species][i]][j]/((Pref2/P[species][i])**gamma2[species][planet][phase]))
				xData2 = np.asarray(xData2)
				zData2 = np.asarray(zData2)
				
				print("\t\tFitting with triple point reference")
				par2, cov2 = curve_fit(vT_fit, np.log10(xData2),np.log10(zData2))
				
				
				x2[species][planet][phase] = 10.**par2[0]*((Pref2/Pref)**(gamma2[species][planet][phase]))
				y2[species][planet][phase] = par2[1]		'''

xtheo = {"NH3": {"snow": 48.606, "rain": 2479.}, "H2O": {"snow": 50.172,"rain": 2615.1}}
ytheo = {"NH3": {"snow": 0.48803, "rain": 0.76217}, "H2O": {"snow": 0.47798,"rain": 0.74245}}
gammatheo = {"snow": 0.47, "rain": 0.33}

## Plot parameters	
pltspec = "H2O"
pltplanet = "Saturn"
pltphase = "snow"
pltP = 1000.
vTp = vT[pltspec][pltplanet][pltphase]

## Initialize a figure
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

## Plot the vT points for the given plot paramter
ax1.plot(vTp["D"]*1000.,vTp[pltP],'k.')

## Get the fitted curve for the plot paramter
fitted = x[pltspec][pltplanet][pltphase]*(vTp["D"]**y[pltspec][pltplanet][pltphase])*(Pref/pltP)**gamma[pltspec][pltplanet][pltphase]

## Calculate R^2 statistic
sstot = np.sum((vTp[pltP] - np.average(vTp[pltP]))**2.)
ssres = np.sum((vTp[pltP] - fitted)**2.)
R2 = 1. - ssres/sstot

## Plot the fitted curve as a - line
ax1.plot(vTp["D"]*1000.,fitted,'k-')

## Annotate with R2 text
## The text function displays the text at the relative axis coordinate (first two numbers)
## So 0, 0 is the bot-left and 1,1 is the top-right and 0.5, 0.5 is in the middle of the plot
ax1.text(0.1,0.5,"R2: %.4f"%(R2),fontsize=12,transform=ax1.transAxes)

'''DEPRECATED
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
if(False):
	filename = "%s_%s_1bar.csv"%(pltspec,pltphase)
	dat = np.loadtxt(filename, delimiter=',')
	ax2.plot(dat[:,0]*1000.,dat[:,1],'b*')

ax2.plot(vTp["D"]*1000.,vTp[pltP],'b.')

fitted2 = x2[pltspec][pltplanet][pltphase]*(vTp["D"]**y2[pltspec][pltplanet][pltphase])*(Pref/pltP)**gamma2[pltspec][pltplanet][pltphase]
sstot = np.sum((vTp[pltP] - np.average(vTp[pltP]))**2.)
ssres = np.sum((vTp[pltP] - fitted2)**2.)

R2 = 1. - ssres/sstot
ax2.plot(vTp["D"]*1000.,fitted2,'b-')
ax2.text(0.8,0.5,"R2_theo: %.4f"%(R2))

if(pltplanet == "Jupiter"):
	if(pltphase in ["rain","snow"]):
		theo = xtheo[pltspec][pltphase]*(vTp["D"]**ytheo[pltspec][pltphase])*(Pref/pltP)**gammatheo[pltphase]
		ssres = np.sum((vTp[pltP] - theo)**2.)

		R2 = 1. - ssres/sstot
		ax2.plot(vTp["D"]*1000.,theo,'b--')
		ax2.text(0.5,0.8,"R2: %.4f"%(R2))

printplanet = "Jupiter"
for species in spec_data.keys():
	print(species)
	print("----------------")
	for phase in ["ice","snow","rain"]:
		print("Phase: %s"%(phase))
		print("x: %.4f \t x2 : %.4f"%(x[species][printplanet][phase],x2[species][printplanet][phase]))
		print("y: %.4f \t y2 : %.4f"%(y[species][printplanet][phase],y2[species][printplanet][phase]))
		print("gamma: %.4f \t gamma2 : %.4f"%(gamma[species][printplanet][phase],gamma2[species][printplanet][phase]))
		print()
	print()
plt.show()
'''

printplanet = "Jupiter"
for species in spec_data.keys():
	print(species)
	print("----------------")
	for phase in ["ice","snow","rain"]:
		print("Phase: %s"%(phase))
		print("x: %.4f"%(x[species][printplanet][phase]))
		print("y: %.4f"%(y[species][printplanet][phase]))
		print("gamma: %.4f"%(gamma[species][printplanet][phase]))
		print()
	print()
plt.show()
