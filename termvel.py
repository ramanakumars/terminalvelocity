import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
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
	x1 = planet_data[planet]["xi"]["H2"]
	x2 = planet_data[planet]["xi"]["He"]
	
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
planet_data["Jupiter"]  = {"g": 22.31, "xi": {"H2":0.864,"He":0.136}, "datfile": "jupiter_data.csv", "Pref": 1000.}
planet_data["Saturn"] = {"g":10.5, "xi":  {"H2":0.96,"He": 0.04}, "datfile": "saturn_data.csv", "Pref": 1000.}

## Uranus and Neptune atmos data from Encrenaz 2004 DOI: 10.1007/s11214-005-1950-6
planet_data["Uranus"] = {"g":8.69, "xi": {"H2":0.83,"He": 0.15,"CH4": 0.03}, "datfile": "uranus_data.csv", "Pref": 1000.}
planet_data["Neptune"] = {"g":11.15, "xi": {"H2":0.79,"He": 0.18,"CH4": 0.03}, "datfile": "neptune_data.csv", "Pref": 1000.}

## Species arrays
## ant_ice and ant_liq are unnecessary and exist in case we need them for another calculation
spec_data = {}
spec_data["H2O"] = {"mass":18., "rho_ice": 917.0, "rho_liq": 1000., "A_Ae_snow": 1.05, "A_Ae_ice": 1., "tp": 273.16, "docalc":True}
spec_data["NH3"] = {"mass":17., "rho_ice": 786.8, "rho_liq":  733., "A_Ae_snow": 1.05, "A_Ae_ice": 1., "tp": 195.20, "docalc":True}

## From Uranus edited by Jay T. Bergstralh, Ellis D. Miner, Mildred ISBN: 978-0816512089 and #http://encyclopedia.airliquide.com/encyclopedia.asp?GasID=41#GeneralData
spec_data["CH4"] = {"mass":16., "rho_ice": 500., "rho_liq":  656., "A_Ae_snow": 1.05, "A_Ae_ice": 1., "tp": 90.54, "docalc":True} 
spec_data["H2"] = {"mass": 2., "docalc": False}
spec_data["He"] = {"mass": 4., "docalc": False}


## Setup the T-P profile, dyn. visc. data and atmospheric data
for planet in planet_data.keys():
	print("Setting up %s"%(planet))
	## Calculate molar mass of atmosphere from H2 and He concentrations
	Matmo = molarmass([planet_data[planet]["xi"][spec] for spec in planet_data[planet]["xi"].keys()],[spec_data[spec]["mass"] for spec in planet_data[planet]["xi"].keys()])
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

P["H2O"] = np.arange(1000.,8000.,500.)
P["NH3"] = np.arange(400.,1000.,100.)
P["CH4"] = np.arange(600.,2000.,100.)

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

## Run terminal velocity for each species for each planet
for species in spec_data.keys():
	if(spec_data[species]["docalc"] == False):
		continue
	print("\nSpecies: %s"%(species))
	vT[species] = {}
	
	x[species] = {}
	y[species] = {}
	gamma[species] = {}
	chi[species] = {}
	
	if not Pref in P[species]:
		P[species] = np.append(P[species],Pref)

	
	for planet in planet_data.keys():
		print("\tPlanet: %s"%(planet))
		vT[species][planet] = {}
		x[species][planet] = {}
		y[species][planet] = {}
		gamma[species][planet] = {}
		chi[species][planet] = {}
		
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
			
			gamma_vals = []			
						
			## Loop through each pressure to calculate the gamma fit for (P0/P)
			## log(vT(P)/vT(P0)) = gamma*np.log(P0/P)
			
			for p in P[species]:
				if(p != Pref):
					gamma_p = np.average(np.log(vT[species][planet][phase][p]/vT[species][planet][phase][Pref])/np.log(Pref/p))
					gamma_vals.append(gamma_p)
			
			gamma[species][planet][phase] = np.average(gamma_vals)
			
			gamma_vals = []

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

## Initialize LateX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams["text.latex.preamble"].append(r'\usepackage{amsmath}')

## vT vs P plot for Neptune CH4
pltspec = "H2O"
pltplanet = "Jupiter"
pltphase = "snow"

Pvals = np.sort(P[pltspec][::2])

plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'k','c','m'])))

## Initialize a figure
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
vTp = vT[pltspec][pltplanet][pltphase]

for pltP in Pvals:
	## Plot the vT points for the given plot paramter
	#ax1.plot(vTp["D"]*1000.,vTp[pltP],'--')

	## Get the fitted curve for the plot paramter
	fitted = x[pltspec][pltplanet][pltphase]*(vTp["D"]**y[pltspec][pltplanet][pltphase])*(Pref/pltP)**gamma[pltspec][pltplanet][pltphase]
	ax1.plot(vTp["D"]*1000.,fitted,label=r"P = %d mbar"%pltP)
	
ax1.set_xlabel(r"Diameter (mm)")
ax1.set_ylabel(r"Terminal velocity (m s$^{-1}$)")
plt.legend(loc='upper right')


## Create a multiple plot
latexspec = {"H2O":"H$_{2}$O","NH3":"NH$_{3}$"}
fig2 = plt.figure()
ax2 = fig2.add_subplot(211)

## Do Jupiter
for spec in ["H2O","NH3"]:
	vTp = vT[spec]["Jupiter"]["snow"]
	xi = x[spec]["Jupiter"]["snow"]
	yi = y[spec]["Jupiter"]["snow"]
	
	D = vTp["D"]
	
	fit = xi*(D**yi)
	
	ax2.plot(D, fit, label = r"%s - %s"%(latexspec[spec], "Snow"))
plt.legend(loc='upper right')	
## Do Saturn

ax3 = fig2.add_subplot(212)
for spec in ["H2O","NH3"]:
	vTp = vT[spec]["Saturn"]["snow"]
	xi = x[spec]["Saturn"]["snow"]
	yi = y[spec]["Saturn"]["snow"]
	
	D = vTp["D"]
	
	fit = xi*(D**yi)
	
	ax3.plot(D, fit, label = r"%s - %s"%(latexspec[spec], "Snow"))
plt.legend(loc='upper right')

fig3 = plt.figure()
	
## Do Uranus

ax4 = fig3.add_subplot(111)
for planet in ["Uranus","Neptune"]:
	vTp = vT["CH4"][planet]["snow"]
	xi = x["CH4"][planet]["snow"]
	yi = y["CH4"][planet]["snow"]
	
	D = vTp["D"]
	
	fit = xi*(D**yi)
	
	ax4.plot(D, fit, label = r"%s - %s"%(planet, "Snow"))
plt.legend(loc='upper right')

plt.show()

''' Do Not Need This
## Plot parameters	
pltspec = "CH4"
pltplanet = "Neptune"
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
ax1.text(0.1,0.5,r"$R^2: %.4f$"%(R2),fontsize=15,transform=ax1.transAxes)'''




## Do not need to print values anymore
printplanet = "Neptune"
for species in spec_data.keys():
	if(spec_data[species]["docalc"] == False):
		continue
	print(species)
	print("----------------")
	for phase in ["ice","snow","rain"]:
		print("Phase: %s"%(phase))
		print("x: %.4f"%(x[species][printplanet][phase]))
		print("y: %.4f"%(y[species][printplanet][phase]))
		print("gamma: %.4f"%(gamma[species][printplanet][phase]))
		print()
	print()
##'''

plt.show()