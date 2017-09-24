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
	
	
	## Titan viscosity from Lorenz (1993) Equation A1
	
	## Venus viscosity from Petropoulos 1988
	
	global planet_data
	
	if(planet == "Titan"):
		return([1.718e-5 - 5.1e-8*273.,5.1e-8])
	elif(planet == "Venus"):
		return([4.46e-6, 3.64e-8])
	else:
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
		#if(planet == "Exo" and p == 100.):
			#print np.aver(10.**W/Re**2.)
		return (dynvisc*Re)/(D*rho_air)
		
	elif(phase == "snow"):
		A_Ae = spec_data[species]["A_Ae_"+phase]
		rho_scale = 0.5*spec_data[species]["rho_ice"]/spec_data["H2O"]["rho_ice"]
		m = 0.333*rho_scale*(D**2.4)
		X = ((8.*A_Ae*m*rho_air*g)/(np.pi*dynvisc**2))
		Re = 8.5*(np.sqrt(1. + 0.1519*np.sqrt(X)) - 1.)**2.

		return ((1./D)*(dynvisc*Re)/(rho_air))
	elif(phase == "ice"):
		A_Ae = spec_data[species]["A_Ae_"+phase]
		rho_scale = spec_data[species]["rho_ice"]/spec_data["H2O"]["rho_ice"]
		m = np.power((D/11.9),2.)*rho_scale
		X = ((8.*A_Ae*m*rho_air*g)/(np.pi*dynvisc**2))
		Re = 8.5*(np.sqrt(1. + 0.1519*np.sqrt(X)) - 1.)**2.

		return ((1./D)*(dynvisc*Re)/(rho_air))

def cD_rain(planet,p, species):
	global planet_data, spec_data
	
	g = planet_data[planet]["g"]
	Ratmo = planet_data[planet]["R"]
	T = planet_data[planet]["fT"](np.log10(p))
	pPa = mbartopa(p)
	rho_air = pPa/(Ratmo*T)
	
	dynvisc_corr = planet_data[planet]["mucorr"]
	dynvisc = dynvisc_corr[0] + dynvisc_corr[1]*T
	rho_liq = spec_data[species]["rho_liq"]
	W = np.log10((4. * (D**3.)*rho_air*g*(rho_liq - rho_air))/(3.*dynvisc**2.))
	Re = 10.**(-1.81391 + 1.34671*W - 0.12427*W**2. + 0.006344*W**3.)
	return (10.**W)/Re**2.
		
def vT_fit (x, a, b):
	return (a + b*x)
		
def vT_fit_old(x, a, b, c):
	return( a + b*x[:,0] + c*x[:,1])
	
global planet_data, spec_data
## Planet arrays
planet_data = {}
# planet_data["Jupiter"]  = {"g": 22.31, "xi": {"H2":0.75,"He":0.25}, "datfile": "jupiter_data.csv", "Pref": 1000.}
# planet_data["Saturn"] = {"g":10.5, "xi":  {"H2":0.96,"He": 0.04}, "datfile": "saturn_data.csv", "Pref": 1000.}

## Uranus and Neptune atmos data from Encrenaz 2004 DOI: 10.1007/s11214-005-1950-6
planet_data["Uranus"] = {"g":8.69, "xi": {"H2":0.83,"He": 0.15,"CH4": 0.03}, "datfile": "uranus_data.csv", "Pref": 1000.}
planet_data["Neptune"] = {"g":11.15, "xi": {"H2":0.79,"He": 0.18,"CH4": 0.03}, "datfile": "neptune_data.csv", "Pref": 1000.}

# ## Titan data from Lorenz 1993
# planet_data["Titan"] = {"g":1.352, "xi": {"N2":0.942,"H2": 0.001,"CH4": 0.056}, "datfile": "titan_data.csv", "Pref": 1000.}

# ## Venus data from Basilevsky and Head 2003
# planet_data["Venus"] = {"g":8.87, "xi": {"CO2":0.965,"N2": 0.0035}, "datfile": "venus_data.csv", "Pref": 1000.}

# ## Exoplanet data from Wakeford et al. 2016
# planet_data["Exo"] = {"g":10., "xi": {"H2":0.96,"He": 0.04}, "datfile": "exo_data.csv", "Pref": 1000.}

## Species arrays
## ant_ice and ant_liq are unnecessary and exist in case we need them for another calculation
spec_data = {}
spec_data["H2O"] = {"mass":18., "rho_ice": 917.0, "rho_liq": 1000., "A_Ae_snow": 1.05, "A_Ae_ice": 1., "docalc":True,"planet":["Jupiter","Saturn"]}
spec_data["NH3"] = {"mass":17., "rho_ice": 786.8, "rho_liq":  733., "A_Ae_snow": 1.05, "A_Ae_ice": 1., "docalc":True,"planet":["Jupiter","Saturn"]}
spec_data["NH4SH"] = {"mass":51., "rho_ice": (51./18.)*917.8, "rho_liq":  100., "A_Ae_snow": 1.05, "A_Ae_ice": 1., "docalc":True,"planet":["Jupiter","Saturn"]}

## From Uranus edited by Jay T. Bergstralh, Ellis D. Miner, Mildred ISBN: 978-0816512089 and #http://encyclopedia.airliquide.com/encyclopedia.asp?GasID=41#GeneralData
spec_data["CH4"] = {"mass":16., "rho_ice": 100., "rho_liq":  656., "A_Ae_snow": 1.05, "A_Ae_ice": 1., "docalc":True,"planet":["Titan","Uranus","Neptune"]}
spec_data["H2S"] = {"mass":34., "rho_ice": 485., "rho_liq":  100., "A_Ae_snow": 1.05, "A_Ae_ice": 1.,  "docalc":True,"planet":["Uranus","Neptune"]}

## Ethane is turned off for lack of densities
spec_data["C2H6"] = {"mass":30., "rho_ice": 100., "rho_liq":  656., "A_Ae_snow": 1.05, "A_Ae_ice": 1., "docalc":False,"planet":["Titan"]}

spec_data["H2SO4"] = {"mass":98., "rho_ice": 100., "rho_liq":  1840., "A_Ae_snow": 1.05, "A_Ae_ice": 1., "docalc":True,"planet":["Venus"]}

spec_data["Fe"] = {"mass":142., "rho_ice": 100., "rho_liq":  6980., "A_Ae_snow": 1.05, "A_Ae_ice": 1., "docalc":True,"planet":["Exo"]}



## Non-condensing species
spec_data["H2"] = {"mass": 2., "docalc": False}
spec_data["He"] = {"mass": 4., "docalc": False}
spec_data["CO2"] = {"mass": 38., "docalc": False}
spec_data["N2"] = {"mass": 28., "docalc": False}


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
Pref = 100.

P["H2O"] = np.arange(1000.,5000.,100.)
P["NH3"] = np.arange(400.,1000.,100.)
P["CH4"] = np.arange(600.,2000.,100.)
P["NH4SH"] = {}
P["NH4SH"]["Jupiter"] = np.arange(2000.,4000.,100.)
P["NH4SH"]["Saturn"] = np.arange(6000.,8000.,100.)
P["H2S"] = {}
P["H2S"]["Uranus"] = np.arange(3000.,5000.,100.)
P["H2S"]["Neptune"] = np.arange(8000.,12000.,100.)
P["H2SO4"] = np.arange(600., 2000., 100.)
P["Fe"] = np.arange(1., 100., 5.)

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
	
	for planet in planet_data.keys():
		if(planet not in spec_data[species]["planet"]):
			continue
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
			
			if(planet=="Titan"):
				Prange = np.linspace(np.min(P[species]),np.max(planet_data[planet]["Prange"]),100.)
			elif(species == "H2S"):
				Prange = P["H2S"][planet]
			elif(species == "NH4SH"):
				Prange = P["NH4SH"][planet]
			else:
				Prange = P[species]
			
			if not Pref in Prange:
				Prange = np.append(Prange,Pref)

			
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
			for p in Prange:
				vT[species][planet][phase][p] = np.zeros(len(D))
			
			
			## This is formatting the vT data to save as .csv
			## In case we need to see the raw data
			vT_dat = np.zeros((len(D),2*len(Prange)+2))
			header = ""
			for i,p in enumerate(Prange):
				vT_d = vT_calc(planet,species, phase, p, D, planet_data, spec_data)
				vT[species][planet][phase][Prange[i]] = vT_d
				header = header + "P=%i,vT,"%(int(p))				
				vT_dat[:,2*i] = D
				vT_dat[:,2*i+1] = vT_d
			
			## save vT data to csv
			outname = "data\%s_%s_%s.csv"%(planet,species,phase)
			np.savetxt(outname,vT_dat,header=header,delimiter=",")
			
			gamma_vals = []			
						
			## Loop through each pressure to calculate the gamma fit for (P0/P)
			## log(vT(P)/vT(P0)) = gamma*np.log(P0/P)
			
			for p in Prange:
				if(p != Pref):
					gamma_p = np.average(np.log(vT[species][planet][phase][p]/vT[species][planet][phase][Pref])/np.log(Pref/p))
					gamma_vals.append(gamma_p)
			
			gamma[species][planet][phase] = np.average(gamma_vals)
			#if(phase == 'rain'):
			#	gamma[species][planet][phase] = 0.33
			#else:
			#	gamma[species][planet][phase] = 0.47

			## Convert 2D array of pressure and particle size to 1D (xData)
			## Also convert vT to 1D array (zData)
			## For fitting
			xData = []
			zData = []
			
			print("\t\tCreating array to fit")
			for i in range(len(Prange)):
				for j in range(len(D)):
					#index = (i*len(Prange)) + j
					xData.append(D[j])
					zData.append(vT[species][planet][phase][Prange[i]][j]/((Pref/Prange[i])**gamma[species][planet][phase]))
			xData = np.asarray(xData)
			zData = np.asarray(zData)

			print("\t\tFitting")
			par, cov = curve_fit(vT_fit, np.log10(xData),np.log10(zData))
			
			## Save the parameters
			x[species][planet][phase] = 10.**par[0]
			y[species][planet][phase] = par[1]

			
print("Doing plots")
## Initialize LateX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size='14')
plt.rcParams["text.latex.preamble"].append(r'\usepackage{amsmath}')


params = {'xtick.labelsize':'xx-large','ytick.labelsize':'xx-large'}
plt.rcParams.update(params)

plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'k','c','m'])))

## Define whitespaces
left = 0.1
right = 0.99
top = 0.98
bottom = 0.1
wspace = 0.08
hspace = 0.17

## textsizes
textsize = 28
labelsize = 28

## x, y pos
titlex = 0.4
titley = 0.95

## textposition
textpos = (0.05, 0.85)

## Initialize a figure
if(False):
	## vT vs P plot for Jupiter CH4
	pltspec = "H2O"
	pltplanet = "Jupiter"
	pltphase = "snow"

	Pvals = np.sort(P[pltspec][::10])


	fig1 = plt.figure(figsize=(10,10))
	ax1 = fig1.add_subplot(111)
	vTp = vT[pltspec][pltplanet][pltphase]

	colors = ['r','g','b','k','c','m']

	for i,pltP in enumerate(Pvals):
		## Plot the vT points for the given plot paramter
		ax1.plot(vTp["D"]*1000.,vTp[pltP],'--',color=colors[i])

		## Get the fitted curve for the plot paramter
		fitted = x[pltspec][pltplanet][pltphase]*(vTp["D"]**y[pltspec][pltplanet][pltphase])*(Pref/pltP)**gamma[pltspec][pltplanet][pltphase]
		
		## Calculate R^2 statistic
		sstot = np.sum((vTp[pltP] - np.average(vTp[pltP]))**2.)
		ssres = np.sum((vTp[pltP] - fitted)**2.)
		R2 = 1. - ssres/sstot
		
		ax1.plot(vTp["D"]*1000.,fitted,color=colors[i],label=r"P = $%d$ bar - $R^{2} = %.3f$"%(pltP/1000.,R2))
	plt.grid()
	plt.subplots_adjust(bottom = bottom, top = top, right = right, left = left, wspace = wspace, hspace = hspace)
	ax1.set_xlabel(r"Diameter [mm]",fontsize=labelsize)
	ax1.set_ylabel(r"Terminal velocity [m s$^{-1}$]",fontsize=labelsize)
	ax1.set_xlim((0.9*np.min(vTp["D"]*1000.), 1.05*np.max(vTp["D"]*1000.)))
	plt.legend(loc='lower right',fontsize="xx-large")
	ax1.text(titlex,titley,"Fig. 2",fontsize=textsize,transform=ax1.transAxes)
	plt.savefig('plots/jup_h2o_vT_P.png')


if(False):
	## Create a multiple plot
	latexspec = {"H2O":"H$_{2}$O","NH3":"NH$_{3}$","NH4SH":"NH$_{4}$SH"}
	fig2 = plt.figure(figsize=(10,10))
	ax2 = fig2.add_subplot(111)

	'''
	## Do H2O
	for planet in ["Jupiter","Saturn"]:
		vTp = vT["H2O"][planet]["snow"]
		xi = x["H2O"][planet]["snow"]
		yi = y["H2O"][planet]["snow"]
		
		D = vTp["D"]
		
		fit = xi*(D**yi)
		
		ax2.plot(D*1000., fit, label = r"%s - H$_{2}$O %s"%(planet, "snow"))
		ax2.set_xlim((0.9*np.min(D*1000.), 1.05*np.max(D*1000.)))
	plt.subplots_adjust(bottom = bottom, top = top, right = right, left = left, wspace = wspace, hspace = hspace)
	plt.legend(loc='upper left')
	ax2.text(textpos[0],textpos[1],"(a)",fontsize=textsize,transform=ax2.transAxes)
	ax2.set_xlabel(r"Diameter [mm]",fontsize=labelsize)
	ax2.set_ylabel(r"Terminal velocity [m s$^{-1}$]",fontsize=labelsize)

	## Do NH3
	for planet in ["Jupiter","Saturn"]:
		vTp = vT["NH3"][planet]["snow"]
		xi = x["NH3"][planet]["snow"]
		yi = y["NH3"][planet]["snow"]
		
		D = vTp["D"]
		
		fit = xi*(D**yi)
		
		ax2.plot(D*1000., fit, '--',label = r"%s - NH$_{3}$ %s"%(planet, "snow"))
		ax2.set_xlim((0.9*np.min(D*1000.), 1.05*np.max(D*1000.)))
		
	## Do NH4SH
	for planet in ["Jupiter","Saturn"]:
		vTp = vT["NH4SH"][planet]["snow"]
		xi = x["NH4SH"][planet]["snow"]
		yi = y["NH4SH"][planet]["snow"]
		
		D = vTp["D"]
		
		fit = xi*(D**yi)
		
		ax2.plot(D*1000., fit, '-.',label = r"%s - NH$_{4}$SH %s"%(planet, "snow"))
		ax2.set_xlim((0.9*np.min(D*1000.), 1.05*np.max(D*1000.)))
	'''
	colors = ['r','g','b']
	for i,species in enumerate(["H2O","NH3","NH4SH"]):
		vTpJup = vT[species]["Jupiter"]["snow"]
		xiJup = x[species]["Jupiter"]["snow"]
		yiJup = y[species]["Jupiter"]["snow"]
		
		DJup = vTpJup["D"]
		
		fitJup = xiJup*(DJup**yiJup)
		
		ax2.plot(DJup*1000., fitJup, '-',color=colors[i],label = r"%s - %s %s"%("Jupiter",latexspec[species],"snow"))
	
	for i,species in enumerate(["H2O","NH3","NH4SH"]):
		vTpSat = vT[species]["Saturn"]["snow"]
		xiSat = x[species]["Saturn"]["snow"]
		yiSat = y[species]["Saturn"]["snow"]
		
		DSat = vTpSat["D"]
		
		fitSat = xiSat*(DSat**yiSat)
		
		ax2.plot(DSat*1000., fitSat, '--',color=colors[i],label = r"%s - %s %s"%("Saturn",latexspec[species],"snow"))
		
	plt.subplots_adjust(bottom = 0.08, top = top, right = right, left = left, wspace = wspace, hspace = hspace)
	plt.grid()
	ax2.set_xlim((0.9*np.min(DJup*1000.), 1.05*np.max(DJup*1000.)))
	ax2.set_xlabel(r"Diameter [mm]",fontsize=labelsize)
	ax2.set_ylabel(r"Terminal velocity [m s$^{-1}$]",fontsize=labelsize)
	plt.legend(loc='lower right',fontsize="large")
	ax2.text(titlex, titley,"Fig. 4",fontsize=textsize,transform=ax2.transAxes)
#	ax2.set_xlabel(r"Diameter [mm]",fontsize=labelsize)
	plt.savefig('plots/jup_sat_h2o_nh3.png')
	
## Do Uranus and Neptune
if(False):
	fig3 = plt.figure(figsize=(10,10))
	ax4 = fig3.add_subplot(111)
	latexspec = {"CH4":"CH$_{4}$","H2S":"H$_{2}$S"}
	
	for i,species in enumerate(["CH4","H2S"]):
		vTpUra = vT[species]["Uranus"]["snow"]
		xiUra = x[species]["Uranus"]["snow"]
		yiUra = y[species]["Uranus"]["snow"]
		
		DUra = vTpUra["D"]
		
		fitUra = xiUra*(DUra**yiUra)
		
		ax4.plot(DUra*1000., fitUra, '-',color=colors[i],label = r"%s - %s %s"%("Uranus",latexspec[species],"snow"))
	
	for i,species in enumerate(["CH4","H2S"]):
		vTpNep = vT[species]["Neptune"]["snow"]
		xiNep = x[species]["Neptune"]["snow"]
		yiNep = y[species]["Neptune"]["snow"]
		
		DNep = vTpNep["D"]
		
		fitNep = xiNep*(DNep**yiNep)
		
		ax4.plot(DNep*1000., fitNep, '--',color=colors[i],label = r"%s - %s %s"%("Neptune",latexspec[species],"snow"))
		
	plt.subplots_adjust(bottom = bottom, top = top, right = right, left = left, wspace = wspace, hspace = hspace)
	plt.grid()
	ax4.set_xlim((0.9*np.min(DUra*1000.), 1.05*np.max(DUra*1000.)))
	ax4.set_xlabel(r"Diameter [mm]",fontsize=labelsize)
	ax4.set_ylabel(r"Terminal velocity [m s$^{-1}$]",fontsize=labelsize)
	plt.legend(loc='lower right',fontsize="xx-large")
	ax4.text(titlex,titley,"Fig. 5",fontsize=textsize,transform=ax4.transAxes)
	plt.subplots_adjust(bottom = bottom, top = top, right = right, left = left, wspace = wspace, hspace = hspace)
	plt.savefig('plots/uranus_neptune_h2s_ch4.png')
## Do Venus
## 1 bat at ~50km
if(False):
	fig4 = plt.figure(figsize=(10,10))
	ax6 = fig4.add_subplot(111)
	vTp = vT["H2SO4"]["Venus"]["rain"]
	xi = x["H2SO4"]["Venus"]["rain"]
	yi = y["H2SO4"]["Venus"]["rain"]
	
	D = vTp["D"]
	
	fit = xi*(D**yi)
	
	ax6.plot(D*1000., fit, label = r"Venus - H$_{2}$SO$_{4}$ rain")
	ax6.set_xlabel(r"Diameter [mm]",fontsize=labelsize)
	ax6.set_ylabel(r"Terminal velocity [m s$^{-1}$]",fontsize=labelsize)
	plt.legend(loc='upper left')
## Do Titan
## 1 bar at ~8.25km
	vTp = vT["CH4"]["Titan"]["rain"]
	xi = x["CH4"]["Titan"]["rain"]
	yi = y["CH4"]["Titan"]["rain"]
	
	D = vTp["D"]
	
	fit = xi*(D**yi)
	
	ax6.plot(D*1000., fit, label = r"Titan - CH$_{4}$ rain")

	ax6.set_xlabel(r"Diameter [mm]",fontsize=labelsize)
	ax6.set_ylabel(r"Terminal velocity [m s$^{-1}$]",fontsize=labelsize)
	
	ax6.set_xlim((0.9*np.min(D*1000.), 1.05*np.max(D*1000.)))
	
	plt.legend(loc='upper left')
	ax6.text(textpos[0],textpos[1],"(e)",fontsize=textsize,transform=ax6.transAxes)
	plt.grid()
	plt.subplots_adjust(bottom = bottom, top = top, right = right, left = left, wspace = wspace, hspace = hspace)
	plt.savefig('plots/venus_titan_rain.png')
	
	
if(False):
	fig5 = plt.figure(figsize=(10,10))
	ax7 = fig5.add_subplot(111)
	vTp = vT["Fe"]["Exo"]["rain"]
	xi = x["Fe"]["Exo"]["rain"]
	yi = y["Fe"]["Exo"]["rain"]
	
	D = vTp["D"]
	
	fit = xi*(D**yi)
	
	ax7.plot(D*1000., fit, label = r"Hot Jupiter - Fe rain")
	ax7.set_xlabel(r"Diameter [mm]",fontsize=labelsize)
	ax7.set_ylabel(r"Terminal velocity [m s$^{-1}$]",fontsize=labelsize)
	plt.legend(loc='upper left',fontsize="xx-large")
	ax7.set_xlim((0.9*np.min(D*1000.), 1.05*np.max(D*1000.)))
	ax7.text(textpos[0],textpos[1],"(f)",fontsize=textsize,transform=ax7.transAxes)
	plt.grid()
	plt.subplots_adjust(bottom = bottom, top = top, right = right, left = 0.09, wspace = wspace, hspace = hspace)
	plt.savefig('plots/exo_rain.png')

if(False):
	## Plot parameters	
	pltspec = "Fe"
	pltplanet = "Exo"
	pltphase = "rain"
	pltP = 100.
	T = planet_data[pltplanet]["fT"](np.log10(pltP))
	rho = (pltP*100.)/(8.314*T)

	vTp = vT[pltspec][pltplanet][pltphase]

	dmax = 	(8.*sigma)/(cD*rho*(vTp[pltP]**2.))
	
	## Initialize a figure
	fig1 = plt.figure()
	ax1 = fig1.add_subplot(111)

	## Plot the vT points for the given plot paramter
	ax1.plot(vTp["D"]*1000.,vTp[pltP],'k-')
	ax1.plot(dmax*1000.,vTp[pltP],'k-')
	
	meeting = np.argmin((dmax-vTp["D"])**2.)
	
	print(dmax[meeting]*1000., vTp[pltP][meeting])
	
	ax1.set_xlim((0.9*np.min(D*1000.), 1.05*np.max(D*1000.)))
	
	## Get the fitted curve for the plot paramter
	fitted = x[pltspec][pltplanet][pltphase]*(vTp["D"]**y[pltspec][pltplanet][pltphase])*(Pref/pltP)**gamma[pltspec][pltplanet][pltphase]

	## Calculate R^2 statistic
	sstot = np.sum((vTp[pltP] - np.average(vTp[pltP]))**2.)
	ssres = np.sum((vTp[pltP] - fitted)**2.)
	R2 = 1. - ssres/sstot

	## Plot the fitted curve as a - line
	#ax1.plot(vTp["D"]*1000.,fitted,'k-')

	## Annotate with R2 text
	## The text function displays the text at the relative axis coordinate (first two numbers)
	## So 0, 0 is the bot-left and 1,1 is the top-right and 0.5, 0.5 is in the middle of the plot
	ax1.text(0.1,0.5,r"$R^2: %.4f$"%(R2),fontsize=textsize,transform=ax1.transAxes)
	plt.show()
if(True):
	## Do not need to print values anymore
	for printplanet in planet_data.keys():
		## Do not need to print values anymore
		print(printplanet)
		for species in spec_data.keys():
			if(spec_data[species]["docalc"] == False):
				continue
			if(printplanet not in spec_data[species]["planet"]):
				continue
			print(species)
			print("----------------")
			for phase in ["rain","ice","snow"]:
				print("Phase: %s"%(phase))
				print("x: %.4f"%(x[species][printplanet][phase]))
				print("y: %.4f"%(y[species][printplanet][phase]))
				print("gamma: %.4f"%(gamma[species][printplanet][phase]))
				print()
			print()
	##'''

#plt.show()