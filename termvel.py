import numpy as np
import matplotlib.pyplot as plt
#from cycler import cycler
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
	# elif(planet == "Uranus"):  ## FROM Smyshlyaev et al. 2010 Izv. Atmos. Ocean Phys. 46: 265.
	# 	molwt = planet_data["Uranus"]["M"]
	# 	m = molwt/(6.02e23)*1.e-3
	# 	dmol = 2.*0.137e-9#0.3988e-9
	# 	kb = (1.38e-23)
	# 	sigma = np.pi*dmol**2.
	# 	t = np.linspace(80, 200.,500)
	# 	nu = np.sqrt(8.*m*t*kb/np.pi)/(3.*np.sqrt(2.)*sigma)
	# 	#corr = [0., np.sqrt(8.*m*kb/np.pi)/(3.*np.sqrt(2.)*sigma)]
	# 	return(least_squares(t,nu))
	else:
		## Fetch H2 and He concentrations
		x1 = planet_data[planet]["xi"]["H2"]
		x2 = planet_data[planet]["xi"]["He"]
		
		n = 400 ## Increased sample points
		viscx = np.zeros(n+1)
		viscy = np.zeros(n+1)
		for i in range(n+1):
			## Added larger temperature range to account for Uranus/Neptune atmosphere
			temperature = 70. + i*(500.-80.)/n; 
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
		rho = spec_data[species]["rho_ice"]
		m = spec_data[species]["rho_ice"]*((4./3.)*np.pi*(D/2.)**3.)
		
		
		X = ((8.*A_Ae*m*rho_air*g)/(np.pi*dynvisc**2))
		Re = 8.5*(np.sqrt(1. + 0.1519*np.sqrt(X)) - 1.)**2.

		if(species == "CH4"):
			dp = 2.*0.137e-9


		#gforce = m*g
		#vT = (gforce)/(6.*np.pi*dynvisc*(D/2.))
		#vT = ((1./D)*(dynvisc*Re)/(rho_air))
		vT = (2./9.)*(((D/2)**2.)*g/dynvisc)*(rho)

		for i, Di in enumerate(D):
			if Re[i] > 1.e-25:
				lambda_a = (1.38e-23)*T/(np.sqrt(2.)*np.pi*(dp**2.)*pPa)
				kn = lambda_a/(Di/2.)
				ckn = (1. + 1.26*kn)
				if(Di == 0.2e-6):
					print(p, vT[i]*100.)
				vT[i] = ckn*vT[i]
		return vT

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
#planet_data["Jupiter"]  = {"g": 22.31, "xi": {"H2":0.75,"He":0.25}, "datfile": "jupiter_data.csv", "Pref": 1000.}
#planet_data["Saturn"] = {"g":10.5, "xi":  {"H2":0.96,"He": 0.04}, "datfile": "saturn_data.csv", "Pref": 1000.}

## Uranus and Neptune atmos data from Encrenaz 2004 DOI: 10.1007/s11214-005-1950-6
planet_data["Uranus"] = {"g":8.69, "xi": {"H2":0.83,"He": 0.15,"CH4": 0.03}, "datfile": "uranus_data.csv", "Pref": 1000.}
#planet_data["Neptune"] = {"g":11.15, "xi": {"H2":0.79,"He": 0.18,"CH4": 0.03}, "datfile": "neptune_data.csv", "Pref": 1000.}

## Titan data from Lorenz 1993
#planet_data["Titan"] = {"g":1.352, "xi": {"N2":0.942,"H2": 0.001,"CH4": 0.056}, "datfile": "titan_data.csv", "Pref": 1000.}

## Venus data from Basilevsky and Head 2003
#planet_data["Venus"] = {"g":8.87, "xi": {"CO2":0.965,"N2": 0.0035}, "datfile": "venus_data.csv", "Pref": 1000.}

## Exoplanet data from Wakeford et al. 2016
#planet_data["Exo"] = {"g":10., "xi": {"H2":0.96,"He": 0.04}, "datfile": "exo_data.csv", "Pref": 1000.}

## Species arrays
## ant_ice and ant_liq are unnecessary and exist in case we need them for another calculation
spec_data = {}
spec_data["H2O"] = {"mass":18., "rho_ice": 917.0, "rho_liq": 1000., "A_Ae_snow": 1.05, "A_Ae_ice": 1., "docalc":True,"planet":["Jupiter","Saturn"]}
spec_data["NH3"] = {"mass":17., "rho_ice": 786.8, "rho_liq":  733., "A_Ae_snow": 1.05, "A_Ae_ice": 1., "docalc":True,"planet":["Jupiter","Saturn"]}
spec_data["NH4SH"] = {"mass":51., "rho_ice": (51./18.)*917.8, "rho_liq":  100., "A_Ae_snow": 1.05, "A_Ae_ice": 1., "docalc":True,"planet":["Jupiter","Saturn"]}

## From Uranus edited by Jay T. Bergstralh, Ellis D. Miner, Mildred ISBN: 978-0816512089 and #http://encyclopedia.airliquide.com/encyclopedia.asp?GasID=41#GeneralData
spec_data["CH4"] = {"mass":16., "rho_ice": 430., "rho_liq":  656., "A_Ae_snow": 1.05, "A_Ae_ice": 1., "docalc":True,"planet":["Titan","Uranus","Neptune"]}
#spec_data["H2S"] = {"mass":34., "rho_ice": 485., "rho_liq":  100., "A_Ae_snow": 1.05, "A_Ae_ice": 1.,  "docalc":True,"planet":["Uranus","Neptune"]}

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
	planet_data[planet]["M"] = Matmo
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

# P["CH4"] = np.asarray([30., 40.,50., 60., 70., 100., 1000., 1100., 1200., 1300., 2500., 2600., 2700., 2800.,\
# 	2900., 3000.])

P["CH4"] = 10.**np.arange(0.,4, 0.1)

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
	#print("\nSpecies: %s"%(species))
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
			#print("\t\tPhase: %s"%(phase))
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
				D = np.asarray([0.08,0.1,0.11,0.19,0.2,0.21,0.59,0.60,0.61, 2.])*1.e-6
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

# printplanet = "Uranus"
# printspecies = "CH4"
# print("\n\nP(mbar) \t D (mu m) \t m (kg) \t V (cm m/s) \n")
# print("-------------------------------------------------------------\n")
# for i, Pi in enumerate(P["CH4"]):
# 	# if(Pi < 1000.):
# 	# 	Di = 4
# 	# elif(Pi < 2000.):
# 	# 	Di = 6
# 	# else:
# 	# 	Di = 7
# 	Di = 4
# 	vT_dat = vT[printspecies][printplanet]["ice"]
# 	print("%4.f \t\t %.1f \t\t %.2e \t %.2e \n"%(Pi,vT_dat["D"][Di]*1.e6,430.*(4./3.)*np.pi*(vT_dat["D"][Di]/2.)**3.,vT_dat[Pi][Di]*1.e3))
# 	if(Pi in [100., 1300.]):
# 		print ("-----------------------------------------------------------\n")
# print("%4.f \t\t %.1f \t\t %.2e \t %.2e \n"%(Pi,vT_dat["D"][-1]*1.e6,430.*(4./3.)*np.pi*(vT_dat["D"][-1]/2.)**3.,vT_dat[Pi][-1]*1.e3))


# ; INPUTS
# ;   p = pressure in bars
# ;   t = temperature in K
# ;   g = acceleration of gravity in cm/s^2
# ;   vv = particle volume in microns^3
# ; OUTPUT
# ;  sedspeed = sedimentation speed in cm/s
# ; KEYWORDS
# ;  molwt = molecular weight of gas in grams/mole
# ;  cknsw = switch, set to 1 to turn off Cunningham correction
# ;  verbose = switch set to 1 to turn on printing of molwt,
# ;            P,T,mean freee path, Knudsen no.,Cunningham
# ;            correction factor
# ;==========================================================
# ; compute mean free path
# if n_elements(molwt) ne 0 then molwt = molwt else molwt=2
# ; compute mean free path lambda and viscosity eta
# ;print,'molwt=',molwt
# m = molwt/6.023D23   ; mass of molecule in grams
# ;dmol= 0.289E-7        ; diameter of an H2 molecule in cm
# if molwt eq 2 then dmol=2*0.137E-7 ; Sears Thermodynamics Text
# if molwt eq 29 then dmol=2*0.188E-7 ; Sears Thermodynamics Text
# kb = 1.38064852D-23  ; Boltzmann constant, m2 kg s-2 K-1
# kb=kb * 1e4          ; cm2 kg s-2 /K
# kb=kb * 1e3          ; cm2 g s^-2 K^-1
# sigma= !pi*dmol^2     ; cm^2
# pnpm2 = p*1E5 ; converts bars to Newton/m^2 = kg m/s^2/m2 = kg s^-2 m^-1
# pgps2pm = pnpm2*1e3 ; g/s2/m
# lambda = kb *T/(sqrt(2)*!pi*dmol^2*pgps2pm) ; cm2 g s-2 /(cm2 g s-2 m-1)
# lambda = lambda *1e2 ; mean free path in cm (for gas, but may need value for particle)
# ; Below, eta is viscosity 
# eta = sqrt(8*m*kb*T/!pi)/(3*Sqrt(2.)*sigma) ; (gm cm2 gm s-2 K-1 K)^1/2 cm-2
# ;       = gm cm /s  cm-2 = gm / cm-s  = poise = 10-3 kg/cm-s = 10^-1 kg/m-s 
# r=1e-4*(vv*3/(4.*!pi))^(1/3.) ; particle radius in cm
# kn=lambda/r  ; Knudsen number
# if n_elements(knval) ne 0 then knval=kn
# ;ckn = 1.+1.257*kn+0.4*kn*exp(-1.1/kn)   ; Fuchs (1064)
# ckn = 1.+1.249*kn+0.42*kn*exp(-0.87/kn) ; from Kasten (1968)
# if n_elements(cknsw) ne 0 then ckn=1.0
# rho = 1.0 ; g/cm^3  particle density
# xs = 1.0  ; shape factor
# xc = 1.0  ; collision factor
# w=xs*xc*(2/9.)*rho*r^2*g*ckn/eta
# if n_elements(verbose) ne 0 then begin
#     if verbose gt 0 then print,string(format=$
#      "('molwt=',i3,', P=',E10.2,' bars, T=',F7.2,'K, lambda=',E10.2,' cm, kn=',E10.4,', ckn=',E10.2,' Speed=',E10.2,' cm/s')",$
#      molwt,p,T,lambda,Kn,ckn,w)
# endif
# RETURN,w
# END


def sedspeed(p, t, g, r, molwt):
	rho = 0.43
	xs = 1.
	xc = 1.

	m = molwt/(6.02e23)
	dmol = 2*0.137e-7
	kb = (1.38e-23)*(1.e7)
	sigma = np.pi*dmol**2.
	pgps2pm = p*1.e8
	lambda_a = (kb*t/ (np.sqrt(2.)*np.pi*(dmol**2.)*pgps2pm))*1.e2
	eta = np.sqrt(8.*m*kb*t/np.pi)/(3.*np.sqrt(2.)*sigma)
	kn = lambda_a/r
	ckn = 1. + 1.249*kn + 0.42*kn*np.exp(-0.87/kn)
	v = xs*xc*(2./9.)*rho*(r**2.)*g/eta
	w = v*ckn
	for i, pi in enumerate(p):
		print(pi, v[i])
	return w


vT_dat = vT["CH4"]["Uranus"]["ice"]
D = vT_dat["D"][4]
vT = np.asarray([vT_dat[Pi][4]*1.e2 for Pi in P["CH4"]])

dynvisc_corr = planet_data["Uranus"]["mucorr"]
t = planet_data["Uranus"]["fT"](np.log10(P["CH4"]))

molwt = planet_data["Uranus"]["M"]
m = molwt/(6.02e23)*1.e-3
dmol = 2.*0.137e-9#0.3988e-9
kb = (1.38e-23)
sigma = np.pi*dmol**2.
eta1 = dynvisc_corr[0] + dynvisc_corr[1]*t
eta2 = np.sqrt(8.*m*kb*t/np.pi)/(3.*np.sqrt(2.)*sigma)

sed = sedspeed(P["CH4"]/1.e3,planet_data["Uranus"]["fT"](np.log10(P["CH4"])),869.,(D/2.)*100.,molwt)

plt.rc('text',usetex=True)
plt.rc('font',family='serif')

#plt.plot(sedx, sedy, '--')
plt.plot(vT,P["CH4"],'-')
plt.plot(sed, P["CH4"], '--')
plt.title(r'Terminal velocity')
plt.axes().set_yscale('log')
plt.axes().set_xscale('log')
plt.axes().set_ylim((10000.,1.))
plt.axes().set_xlim((1.e-5,1.))
plt.axes().set_ylabel(r'Pressure [mbar]')
plt.axes().set_xlabel(r'Terminal velocity [$\mu$m s$^{-1}$]')

plt.figure()
plt.plot(eta1, P["CH4"],'-')
plt.plot(eta2, P["CH4"],'--')
plt.plot(eta1 - eta2, P["CH4"],'-.')
plt.title(r'Viscosity')
plt.axes().set_xlim((1.e-6,1.e-5))
plt.axes().set_yscale('log')
plt.axes().set_xscale('log')
plt.axes().set_ylim((10000.,1.))
plt.axes().set_ylabel(r'Pressure [mbar]')
plt.axes().set_xlabel(r'Viscosity [kg m$^{-1}$ s$^{-1}$]')


# plt.figure()
# plt.plot(eta1-eta2, P["CH4"],'-')
# plt.axes().set_yscale('log')
# plt.axes().set_xscale('log')
# plt.axes().set_ylim((10000.,1.))

plt.show()