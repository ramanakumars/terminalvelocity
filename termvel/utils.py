import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
# plt.subplots_adjust(bottom = bottom, top = top, right = right, left = left, wspace = wspace, hspace = hspace)


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
    
def vT_calc(planet, species, phase, p, D):
    ## Terminal velocity calculation
    ## From Palotai, Cs. and Dowling. T. E. 2008, Icarus 194, 303-326
    ## Variables: 
    ##  planet [string] - name of the planet
    ## 	species [string] - chemical formula of species (as in spec_data array)
    ## 	phase [string] - either rain, snow or ice
    ##	p [float] - Pressure in mbar
    ## 	D [float] - diameter of particle

    g = planet.g
    Ratmo = planet.Ratmo
    T = planet.fT(np.log10(p))
    pPa = mbartopa(p)
    rho_air = pPa/(Ratmo*T)
    
    dynvisc_corr = planet.dynvisc_corr
    dynvisc = dynvisc_corr[0] + dynvisc_corr[1]*T
    if(phase == "rain"):
        rho_liq = species.rho_liq
        W = np.log10((4. * (D**3.)*rho_air*g*(rho_liq - rho_air))/(3.*dynvisc**2.))
        Re = 10.**(-1.81391 + 1.34671*W - 0.12427*W**2. + 0.006344*W**3.)
        #if(planet == "Exo" and p == 100.):
            #print np.aver(10.**W/Re**2.)
        return (dynvisc*Re)/(D*rho_air)
        
    elif(phase == "snow"):
        A_Ae = species.A_Ae_snow
        rho_scale = 0.5*species.rho_ice_scale
        m = 0.333*rho_scale*(D**2.4)
        X = ((8.*A_Ae*m*rho_air*g)/(np.pi*dynvisc**2))
        Re = 8.5*(np.sqrt(1. + 0.1519*np.sqrt(X)) - 1.)**2.

        return ((1./D)*(dynvisc*Re)/(rho_air))
    elif(phase == "ice"):
        A_Ae = species.A_Ae_ice
        rho_scale = species.rho_ice_scale
        m = np.power((D/11.9),2.)*rho_scale
        X = ((8.*A_Ae*m*rho_air*g)/(np.pi*dynvisc**2))
        Re = 8.5*(np.sqrt(1. + 0.1519*np.sqrt(X)) - 1.)**2.

        return ((1./D)*(dynvisc*Re)/(rho_air))

        # Cd = np.zeros_like(D)
        # Cd[D<100.e-6] = 24./Re[D<100.e-6]
        # Cd[(D>=100.e-6)&(Re<=3.e5)] = 0.6
        # Cd[(D>=100.e-6)&(Re>3.e5)]  = 0.1

        # Re  = np.zeros_like(D)
        # Cd  = np.zeros_like(D)
        # Cd2 = np.zeros_like(D)
        # return (np.sqrt((4*species.rho_ice*g*D)/(3.*Cd2*rho_air)))
        Cd2 = 24./Re

        ''' from Rasmussen & (someone else) 1987 '''
        # for i, Di in enumerate(D):
        #     Xi = X[i]
        #     Wi = np.log10(Xi)
        #     if((Xi>73.)&(Xi<562.)):
        #         Re[i] = 10.**(-1.7095 + 1.33438*Wi - 0.11591*Wi**2.)

        #     elif((Xi>562.)&(Xi<=1.83e3)):
        #         Re[i] = 10.**(-1.81391 + 1.34671*Wi - 0.12427*Wi**2. + 0.0063*Wi**3.)

        #     elif((Xi>1.83e3)&(Xi<3.46e8)):
        #         Re[i] = 0.4487*(Xi**0.5536)

        #     else:
        #         Re[i] = (Xi/0.6)**(0.5)
            
        #     Cd2[i] = 24./Re[i]

        #     if(Di < 100.e-6):
        #         Cd[i] = 24./Re[i]
                
        #     else:
        #         if(Re[i]<3.e5):
        #             Cd[i] = 0.6
        #         else:
        #             Cd[i] = 0.1

        # if(p==5000.):
        #     fig1 = plt.figure(figsize=(10,10))
        #     plt.loglog(D*100, X, 'k-')
        #     plt.xlabel("Diameter [cm]")
        #     plt.ylabel("Best number")

        #     fig2 = plt.figure(figsize=(10,10))
        #     plt.loglog(D*100, Re, 'k-')
        #     plt.xlabel("Diameter [cm]")
        #     plt.ylabel("Reynolds number")

        #     fig3 = plt.figure(figsize=(10,10))
        #     plt.loglog(Re, Cd, 'k-')
        #     plt.loglog(Re, Cd2, 'k--')
        #     plt.xlabel("Reynolds Number")
        #     plt.ylabel("Drag coefficient")

        #     # fig1.savefig('plots/%s_d_Cd.png'%planet)
        #     # fig2.savefig('plots/%s_d_Re.png'%planet)
        #     # fig3.savefig('plots/%s_Re_Cd.png'%planet)

        #     # plt.close('all')
            

        #     plt.show()    

        return (np.sqrt((4*species.rho_ice*g*D)/(3.*Cd2*rho_air)))

        # return ((1./D)*(dynvisc*Re)/(rho_air))

def cD_rain(planet, species, p, D):
    g = planet.g
    Ratmo = planet.Ratmo
    T = planet.fT(np.log10(p))
    pPa = mbartopa(p)
    rho_air = pPa/(Ratmo*T)
    
    dynvisc_corr = planet.dynvisc_corr
    dynvisc = dynvisc_corr[0] + dynvisc_corr[1]*T
    
    rho_liq = species.rho_liq

    W = np.log10((4. * (D**3.)*rho_air*g*(rho_liq - rho_air))/(3.*dynvisc**2.))
    Re = 10.**(-1.81391 + 1.34671*W - 0.12427*W**2. + 0.006344*W**3.)
    return (10.**W)/Re**2.

# ## Titan data from Lorenz 1993
# planet_data["Titan"] = {"g":1.352, "xi": {"N2":0.942,"H2": 0.001,"CH4": 0.056}, "datfile": "titan_data.csv", "Pref": 1000.}

# ## Venus data from Basilevsky and Head 2003
# planet_data["Venus"] = {"g":8.87, "xi": {"CO2":0.965,"N2": 0.0035}, "datfile": "venus_data.csv", "Pref": 1000.}

## Earth data
# planet_data["Earth"] = {"g":9.81, "xi": {"N2":0.79,"O2": 0.21}, "datfile": "earth_data.csv", "Pref": 1000.}

## Exoplanet data from Wakeford et al. 2016
# planet_data["Exo"] = {"g":10., "xi": {"H2":0.96,"He": 0.04}, "datfile": "exo_data.csv", "Pref": 1000.}