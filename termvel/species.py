from .utils import *

## Species arrays
## ant_ice and ant_liq are unnecessary and exist in case we need them for another calculation
spec_data = {}
spec_data["H2O"] = {"mass":18., "rho_ice": 900.0, "rho_liq": 1000., "A_Ae_snow": 1.05, "A_Ae_ice": 1., "docalc":True,"planet":["Earth","Jupiter","Saturn"]}
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
spec_data["O2"] = {"mass": 16., "docalc": False}
spec_data["He"] = {"mass": 4., "docalc": False}
spec_data["CO2"] = {"mass": 38., "docalc": False}
spec_data["N2"] = {"mass": 28., "docalc": False}

class Species(object):
    def __init__(self, species):

        self.species_name = species

        self.mass   = spec_data[species]['mass']
        self.docalc = spec_data[species]['docalc']
        if self.docalc:
            self.rho_ice = spec_data[species]['rho_ice']
            self.rho_liq = spec_data[species]['rho_liq']
            
            self.A_Ae_snow = spec_data[species]['A_Ae_snow']
            self.A_Ae_ice  = spec_data[species]['A_Ae_ice']

            self.rho_ice_scale = self.rho_ice/spec_data['H2O']['rho_ice']

            self.planets = spec_data[species]['planet']
