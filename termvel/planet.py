from .utils import *
from .species import Species

## Planet arrays
planet_data = {}
planet_data["jupiter"]  = {"g": 22.31, "xi": {"H2":0.81,"He":0.19}, "datfile": "jupiter_data.csv", "Pref": 1000.}
planet_data["saturn"] = {"g":10.5, "xi":  {"H2":0.96,"He": 0.04}, "datfile": "saturn_data.csv", "Pref": 1000.}

# ## Uranus and Neptune atmos data from Encrenaz 2004 DOI: 10.1007/s11214-005-1950-6
planet_data["uranus"] = {"g":8.69, "xi": {"H2":0.83,"He": 0.15,"CH4": 0.03}, "datfile": "uranus_data.csv", "Pref": 1000.}
planet_data["neptune"] = {"g":11.15, "xi": {"H2":0.79,"He": 0.18,"CH4": 0.03}, "datfile": "neptune_data.csv", "Pref": 1000.}

class Planet(object):
    def __init__(self, planet):
        self.planet_name = planet

        print("Setting up %s"%(planet))
        ## Calculate molar mass of atmosphere from H2 and He concentrations

        self.species = [Species(spec) for spec in planet_data[planet]["xi"].keys()]
        self.Matmo = molarmass([planet_data[planet]["xi"][spec] for spec in planet_data[planet]["xi"].keys()],[compi.mass for compi in self.species])
        self.Ratmo = 8314.4598/self.Matmo

        ## Open the T-P and viscosity file
        file = planet_data[planet]["datfile"]

        data = np.genfromtxt(file,skip_header = 1, delimiter = ",",missing_values = '', filling_values = None)

        ## Clean up missing values
        data = data[np.isfinite(data[:,1])]

        self.g = planet_data[planet]['g']

        if 'H2' in planet_data[planet]['xi']:
            self.H2 = planet_data[planet]['xi']['H2']
            self.He = planet_data[planet]['xi']['He']

        # Set up P with better resolution
        self.P = 10.**np.arange(np.log10(data[0,0]), np.log10(data[-1,0]), 0.01) ## P from 10 mbar to 10 bar

        ## Interpolate T values so we can get better resolution
        ## NOTE: interpolation is done in logP-space for better accuracy
        self.fT = interp1d(np.log10(data[:,0]), data[:,1], kind='cubic')

        ## Find correlation between dynamic viscosity and temperature
        self.dynvisc_corr = self.dynvisc()
        self.mu = self.dynvisc_corr[0] + self.dynvisc_corr[1]*self.fT(np.log10(self.P))

        ## Save minimum and maximum values of P used in interpolation
        self.Prange = np.asarray([data[0,0],data[-1,0]])

    def dynvisc(self):
        ## C Palotai's code from EPIC model (epic_microphysics_funcs.c)
        ## Ref - Jupiter: C. F. Hansen (1979)
        ## planet_data is global dictionary of planet parameters
        ## planet is string with the name of the planet
        
        
        ## Titan viscosity from Lorenz (1993) Equation A1
        
        ## Venus viscosity from Petropoulos 1988
        
        if(self.planet_name == "Titan"):
            return([1.718e-5 - 5.1e-8*273.,5.1e-8])
        elif(self.planet_name == "Venus"):
            return([4.46e-6, 3.64e-8])
        elif(self.planet_name == "Earth"):
            return([2.95e-6, 5.206e-8])
        else:
            ## Fetch H2 and He concentrations
            x1 = self.H2
            x2 = self.He
            
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
