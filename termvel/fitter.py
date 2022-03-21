from venv import create
from .utils import *
from .planet import Planet
from .species import Species

## Setup pressure intervals for vT calculation
P = {}
Pref = 100.

P["H2O"] = {}
P["H2O"]["earth"] = np.arange(500.,1000.,100.)
P["H2O"]["jupiter"] = np.arange(1000.,20000.,100.)
P["H2O"]["saturn"] = np.arange(1000.,5000.,100.)
P["NH3"] = np.arange(400.,1000.,100.)
P["CH4"] = np.arange(600.,2000.,100.)
P["NH4SH"] = {}
P["NH4SH"]["jupiter"] = np.arange(2000.,4000.,100.)
P["NH4SH"]["saturn"] = np.arange(6000.,8000.,100.)
P["H2S"] = {}
P["H2S"]["uranus"] = np.arange(3000.,5000.,100.)
P["H2S"]["neptune"] = np.arange(8000.,12000.,100.)
P["H2SO4"] = np.arange(600., 2000., 100.)
P["Fe"] = np.arange(1., 100., 5.)

phases = ["rain","ice","snow"]
colors = {"rain": 'b', "ice": 'g', "snow": 'y'}
Pref   = 1000.

def vT_fit (x, a, b):
	return (a + b*x)

		
def vT_fit_old(x, a, b, c):
	return( a + b*x[:,0] + c*x[:,1])

class TermVelFitter(object):
    def __init__(self, planet, species):
        self.planet  = planet
        self.species = species

        self.vT = {"rain": {}, "ice": {}, "snow": {}}

        self.gamma = {"rain": 0., "ice": 0., "snow": 0.}
        self.x     = {"rain": 0., "ice": 0., "snow": 0.}
        self.y     = {"rain": 0., "ice": 0., "snow": 0.}

    def set_prange(self, Prange=None):
        if Prange is None:
            if(self.planet.planet_name=="Titan"):
                Prange = np.linspace(np.min(P[self.species.species_name]),np.max(self.planet.Prange),100.)
            elif(self.species.species_name == "H2S"):
                Prange = P["H2S"][self.planet.planet_name]
            elif(self.species.species_name == "NH4SH"):
                Prange = P["NH4SH"][self.planet.planet_name]
            elif(self.species.species_name == "H2O"):
                Prange = P["H2O"][self.planet.planet_name]
            else:
                Prange = P[self.species.species_name]

        self.Prange = Prange

    def get_tvel_fit(self):
        for phase in phases:
            print(f"Planet: {self.planet.planet_name} Phase: {phase}")
            ## setup particle sizes
            ## different sizes for each phase

            if not hasattr(self, 'Prange'):
                raise RuntimeError("Please call set_prange first")

            Prange = self.Prange
            
            if not Pref in Prange:
                Prange = np.append(Prange,Pref)

            
            if(phase == "rain"):
                D = np.linspace(20.e-6, 5.e-3, 1000)
            elif(phase == "ice"):
                D = np.linspace(0.1e-6, 500.e-6, 1000)
            elif(phase == "snow"):
                D = np.linspace(0.5e-3, 5.e-3, 1000)
                
            self.vT[phase] = {}
            self.vT[phase]["D"] = D
            self.vT[phase]["p"] = Prange
            
            print("\tCalculating terminal velocity profile")
            
            ## Initiaze the vT array with empty array
            for p in Prange:
                self.vT[phase]['vT'] = np.zeros((len(Prange), len(D)))
            
            
            ## This is formatting the vT data to save as .csv
            ## In case we need to see the raw data
            vT_dat = np.zeros((len(D),2*len(Prange)+2))
            header = ""
            for i, p in enumerate(Prange):
                vT_d = vT_calc(self.planet, self.species, phase, p, D)
                self.vT[phase]['vT'][i] = vT_d
                header = header + "P=%04d,vT,"%(int(p))				
                vT_dat[:,2*i] = D
                vT_dat[:,2*i+1] = vT_d
            
            ## save vT data to csv
            outname = "data/%s_%s_%s.csv"%(self.planet.planet_name,self.species.species_name,phase)
            np.savetxt(outname,vT_dat,header=header,delimiter=",")

            ## Convert 2D array of pressure and particle size to 1D (xData)
            ## Also convert vT to 1D array (zData)
            ## For fitting
            xData = []
            zData = []
            
            print("\tCreating array to fit")
            for i in range(len(Prange)):
                for j in range(len(D)):
                    #index = (i*len(Prange)) + j
                    xData.append([D[j],(Pref/Prange[i])])
                    zData.append(self.vT[phase]['vT'][i,j])
            xData = np.asarray(xData)
            zData = np.asarray(zData)

            print("\tFitting")
            par, cov = curve_fit(vT_fit_old, np.log10(xData),np.log10(zData))
            
            ## Save the parameters
            self.x[phase] = 10.**par[0]
            self.y[phase] = par[1]
            self.gamma[phase] = par[2]

    def get_vT(self, phase, p, D):
        return self.vT[phase]['vT'][np.argmin((self.vT[phase]['p']-p)**2.),:]

    def get_vT_fit(self, phase, p, D):
        x = self.x[phase]
        y = self.y[phase]
        g = self.gamma[phase]

        return x*(D**y)*(Pref/p)**g

    def create_plot(self, phases=['rain','snow','ice'], p=1000., ax=None):
        create_plot = False
        if ax is None:
            create_plot = True
            fig, ax = plt.subplots(1,1)

        for phase in phases:
            Pi = np.argmin((self.vT[phase]['p']-p)**2.)

            Di = self.vT[phase]['D']

            ax.plot(Di*1.e6, self.vT[phase]['vT'][Pi,:], '.', color=colors[phase])
            
            ax.plot(Di*1.e6, self.x[phase]*(Di**self.y[phase])*(Pref/p)**self.gamma[phase], '-', color=colors[phase])

            if create_plot:
                ax.set_xscale('log')
                # ax.set_yscale('log')

                ax.set_xlabel(r'Diameter [$\mu$m]')
                ax.set_ylabel(r'Terminal velocity [m/s]')

        if create_plot:
            plt.tight_layout()
            plt.show()