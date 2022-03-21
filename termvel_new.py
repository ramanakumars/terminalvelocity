from termvel.utils import *
from termvel.planet import Planet
from termvel.species import Species
from termvel.fitter import TermVelFitter

## textsizes
textsize = 18
labelsize = 18

## Initialize LateX
plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.rc('font', size=textsize)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

# params = {'xtick.labelsize':labelsize,'ytick.labelsize':labelsize}
# plt.rcParams.update(params)

plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'k','c','m'])))

## Define whitespaces
left = 0.15
right = 0.98
top = 0.98
bottom = 0.15
wspace = 0.08
hspace = 0.17

jup     = Planet('jupiter')
sat     = Planet('saturn')
uranus  = Planet('uranus')
neptune = Planet('neptune')

H2O = Species('H2O')
NH3 = Species('NH3')
CH4 = Species('CH4')

jupH2Ofitter = TermVelFitter(jup, H2O)
jupH2Ofitter.set_prange()
jupH2Ofitter.get_tvel_fit()

satH2Ofitter = TermVelFitter(sat, H2O)
satH2Ofitter.set_prange()
satH2Ofitter.get_tvel_fit()

jupNH3fitter = TermVelFitter(jup, NH3)
jupNH3fitter.set_prange()
jupNH3fitter.get_tvel_fit()

satNH3fitter = TermVelFitter(sat, NH3)
satNH3fitter.set_prange()
satNH3fitter.get_tvel_fit()

urCH4fitter = TermVelFitter(uranus, CH4)
urCH4fitter.set_prange()
urCH4fitter.get_tvel_fit()

nepCH4fitter = TermVelFitter(neptune, CH4)
nepCH4fitter.set_prange()
nepCH4fitter.get_tvel_fit()

# jupH2Ofitter.create_plot(5000.)


fig, axs = plt.subplots(1, 3, figsize=(8,4), dpi=150, sharex=True)

axs[0].plot(jupH2Ofitter.vT['ice']['D']*1.e3, jupH2Ofitter.get_vT('ice', 5000., jupH2Ofitter.vT['ice']['D']), 'k-', label='Jupiter ice')
axs[0].plot(satH2Ofitter.vT['ice']['D']*1.e3, satH2Ofitter.get_vT('ice', 5000., satH2Ofitter.vT['ice']['D']), 'r-', label='Saturn ice')

axs[0].plot(jupH2Ofitter.vT['snow']['D']*1.e3, jupH2Ofitter.get_vT('snow', 5000., jupH2Ofitter.vT['snow']['D']), 'k--', label='Jupiter snow')
axs[0].plot(satH2Ofitter.vT['snow']['D']*1.e3, satH2Ofitter.get_vT('snow', 5000., satH2Ofitter.vT['snow']['D']), 'r--', label='Saturn snow')

axs[1].plot(jupNH3fitter.vT['ice']['D']*1.e3, jupNH3fitter.get_vT('ice', 5000., jupNH3fitter.vT['ice']['D']), 'k-')
axs[1].plot(satNH3fitter.vT['ice']['D']*1.e3, satNH3fitter.get_vT('ice', 5000., satNH3fitter.vT['ice']['D']), 'r-')

axs[1].plot(jupNH3fitter.vT['snow']['D']*1.e3, jupNH3fitter.get_vT('snow', 5000., jupNH3fitter.vT['snow']['D']), 'k--')
axs[1].plot(satNH3fitter.vT['snow']['D']*1.e3, satNH3fitter.get_vT('snow', 5000., satNH3fitter.vT['snow']['D']), 'r--')

axs[2].plot(urCH4fitter.vT['ice']['D']*1.e3, urCH4fitter.get_vT('ice', 1000., urCH4fitter.vT['ice']['D']), '-', color='#008b8b')
axs[2].plot(nepCH4fitter.vT['ice']['D']*1.e3, nepCH4fitter.get_vT('ice', 1000., nepCH4fitter.vT['ice']['D']), '-', color='darkblue')

axs[2].plot(urCH4fitter.vT['snow']['D']*1.e3, urCH4fitter.get_vT('snow', 1000., urCH4fitter.vT['snow']['D']), '--', color='#008b8b')
axs[2].plot(nepCH4fitter.vT['snow']['D']*1.e3, nepCH4fitter.get_vT('snow', 1000., nepCH4fitter.vT['snow']['D']), '--', color='darkblue')

axs[0].set_title(r'H$_2$O')
axs[1].set_title(r'NH$_3$')
axs[2].set_title(r'CH$_4$')

# axs[0].set_xscale('log')
axs[0].set_xlabel(r'Diameter [mm]')
axs[1].set_xlabel(r'Diameter [mm]')
axs[2].set_xlabel(r'Diameter [mm]')
axs[0].set_ylabel(r'Terminal velocity [m/s]')

plt.tight_layout()
plt.show()