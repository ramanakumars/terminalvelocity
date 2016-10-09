# Terminal Velocity 
Terminal velocity calculations for Jupiter cloud modelling

## Log
**07 Jul 2016** *ramana*
> Implemented importing of T-P and mu-P profiles and interpolation. 

> Added code for calculating dynamic viscosity (from EPIC) - **WIP**

**08 Jul 2016** *ramana*
> Implemented terminal velocity calculations at various P, D, Planet, Species values

> TODO: Graphing the output and comparing with Palotai2008

**13 Jul 2016** *ramana*
> Setup scipy.optimize.curve_fit to calculate fit parameters for power law

**29 Jul 2016** *ramana*
> Modified vT calculation function to account for ice mass calculation

> Printing the x, y and gamma values for Jupiter for comparison

**09 Aug 2016** *ramana*
> Added parallel processing for speeeeeed 

**25 Aug 2016** *ramana*
> Updated T-P profile for Saturn from Lindal 1985

> *triplepoint*: Added triplepoint reference pressure calculation 

> *triplepoint*: Added csv saving

**17 Sep 2016** *ramana*
> Merged triplepoint with master

> Added new comments and marked the triplepoint calculations as "Deprecated"

**03 Oct 2016** *ramana*
> *plot* Added Uranus

> *plot* Removed ant_liq, ant_ice and q values for planets

> *plot* Removed DEPCRECATED tags


**07 Oct 2016** *ramana*
> *plot* Added H2, He and CH4 ratios and created non-condensing species data

> *plot* Updated Jupiter data

> *plot* Added other plots

**08 Oct 2016** *ramana*
> *plot* Added planet specific species


**09 Oct 2016** *ramana*
> *plot* Added Venus and H2SO4