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