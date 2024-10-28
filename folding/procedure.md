Once a multinest run is done. Check for the MAP (i.e. Maximum aposteriori parameters)

This is available as one of the dataproducts ending with "stats.dat". Below is an example

```
MAP Parameters
Dim No.        Parameter
   1    0.348726662281483080E+00
   2   -0.159784384814400449E+01
   3    0.335434459495723971E-02
   4   -0.200881209615878475E+01
   5    0.367613419671191766E+01
   6   -0.692578383166166045E+00
   7    0.476741532517266364E+00
   8    0.335433607202673730E-02
   9    0.464698486456038909E-01
  10    0.585074928875855660E+00
  11    0.202965216724130870E+00
```
The above parameters reads as:

1. Rotational Phase ($\phi$; Ranging from 0-1)
2. Pulse Width ($W_{log_{10}}$) 
3. Spin period ($P$; s)
4. log\_10 projected semi major axis($x_{log_{10}}$; lt-s)
5. Binary phase ($\psi_{2\pi}$; between 0 to 2$\pi$)
6. log10 binary period (log $P_b$; days)
7. Rotational Phase corrected ($\phi_{corr}$; 0-1)
8. Spin period corrected ($P\_{corr}$; s)
9. Projected semi-major axis ($x$; lt-s)
10. Binary phase ($\psi$; between 0 and 1)
11. Binary period ($P_b$; days) 


To calculate T0:


Use RefMJD of the first dedispersed time series file and apply the following

```
T0\_corrected = T0\_ref - $\frac{\psi_{2\pi}}{2\pi}$ * P_b$
```

apply the following convention when folding with prepfold:

```
prepfold -p $P_corr$ -bin -pb $P_b$ * 86400 -x $x$ -To T0\_corrected *.fil  
```

