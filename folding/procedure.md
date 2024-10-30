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


The above parameters correspond to:

1. Rotational Phase ($\phi$; ranging from 0 to 1)
2. Pulse Width ($W_{\log_{10}}$)
3. Spin Period ($P$; seconds)
4. $\log_{10}$ of the projected semi-major axis ($x_{\log_{10}}$; light-seconds)
5. Binary phase ($\psi_{2\pi}$; between 0 and $2\pi$)
6. $\log_{10}$ of the binary period ($P_b$(days); days)
7. Corrected Rotational Phase ($\phi_{\text{corr}}$; 0 to 1)
8. Corrected Spin Period ($P_{\text{corr}}$; seconds)
9. Projected semi-major axis ($x$; light-seconds)
10. Binary phase ($\psi$; between 0 and 1)
11. Binary period ($P_b$; days)

### To calculate $T_0$:

Use the reference MJD of the first dedispersed time series file and apply the following formula:


$T_0^{\text{corrected}} = T_0^{\text{ref}} - \left(\frac{\psi_{2\pi}}{2\pi}\right) \cdot P_b$

### To calculate $P_b$:

$P_b$ = $P_b$(days) * 86400


Apply the following convention when folding with prepfold:


prepfold  -p $P_{\text{corr}}$  -bin  -pb  $P_b$  -x  $x$  -To   $T_0^{\text{corrected}}$  \<time ordered filterbanks\> 



