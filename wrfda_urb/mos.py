import numpy


def mos(B1,RIB,Z,Z0,UA,TA,TSF,RHO):
  # XXX:   z/L (requires iteration by Newton-Rapson method)
  # B1:    Stanton number
  # PSIM:  = PSIX of LSM
  # PSIH:  = PSIT of LSM
  # initialize
  XXX = numpy.zeros(numpy.shape(Z))
  PSIM = numpy.zeros(numpy.shape(Z))
  PSIH = numpy.zeros(numpy.shape(Z))
  
  CP = 0.24  # constant
  NEWT_END = 10

  boolean = (RIB <= -15.)
  RIB[boolean] = -15
  boolean = (RIB < 0.)
#  if (RIB < 0.):
  for NEWT in range(1, NEWT_END):
    xxx_bool = (XXX >= 0.)
    XXX[xxx_bool] = -1e-3
    XXX0=XXX*Z0/(Z+Z0)
    X=(1.-16.*XXX)**0.25
    X0=(1.-16.*XXX0)**0.25
    PSIM[boolean] = (numpy.log((Z+Z0)/Z0) -
                     numpy.log((X+1.)**2.*(X**2.+1.)) +
                     2.*numpy.arctan(X) + numpy.log((X+1.)**2.*(X0**2.+1.)) -
                     2.*numpy.arctan(X0))[boolean]
    FAIH = 1.0 /numpy.sqrt(1.0-16.0*XXX)
    PSIH[boolean] = (numpy.log((Z+Z0)/Z0)+0.4*B1 -
                     2.*numpy.log(numpy.sqrt(1.-16.*XXX)+1.) +
                     2.*numpy.log(numpy.sqrt(1.-16.*XXX0)+1.))[boolean]
    DPSIM = ((1.-16.*XXX)**(-0.25)/XXX -
             (1.-16.*XXX0)**(-0.25)/XXX)
    DPSIH = (1./numpy.sqrt(1.-16.*XXX)/XXX -
             1./numpy.sqrt(1.-16.*XXX0)/XXX)
    F=RIB*PSIM**2./PSIH-XXX
    DF=RIB*(2.*DPSIM*PSIM*PSIH-DPSIH*PSIM**2.) /PSIH**2.-1.
    XXXP=XXX
    XXX[boolean]=(XXXP-F/DF)[boolean]
    xxx_bool = (XXX <= -10.)
    XXX[xxx_bool] = -10

  boolean = (RIB >= 0.142857)
  XXX[boolean] = 0.714
  PSIM[boolean] = (numpy.log((Z+Z0)/Z0)+7.*XXX)[boolean]
  PSIH[boolean] = (PSIM+0.4*B1)[boolean]
  
 # else:
  boolean = ((RIB< 0.142857) & (RIB>=0))
  AL = numpy.log((Z+Z0)/Z0)
  XKB = 0.4*B1
  DD = -4.*RIB*7.*XKB*AL+(AL+XKB)**2.
  dd_bool = (DD <= 0.)
  DD[dd_bool] = 0.

  XXX[boolean] = ((AL+XKB-2.*RIB*7.*AL-numpy.sqrt(DD))/(2.*(RIB*7.**2-7.)))[boolean]
  PSIM[boolean] = (numpy.log((Z+Z0)/Z0)+7.*numpy.minimum(XXX,0.714))[boolean]
  PSIH[boolean] = (PSIM+0.4*B1)[boolean]

  US = 0.4*UA/PSIM             # u*
  us_bool = (US <= 0.01) 
  US[us_bool] = 0.01
  TS = 0.4*(TA-TSF)/PSIH       # T*
  CD = US*US/UA**2.            # CD
  ALPHA = RHO*CP*0.4*US/PSIH   # RHO*CP*CH*U

  return ALPHA, CD, XXX, RIB
