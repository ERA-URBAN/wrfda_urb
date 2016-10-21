import numpy

### LECH'S SURFACE FUNCTIONS
def PSLMU(ZZ):
  return -0.96* numpy.log (1.0-4.5* ZZ)
def PSLMS(ZZ):
  RIC = 0.183
  RRIC = 1.0/ RIC
  return ZZ * RRIC -2.076* (1. -1./ (ZZ +1.))
def PSLHU(ZZ):
  return -0.96* numpy.log (1.0-4.5* ZZ)
### /LECH'S SURFACE FUNCTIONS


### PAULSON'S SURFACE FUNCTIONS
def PSLHS(ZZ):
  RFC = 0.191
  FHNEU = 0.8
  RIC = 0.183  
  RFAC = RIC / (FHNEU * RFC * RFC)
  return ZZ * RFAC -2.076* (1. -1./ (ZZ +1.))
def PSPMU(XX):
  PIHF = 3.14159265/2.
  return (-2.* numpy.log((XX +1.)*0.5) -
          numpy.log((XX * XX +1.)*0.5) + 2.* numpy.arctan(XX)- PIHF)
def PSPMS(YY):
  return 5.* YY
def PSPHU(XX):
  return -2.* numpy.log ( (XX * XX +1.)*0.5)
### /PAULSON'S SURFACE FUNCTIONS

def PSPHS(YY):
  return 5.* YY

def sfcdif_urb(ZLM,Z0,THZ0,THLM,SFCSPD,AKANDA,AKMS,AKHS,RLMO):
  '''
  CALCULATE SURFACE LAYER EXCHANGE COEFFICIENTS VIA ITERATIVE PROCESS.
  SEE CHEN ET AL (1997, BLM)
  '''
  # define constants
  WWST = 1.2
  WWST2 = WWST * WWST
  G = 9.8
  VKRM = 0.40
  EXCM = 0.001
  BETA = 1./270.
  BTG = BETA * G
  ELFC = VKRM * BTG
  WOLD =.15
  WNEW = 1. - WOLD
  ITRMX = 5
  EPSU2 = 1.E-4
  EPSUST = 0.07
  EPSIT = 1.E-4
  EPSA = 1.E-8
  ZTMIN = -5.
  ZTMAX = 1.
  HPBL = 1000.0
  SQVISC = 258.2
  RLMO_THR = 0.001
  ILECH = 0
  ZU = Z0
  RDZ = 1./ ZLM
  CXCH = EXCM * RDZ
  DTHV = THLM - THZ0

  # BELJARS CORRECTION OF USTAR
  DU2 = numpy.maximum(SFCSPD * SFCSPD,EPSU2)
  # If statements to avoid TANGENT LINEAR problems near zero
  BTGH = BTG * HPBL
  
  boolean = (BTGH * AKHS * DTHV != 0.0)
  WSTAR2 = numpy.zeros(numpy.shape(DTHV))
  WSTAR2[boolean] = (WWST2* abs(BTGH * AKHS * DTHV)** (2./3.))[boolean]
  WSTAR2[~boolean] = 0.0

  # ZILITINKEVITCH APPROACH FOR ZT
  USTAR = numpy.maximum(numpy.sqrt(AKMS * numpy.sqrt(DU2+ WSTAR2)),EPSUST)

  # KCL/TL Try Kanda approach instead (Kanda et al. 2007, JAMC)
  ZT = numpy.exp(2.0-AKANDA*(SQVISC**2 * USTAR * Z0)**0.25)* Z0
  ZSLU = ZLM + ZU
  ZSLT = ZLM + ZT
  RLOGU = numpy.log (ZSLU / ZU)
  RLOGT = numpy.log (ZSLT / ZT)
  RLMO = ELFC * AKHS * DTHV / USTAR **3

  # 1./MONIN-OBUKKHOV LENGTH-SCALE
  # initialize
  SIMM = numpy.zeros(numpy.shape(ZSLT))
  PSHZ = numpy.zeros(numpy.shape(ZSLT))
  SIMH = numpy.zeros(numpy.shape(ZSLT))
  for ITR in range(0,ITRMX):
    ZETALT = numpy.maximum(ZSLT * RLMO, ZTMIN)
    RLMO = ZETALT / ZSLT
    ZETALU = ZSLU * RLMO
    ZETAU = ZU * RLMO
    ZETAT = ZT * RLMO
    if (ILECH == 0):
      unstable = RLMO < 0.0  # check for stability
      XLU4 = 1. -16.* ZETALU
      XLT4 = 1. -16.* ZETALT
      XU4 = 1. -16.* ZETAU
      XT4 = 1. -16.* ZETAT
      XLU = numpy.sqrt(numpy.sqrt(XLU4))
      XLT = numpy.sqrt(numpy.sqrt(XLT4))
      XU = numpy.sqrt(numpy.sqrt(XU4))
      XT = numpy.sqrt(numpy.sqrt(XT4))
      PSMZ = (PSPMU(XU))
      SIMM[unstable] = (PSPMU(XLU) - PSMZ + RLOGU)[unstable]
      PSHZ[unstable] = (PSPHU(XT))[unstable]
      SIMH[unstable] = (PSPHU(XLT) - PSHZ + RLOGT)[unstable]
      ZETALU = numpy.minimum(ZETALU,ZTMAX)
      ZETALT = numpy.minimum(ZETALT,ZTMAX)
      PSMZ = PSPMS(ZETAU)
      SIMM[~unstable] = (PSPMS(ZETALU) - PSMZ + RLOGU)[~unstable]
      PSHZ[~unstable] = PSPHS(ZETAT)[~unstable]
      SIMH[~unstable] = (PSPHS(ZETALT) - PSHZ + RLOGT)[~unstable]
    else:
      # LECH's functions
      # unstable
      unstable = RLMO < 0  # check for stability
      PSMZ = PSLMU(ZETAU)[unstable]
      SIMM = (PSLMU(ZETALU) - PSMZ + RLOGU)[unstable]
      PSHZ = (PSLHU(ZETAT))[unstable]
      SIMH = (PSLHU(ZETALT) - PSHZ + RLOGT)[unstable]
      # stable
      ZETALU = numpy.minimum(ZETALU,ZTMAX)
      ZETALT = numpy.minimum(ZETALT,ZTMAX)
      PSMZ = PSLMS(ZETAU)[~unstable]
      SIMM = (PSLMS(ZETALU) - PSMZ + RLOGU)[~unstable]
      PSHZ = (PSLHS(ZETAT))[~unstable]
      SIMH = (PSLHS(ZETALT) - PSHZ + RLOGT)[~unstable]
    # BELJAARS CORRECTION FOR USTAR
    USTAR = numpy.maximum(numpy.sqrt(AKMS * numpy.sqrt(DU2+ WSTAR2)),EPSUST)
    ZT = numpy.exp(2.0-AKANDA*(SQVISC**2 * USTAR * Z0)**0.25)* Z0
    ZSLT = ZLM + ZT
    RLOGT = numpy.log(ZSLT / ZT)
    USTARK = USTAR * VKRM
    AKMS = numpy.maximum(USTARK / SIMM,CXCH)
    AKHS = numpy.maximum(USTARK / SIMH,CXCH)

    RLMN = ELFC * AKHS * DTHV / USTAR **3
    RLMA = RLMO * WOLD+ RLMN * WNEW
    RLMO = RLMA

  CD = USTAR*USTAR/SFCSPD**2
  
  return CD
