import numpy

def multi_layer(KM,BOUND,G0,CAP,AKS,TSL,DZ,DELT):
  DZEND=DZ[KM - 1]
  TSLEND = TSL[KM - 1]

  # initialize
  A = numpy.zeros((KM,) + numpy.shape(G0))
  B = numpy.zeros((KM,) + numpy.shape(G0))
  C = numpy.zeros((KM,) + numpy.shape(G0))
  D = numpy.zeros((KM,) + numpy.shape(G0))
  P = numpy.zeros((KM,) + numpy.shape(G0))
  Q = numpy.zeros((KM,) + numpy.shape(G0))
  X = numpy.zeros((KM,) + numpy.shape(G0))
  A[0,:] = 0.0
  B[0,:] = CAP*DZ[0]/DELT +2.*AKS/(DZ[0]+DZ[1])
  C[0,:] = -2.*AKS/(DZ[0]+DZ[1])
  D[0,:] = CAP*DZ[0]/DELT*TSL[0] + G0

  for K in range(1,KM-1):
      A[K,:] = -2.*AKS/(DZ[K-1]+DZ[K])
      B[K,:] = CAP*DZ[K]/DELT + 2.*AKS/(DZ[K-1]+DZ[K]) + 2.*AKS/(DZ[K]+DZ[K+1])
      C[K,:] = -2.*AKS/(DZ[K]+DZ[K+1])
      D[K,:] = CAP*DZ[K]/DELT*TSL[K]
  if (BOUND == 1):  # Flux=0
    A[KM-1] = -2.*AKS/(DZ[KM-2]+DZ[KM-1])
    B[KM-1] = CAP*DZ[KM-1]/DELT + 2.*AKS/(DZ[KM-2]+DZ[KM-1])
    C[KM-1] = 0.0
    D[KM-1] = CAP*DZ[KM-1]/DELT*TSL[KM-1]
  else:  # T=constant
    A[KM-1] = -2.*AKS/(DZ[KM-2]+DZ[KM-1])
    B[KM-1] = CAP*DZ[KM-1]/DELT + 2.*AKS/(DZ[KM-2]+DZ[KM-1]) + 2.*AKS/(DZ[KM-1]+DZEND)
    C[KM-1] = 0.0
    D[KM-1] = CAP*DZ[KM-1]/DELT*TSL[KM-1] + 2.*AKS*TSLEND/(DZ[KM-1]+DZEND)

  P[0] = -C[0]/B[0]
  Q[0] =  D[0]/B[0]

  for K in range(1,KM):
    P[K] = -C[K]/(A[K]*P[K-1]+B[K])
    Q[K] = (-A[K]*Q[K-1]+D[K])/(A[K]*P[K-1]+B[K])

  X[KM-1] = Q[KM-1]

  for K in range(KM-2,-1,-1):  # check if this is correct
    X[K] = P[K]*X[K+1]+Q[K]

  for K in range(0,KM):
    TSL[K] = X[K]
  return TSL
