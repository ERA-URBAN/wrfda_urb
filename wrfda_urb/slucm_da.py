#!/usr/bin/env python

# License: Apache 2.0
# Author: Ronald van Haren

# implement slucm subroutine for data assimilation on WRF slucm

from wrfda_urb.mos import mos
from wrfda_urb.multi_layer import multi_layer
from wrfda_urb.sfcdif_urb import sfcdif_urb
from netCDF4 import Dataset
import numpy
from matplotlib import *
from pylab import *


def force_restore(CAP,AKS,DELT,S,R,H,LE,TSLEND,TSP):
  C2=24.*3600./2./pi
  C1=numpy.sqrt(0.5*C2*CAP*AKS)
  TS = TSP + DELT*( (S+R-H-LE)/C1 -(TSP-TSLEND)/C2 )
  return TS


class slucm:
  def __init__(self, netcdffile, wrfout):
    self.set_constants()
    self.read_netcdf(netcdffile, wrfout)
    self.urbparam_init()
    self.canopy_wind()
    self.net_shortwave_ratiation()
    for ii in range(0,2):
      self.roof()
      self.wall_road()
      self.total_fluxes_urban_canopy()


  def read_netcdf(self, netcdffile, wrfout):
    ncfile = Dataset(netcdffile, 'r')
    ncfile2 = Dataset(wrfout, 'r')
    self.landuse = ncfile.variables['LU_INDEX'][0,:]
    self.urban = numpy.where(self.landuse == 1)
    self.QA = ncfile.variables['QVAPOR'][0,0,:][self.urban]
    self.UA = numpy.sqrt((ncfile.variables['U'][0,1,:,:-1][self.urban]**2) +
                         (ncfile.variables['V'][0,1,:-1,:][self.urban]**2))
    self.SSG = ncfile2.variables['SWDOWN'][1,:][self.urban]
    self.LLG = ncfile2.variables['GLW'][1,:][self.urban]
    self.TRP = ncfile.variables['TR_URB'][0,:][self.urban]
    self.TGP = ncfile.variables['TG_URB'][0,:][self.urban]
    self.TCP = ncfile.variables['TC_URB'][0,:][self.urban]
    self.TBP = ncfile.variables['TB_URB'][0,:][self.urban]
    self.QCP = ncfile.variables['QC_URB'][0,:][self.urban]
    self.Hin = ncfile2.variables['SH_URB'][1,:][self.urban]
    self.PS = ncfile.variables['PSFC'][0,:][self.urban]
    # total potential temperature = T+300 WRF ARW user guide
    self.TA = (ncfile.variables['T'][0,1,:][self.urban] + 300)
    self.STDH_URB = ncfile.variables['STDH_URB2D'][0,:][self.urban]
    self.RAIN = numpy.zeros(numpy.shape(self.LLG))  # TODO: fix
    self.RHOO = 1.225 * numpy.ones(numpy.shape(self.LLG))
    self.RHO = self.RHOO * 0.001  # density of air [g/cm3]
    self.ZA = (ncfile.variables['PH'][0,1,:][self.urban])/9.81 + (ncfile.variables['PHB'][0,1,:][self.urban])/9.81
    #self.COSZ = ncfile.variables['COSZEN'][0,:][self.urban]
    self.DELT = 60
    self.ZR = ncfile.variables['MH_URB2D'][0,:][self.urban]
    self.LSOLAR = False
    self.UTYPE = 4
    self.TRLEND = 293
    self.TBLEND = 293
    self.TGLEND = 293
    self.num_roof_layers = 4
    self.num_wall_layers = 4
    self.num_road_layers = 4
    self.DZR = [5.,5.,5.,5.]
    self.DZB = [5.,5.,5.,5.]
    self.DZG = [5.,20.,25.,25.]
    self.ANTHEAT = 0.
    self.TRL = [ncfile.variables['TRL_URB'][0,:][idx][self.urban]
                for idx in range(0,4)]
    self.TBL = [ncfile.variables['TBL_URB'][0,:][idx][self.urban]
                for idx in range(0,4)]
    self.TGL = [ncfile.variables['TGL_URB'][0,:][idx][self.urban]
                for idx in range(0,4)]
    self.CMR_URB = ncfile2.variables['CMR_SFCDIF'][1,:][self.urban]
    self.CHR_URB = ncfile2.variables['CHR_SFCDIF'][1,:][self.urban]
    self.CMC_URB = ncfile2.variables['CMC_SFCDIF'][1,:][self.urban]
    self.CHC_URB = ncfile2.variables['CHC_SFCDIF'][1,:][self.urban]
    self.Lambda_P = ncfile.variables['BUILD_AREA_FRACTION'][0,:][self.urban]
    self.urbfrac = ncfile.variables['FRC_URB2D'][0,:][self.urban]
    self.Lambda_F = [ncfile.variables['LF_URB2D'][0,:][idx][self.urban]
                     for idx in range(0,4)]/self.urbfrac
    self.stdh_urb = ncfile.variables['STDH_URB2D'][0,:][self.urban]
    self.XXXB = numpy.zeros(numpy.shape(self.ZR))  # update registry
    self.XXXG = numpy.zeros(numpy.shape(self.ZR)) # update registry
    self.TC2Min = ncfile2.variables['TC2M_URB'][1,:]
    self.lb_urb = ncfile.variables['BUILD_SURF_RATIO'][0,:][self.urban]
    ncfile.close()


  def set_constants(self):
    self.CP = 0.24  # heat capacity of dry air
    self.EL = 583.  # latent heat of vaporation
    self.SIG = 8.17e-11  # stefan bolzman constant
    self.SIG_SI = 5.67e-8
    self.AK = 0.4  # kalman constant
    self.TETENA = 7.5  # constant of Tetens equation
    self.TETENB = 237.3  # constant of Tetens equation
    self.SRATIO = 0.75  # ratio betwee direct/total solar
    self.CPP = 1004.5  # heat capacity of dry air
    self.ELL = 2.442e06  # latent heat of vaporiztion
    self.XKA = 2.4e-5
    self.VKMC = 0.4  # Von Karman constant
    self.ALBR = 0.20
    self.ALBV = 0.20
    self.ALBG = 0.20
    self.ALBB = 0.20
    self.road_width = 9.4  # TODO: calculate
    self.roof_width = 9.4  # TODO: calculate
    self.beta_macd = 1
    self.alpha_macd = 4.43
    self.Cd = 1.2
    self.AKANDA_URBAN = 1.29  # get from urbparam
    self.EPSR = 0.9 # get from urbparam
    self.EPSB = 0.9 # get from urbparam
    self.EPSG = 0.95 # get from urbparam
    self.AKSR = 1.01 # get from urbparam
    self.AKSB = 1.01 # get from urbparam
    self.AKSG = 0.4004 # get from urbparam
    self.BOUNDR = 1 # get from urbparam
    self.BOUNDB = 1 # get from urbparam
    self.BOUNDG = 1 # get from urbparam
    self.CAPR = 1.0e6 # get from urbparam
    self.CAPB = 1.0e6 # get from urbparam
    self.CAPG = 1.4e6 # get from urbparam
    self.Z0B = 0.0001 # get from urbparam
    self.Z0G = 0.01 # get from urbparam
    self.Z0HB = 0.1 * self.Z0B # get from urbparam
    self.Z0HG = 0.1 * self.Z0G # get from urbparam
    self.ahoption=0
    self.alhoption=0

  def return_original_shape(self, values, indices, shape):
    '''
    desc
    '''
    out = numpy.zeros(shape)
    out[indices] = values
    return out

  def canopy_wind(self):
    building_lower = self.ZR + 2.0 < self.ZA
    self.UR = self.UA * numpy.log((self.ZR - self.ZDC)/self.Z0C
                                  )/numpy.log((self.ZA-self.ZDC)/self.Z0C)
    self.ZC = 0.7 * self.ZR
    self.XLB = 0.4 * (self.ZR - self.ZDC)
    # BB formulation from Inoue (1963)
    BB = 0.4 * self.ZR / (self.XLB * numpy.log((self.ZR - self.ZDC)/self.Z0C))
    self.UC = numpy.zeros(numpy.shape(self.UR))
    self.UC[building_lower] = (self.UR * numpy.exp(-BB*(1.- self.ZC/self.ZR)))[building_lower]
    self.UC[~building_lower] = self.UA[~building_lower] / 2.0
    self.ZC[~building_lower] = self.ZA[~building_lower] / 2.0
#    else:
#      # 'Warning ZR + 2m  is larger than the 1st WRF level'
#      self.ZC = self.ZA / 2.0
#      self.UC = self.UA / 2.0

  def urbparam_init(self):
    SIGMA_ZED = self.STDH_URB
    self.Lambda_F = numpy.mean(self.Lambda_F, axis=0)
    # Clamp between 10 cm and 20 m
    self.HNORM = 2. * self.ZR * self.urbfrac  / (self.lb_urb - self.Lambda_P)
    # TODO: check if this is correct
    self.HNORM[self.HNORM<self.ZR] = self.ZR[self.HNORM<self.ZR]
    self.HNORM = numpy.maximum( 0.01, numpy.minimum( 100.0, self.HNORM ) )
    self.HNORM[numpy.isnan(self.HNORM)] = 10
    Lambda_FR = numpy.maximum( 0.05, numpy.minimum( 0.35, self.stdh_urb/self.HNORM ) )
    #Lambda_FR  = SIGMA_ZED / ( self.road_width + self.roof_width )
    self.ZDC = self.ZR * (1.0 + (self.alpha_macd ** (-self.Lambda_P)) * (self.Lambda_P - 1.0))
    self.Z0C = self.ZR * (1.0 - (self.ZDC/self.ZR)) * numpy.exp(
      -(0.5 * self.beta_macd * self.Cd / (self.VKMC**2) * (1.0-self.ZDC/self.ZR)*self.Lambda_F)**(-0.5))
    self.Z0R = self.ZR * ( 1.0 - self.ZDC/self.ZR ) * numpy.exp(
      -(0.5 * self.beta_macd * self.Cd / (self.VKMC**2) * ( 1.0-self.ZDC/self.ZR) * Lambda_FR )**(-0.5))
    self.Z0HC = 0.1 * self.Z0C

  def net_shortwave_ratiation(self):
    boolean = (self.ZDC+self.Z0C+2 > self.ZA)
    #self.ZDC[boolean] = (self.ZA-self.Z0C -4)[boolean]
    self.SX = self.SSG/697.7/60.  # downward shortwave radiation [ly/min]
    self.RX = self.LLG/697.7/60.  # download longwave radiation [ly/min]
    self.HGT = self.ZR / self.HNORM # normalized height
    self.R = self.roof_width/(self.road_width+self.roof_width)  # average normalized width streets
    self.RW = 1 - self.R  # canyon width
    self.W = 2.*1.*self.HGT
    # RJR analytical formulation for wall sky view factor
    self.VFWS = 0.5*(1.+self.RW/self.HGT-numpy.sqrt(1.+(self.RW/self.HGT)**2.))
    self.VFGS=1.-2.*self.VFWS*self.HGT/self.RW
    self.SVF = self.VFGS
    self.VFGW=1.- self.SVF
    self.VFWG=(1.- self.SVF)*(1.-self.R)/self.W
    self.VFWW=1.-2.*self.VFWG
    shadow = False  # no shadow effect model
    daytime = self.SSG > 0.0
    if not shadow:
      SR1 = self.SX * (1.0 - self.ALBR)
      SGR1 = self.SX * (1.0 - self.ALBV)
      SG1 = self.SX * self.VFGS * (1.0 - self.ALBG)
      SB1 = self.SX * self.VFWS * (1.0 - self.ALBB)
      SG2 = SB1 * self.ALBB / (1.0 - self.ALBB
                               ) * self.VFGW*(1.0 - self.ALBG)
      SB2 = SG1 * self.ALBG / (1.0 - self.ALBG
                               ) * self.VFWG * (1.0 - self.ALBB)
    # else: #TODO implement shadow effects model
    # initialize
    self.SR = numpy.zeros(numpy.shape(SR1))
    self.SGR = numpy.zeros(numpy.shape(SR1))
    self.SG = numpy.zeros(numpy.shape(SR1))
    self.SB = numpy.zeros(numpy.shape(SR1))
    self.SNET = numpy.zeros(numpy.shape(SR1))
    # daytime
    self.SR[daytime] = SR1[daytime]
    self.SGR[daytime] = SGR1[daytime]
    self.SG[daytime] = SG1[daytime] + SG2[daytime]
    self.SB[daytime] = SB1[daytime] + SB2[daytime]
    # TODO: handle green roof option
    self.SNET[daytime] = ((self.R * self.SR) + (self.W * self.SB) + (self.RW * self.SG))[daytime]
    # night
    self.SR[~daytime] = 0.0
    self.SG[~daytime] = 0.0
    self.SGR[~daytime] = 0.0
    self.SB[~daytime] = 0.0
    self.SNET[~daytime] = 0.0

  def roof(self):
    # virtual temperatures needed by SFCDIF from Noah
    T1VR = self.TRP * (1.0 + 0.61 * self.QA)
    TH2V = (self.TA + (0.0098 * self.ZA)) * (1.0 + 0.61 * self.QA)
    # note that CHR_URB contains UA (=CHR_MOS*UA)
    self.CDR, self.RLMO_CAN = sfcdif_urb(self.ZA,self.Z0R,T1VR,TH2V,
                                         self.UA,self.AKANDA_URBAN,self.CMR_URB,
                                         self.CHR_URB)
    ALPHAR = self.RHO * self.CP * self.CHR_URB
    CHR = ALPHAR/self.RHO/self.CP/self.UA
    # Yang, 03/12/2014 -- LH for impervious roof surface
    RAIN1 = self.RAIN * 0.001 / 3600
    self.IMP_SCHEME = 1  # hardcode for now
    if (self.IMP_SCHEME == 1):
      rain = (self.RAIN > 1.0)
      BETR = numpy.zeros(numpy.shape(self.RAIN))
      BETR[rain] = 0.7
    # TODO: handle self.IMP_SCHEME==2
    self.TS_SCHEME = 1  # hardcode for now
    if self.TS_SCHEME ==1:
      for ii in range(1,20):  # iteration
        ES=6.11 * numpy.exp((2.5e6/461.51)*
                            (self.TRP-273.15)/(273.15*self.TRP))
        DESDT=(2.5e6/461.51)*ES/(self.TRP**2.)
        QS0R=0.622*ES/(self.PS-0.378*ES)
        DQS0RDTR = DESDT*0.622*self.PS/((self.PS-0.378*ES)**2.)

        self.RR=self.EPSR*(self.RX-self.SIG*(self.TRP**4.)/60.)
        self.HR=self.RHO*self.CP*CHR*self.UA*(self.TRP-self.TA)*100.
        self.ELER=self.RHO*self.EL*CHR*self.UA*BETR*(QS0R-self.QA)*100.
        self.G0R=self.AKSR*(self.TRP-self.TRL[0])/(self.DZR[0]/2.)
        F = self.SR + self.RR - self.HR - self.ELER - self.G0R

        DRRDTR = (-4.*self.EPSR*self.SIG*self.TRP**3.)/60.
        DHRDTR = self.RHO*self.CP*CHR*self.UA*100.
        DELERDTR = self.RHO*self.EL*CHR*self.UA*BETR*DQS0RDTR*100.
        DG0RDTR =  2.*self.AKSR/self.DZR[0]

        DFDT = DRRDTR - DHRDTR - DELERDTR - DG0RDTR
        DTR = F/DFDT

        self.TR = self.TRP - DTR
        self.TRP = self.TR
        if ((numpy.max(abs(F), axis=None) < 0.000001) and (numpy.max(abs(DTR), axis=None) < 0.000001)):
          break
      self.TRL = multi_layer(self.num_roof_layers,
                             self.BOUNDR,self.G0R,self.CAPR,self.AKSR,
                             self.TRL,self.DZR,self.DELT)
    else:
      # TR by force-restore method
      ES=6.11*numpy.exp( (2.5e6/461.51)*(self.TRP-273.15)/(273.15*self.TRP) )
      QS0R=0.622*ES/(self.PS-0.378*ES)

      self.RR=self.EPSR*(self.RX-self.SIG*(self.TRP**4.)/60.)
      self.HR=self.RHO*self.CP*CHR*self.UA*(self.TRP-self.TA)*100.
      self.ELER=self.RHO*self.EL*CHR*self.UA*BETR*(QS0R-self.QA)*100.
      self.G0R=self.SR+self.RR-self.HR-self.ELER

      self.TR = force_restore(self.CAPR,self.AKSR,self.DELT,
                              self.SR,self.RR,self.HR,self.ELER,
                              self.TRLEND,self.TRP)

      self.TRP=self.TR
    self.FLXTHR = self.HR/self.RHO/self.CP/100.
    self.FLXHUMR = self.ELER/self.RHO/self.EL/100.

  def wall_road(self):
    T1VC = self.TCP * (1.0 + 0.61 * self.QA)
    TH2V = (self.TA + ( 0.0098 * self.ZA)) * (1.0+ 0.61 * self.QA)
    self.CDC, self.RLMO_CAN = sfcdif_urb(self.ZA,self.Z0C,T1VC,
                          TH2V,self.UA,self.AKANDA_URBAN,
                          self.CMC_URB,self.CHC_URB)
    ALPHAC = self.RHO * self.CP * self.CHC_URB
    self.CH_SCHEME = 1 # TODO: hardcode for now
    if (self.CH_SCHEME == 1):
      Z = self.ZDC
      BHB=numpy.log(self.Z0B/self.Z0HB)/0.4
      BHG=numpy.log(self.Z0G/self.Z0HG)/0.4
      RIBB=(9.8 * 2./(self.TCP+self.TBP))*(self.TCP-self.TBP)*(Z+self.Z0B)/(self.UC*self.UC)
      RIBG=(9.8 * 2./(self.TCP+self.TGP))*(self.TCP-self.TGP)*(Z+self.Z0G)/(self.UC*self.UC)
      ALPHAB, CDB, self.XXXB, RIBB = mos(self.XXXB,
                                         BHB,RIBB,Z,self.Z0B,
                                         self.UC,self.TCP,self.TBP,self.RHO)
      ALPHAG, CDG, self.XXXG, RIBG = mos(self.XXXG,
                                         BHG,RIBG,Z,self.Z0G,
                                         self.UC,self.TCP,self.TGP,self.RHO)
    else:
      ALPHAB = self.RHO * self.CP * (6.15+4.18*self.UC)/1200.
      ALPHAG=self.RHO*self.CP*(6.15+4.18*self.UC)/1200.
      boolean = (self.UC > 5.0)
      ALPHAB[boolean] = (self.RHO*self.CP*(7.51*self.UC**0.78)/1200.)[boolean]
      ALPHAG[boolean]= (self.RHO*self.CP*(7.51*self.UC**0.78)/1200.)[boolean]


    CHC = ALPHAC/self.RHO/self.CP/self.UA
    CHB = ALPHAB/self.RHO/self.CP/self.UC
    CHG = ALPHAG/self.RHO/self.CP/self.UC

    # Yang 10/10/2013 -- LH from impervious wall and ground
    # self.IMP_SCHEME = 1  # TODO: hardcode for now
    if (self.IMP_SCHEME==1):  # TODO: handle IMP_SCHEME !=1
      BETB = numpy.zeros(numpy.shape(self.RAIN))
    rain = (self.RAIN > 1.0)
    BETG = numpy.zeros(numpy.shape(self.RAIN))
    BETG[rain] = 0.7

    # TODO: check if this is correct
    self.TB = self.TBP
    self.TG = self.TGP
    self.TS_SCHEME=0
    if (self.TS_SCHEME == 1):
      # TB, TG  Solving Non-Linear Simultaneous Equation by Newton-Rapson
      # TBL,TGL Solving Heat Equation by Tri Diagonal Matrix Algorithm
      for idx in range(0,20):
        ES=6.11*numpy.exp( (2.5*10.**6./461.51)*(self.TBP-273.15)/(273.15*self.TBP) )
        DESDT=(2.5*10.**6./461.51)*ES/(self.TBP**2.)
        QS0B=0.622*ES/(self.PS-0.378*ES)
        DQS0BDTB=DESDT*0.622*self.PS/((self.PS-0.378*ES)**2.)

        ES=6.11*numpy.exp( (2.5*10.**6./461.51)*(self.TGP-273.15)/(273.15*self.TGP) )
        DESDT=(2.5*10.**6./461.51)*ES/(self.TGP**2.)
        QS0G=0.622*ES/(self.PS-0.378*ES)
        DQS0GDTG=DESDT*0.622*self.PS/((self.PS-0.378*ES)**2.)

        RG1=self.EPSG*(self.RX*self.VFGS +
                       self.EPSB*self.VFGW*self.SIG*self.TBP**4./60. -
                       self.SIG*self.TGP**4./60. )
        RB1=self.EPSB*(self.RX*self.VFWS +
                       self.EPSG*self.VFWG*self.SIG*self.TGP**4./60. +
                       self.EPSB*self.VFWW*self.SIG*self.TBP**4./60. -
                       self.SIG*self.TBP**4./60. )
        RG2=self.EPSG*( (1.-self.EPSB)*(1.-self.SVF)*self.VFWS*self.RX +
                       (1.-self.EPSB)*(1.-self.SVF)*self.VFWG*self.EPSG *
                       self.SIG*self.TGP**4./60. +
                       self.EPSB*(1.-self.EPSB)*(1.-self.SVF)*
                       (1.-2.*self.VFWS)*self.SIG*self.TBP**4./60. )
        RB2 = self.EPSB*( (1.-self.EPSG)*self.VFWG*self.VFGS*self.RX +
                         (1.-self.EPSG)*self.EPSB*self.VFGW*self.VFWG*
                         self.SIG*(self.TBP**4.)/60. +
                         (1.-self.EPSB)*self.VFWS*(1.-2.*self.VFWS)*self.RX +
                         (1.-self.EPSB)*self.VFWG*(1.-2.*self.VFWS)*self.EPSG *
                         self.SIG*self.EPSG*self.TGP**4./60. +
                         self.EPSB*(1.-self.EPSB)*(1.-2.*self.VFWS) *
                         (1.-2.*self.VFWS)*self.SIG*self.TBP**4./60. )

        self.RG = RG1 + RG2
        self.RB = RB1 + RB2

        DRBDTB1 = self.EPSB*(4.*self.EPSB*self.SIG*self.TB**3.*self.VFWW-4.*self.SIG*self.TB**3.)/60.
        DRBDTG1 = self.EPSB*(4.*self.EPSG*self.SIG*self.TG**3.*self.VFWG)/60.
        DRBDTB2 = self.EPSB*(4.*(1.-self.EPSG)*self.EPSB*self.SIG*
                             self.TB**3.*self.VFGW*self.VFWG +4.*self.EPSB*
                             (1.-self.EPSB)*self.SIG*self.TB**3.*self.VFWW*self.VFWW)/60.
        DRBDTG2 = self.EPSB*(4.*(1.-self.EPSB)*self.EPSG*self.SIG *
                             self.TG**3.*self.VFWG*self.VFWW)/60.

        DRGDTB1=self.EPSG*(4.*self.EPSB*self.SIG*self.TB**3.*self.VFGW)/60.
        DRGDTG1=self.EPSG*(-4.*self.SIG*self.TG**3.)/60.
        DRGDTB2=self.EPSG*(4.*self.EPSB*(1.-self.EPSB)*self.SIG*self.TB**3.*self.VFWW*self.VFGW)/60.
        DRGDTG2=self.EPSG*(4.*(1.-self.EPSB)*self.EPSG*self.SIG*self.TG**3.*self.VFWG*self.VFGW)/60.

        DRBDTB = DRBDTB1 + DRBDTB2
        DRBDTG = DRBDTG1 + DRBDTG2
        DRGDTB = DRGDTB1 + DRGDTB2
        DRGDTG = DRGDTG1 + DRGDTG2

        self.HB = self.RHO*self.CP*CHB*self.UC*(self.TBP-self.TCP)*100.
        self.HG = self.RHO*self.CP*CHG*self.UC*(self.TGP-self.TCP)*100.

        DTCDTB = self.W * ALPHAB/(self.RW*ALPHAC+self.RW*ALPHAG+self.W*ALPHAB)
        DTCDTG = self.RW*ALPHAG/(self.RW*ALPHAC+self.RW*ALPHAG+self.W*ALPHAB)

        DHBDTB = self.RHO*self.CP*CHB*self.UC*(1.-DTCDTB)*100.
        DHBDTG = self.RHO*self.CP*CHB*self.UC*(0.-DTCDTG)*100.
        DHGDTG = self.RHO*self.CP*CHG*self.UC*(1.-DTCDTG)*100.
        DHGDTB = self.RHO*self.CP*CHG*self.UC*(0.-DTCDTB)*100.

        self.ELEB = self.RHO*self.EL*CHB*self.UC*BETB*(QS0B-self.QCP)*100.
        self.ELEG = self.RHO*self.EL*CHG*self.UC*BETG*(QS0G-self.QCP)*100.

        DQCDTB = self.W*ALPHAB*BETB*DQS0BDTB/(self.RW*ALPHAC+self.RW*ALPHAG*BETG+self.W*ALPHAB*BETB)
        DQCDTG = self.RW*ALPHAG*BETG*DQS0GDTG/(self.RW*ALPHAC+self.RW*ALPHAG*BETG+self.W*ALPHAB*BETB)

        DELEBDTB = self.RHO*self.EL*CHB*self.UC*BETB*(DQS0BDTB-DQCDTB)*100.
        DELEBDTG = self.RHO*self.EL*CHB*self.UC*BETB*(0.-DQCDTG)*100.
        DELEGDTG = self.RHO*self.EL*CHG*self.UC*BETG*(DQS0GDTG-DQCDTG)*100.
        DELEGDTB = self.RHO*self.EL*CHG*self.UC*BETG*(0.-DQCDTB)*100.

        self.G0B = self.AKSB*(self.TBP-self.TBL[0])/(self.DZB[0]/2.)
        self.G0G = self.AKSG*(self.TGP-self.TGL[0])/(self.DZG[0]/2.)

        DG0BDTB = 2.*self.AKSB/self.DZB[0]
        DG0BDTG = 0.
        DG0GDTG = 2.*self.AKSG/self.DZG[0]
        DG0GDTB = 0.

        F = self.SB + self.RB - self.HB - self.ELEB - self.G0B
        FX = DRBDTB - DHBDTB - DELEBDTB - DG0BDTB
        FY = DRBDTG - DHBDTG - DELEBDTG - DG0BDTG

        GF = self.SG + self.RG - self.HG - self.ELEG - self.G0G
        GX = DRGDTB - DHGDTB - DELEGDTB - DG0GDTB
        GY = DRGDTG - DHGDTG - DELEGDTG - DG0GDTG

        DTB =  (GF*FY-F*GY)/(FX*GY-GX*FY)
        DTG = -(GF+GX*DTB)/GY

        self.TB = self.TBP + DTB
        self.TG = self.TGP + DTG

        self.TBP = self.TB
        self.TGP = self.TG

        TC1 = self.RW*ALPHAC+self.RW*ALPHAG+self.W*ALPHAB
        TC2 = self.RW*ALPHAC*self.TA+self.RW*ALPHAG*self.TGP+self.W*ALPHAB*self.TBP
        self.TC = TC2/TC1
        QC1 = self.RW*ALPHAC+self.RW*ALPHAG*BETG+self.W*ALPHAB*BETB
        QC2 = self.RW*ALPHAC*self.QA+self.RW*ALPHAG*BETG*QS0G+self.W*ALPHAB*BETB*QS0B
        self.QC = QC2/QC1
        DTC = self.TCP - self.TC
        self.TCP = self.TC
        self.QCP = self.QC

        if ((numpy.max(abs(F), axis=None) < 0.000001) and (numpy.max(abs(DTB), axis=None) < 0.000001)
            and (numpy.max(abs(GF), axis=None) < 0.000001) and (numpy.max(abs(DTG), axis=None) < 0.000001)
            and (numpy.max(abs(DTC), axis=None) < 0.000001)):
          break

      self.TBL = multi_layer(self.num_wall_layers,self.BOUNDB,
                             self.G0B,self.CAPB,self.AKSB,self.TBL,
                             self.DZB,self.DELT)
      self.TGL = multi_layer(self.num_road_layers,self.BOUNDG,
                             self.G0G,self.CAPG,self.AKSG,self.TGL,
                             self.DZG,self.DELT)
    else:
      # TB, TG by force-restore method
      ES=6.11*numpy.exp((2.5e6/461.51)*(self.TBP-273.15)/(273.15*self.TBP) )
      QS0B=0.622*ES/(self.PS-0.378*ES)

      ES=6.11*numpy.exp((2.5e6/461.51)*(self.TGP-273.15)/(273.15*self.TGP) )
      QS0G=0.622*ES/(self.PS-0.378*ES)

      RG1=self.EPSG*( self.RX*self.VFGS 
                     +self.EPSB*self.VFGW*self.SIG*self.TBP**4./60.
                     -self.SIG*self.TGP**4./60. )
      RB1=self.EPSB*( self.RX*self.VFWS
                     +self.EPSG*self.VFWG*self.SIG*self.TGP**4./60.
                     +self.EPSB*self.VFWW*self.SIG*self.TBP**4./60.
                     -self.SIG*self.TBP**4./60. )

      RG2=self.EPSG*( (1.-self.EPSB)*(1.-self.SVF)*self.VFWS*self.RX
                     +(1.-self.EPSB)*(1.-self.SVF)
                     *self.VFWG*self.EPSG*self.SIG*self.TGP**4./60.
                     +self.EPSB*(1.-self.EPSB)*(1.-self.SVF)
                     *(1.-2.*self.VFWS)*self.SIG*self.TBP**4./60. )

      RB2=self.EPSB*( (1.-self.EPSG)*self.VFWG*self.VFGS*self.RX
                     +(1.-self.EPSG)*self.EPSB*self.VFGW*self.VFWG
                     *self.SIG*(self.TBP**4.)/60.
                     +(1.-self.EPSB)*self.VFWS*(1.-2.*self.VFWS)*self.RX
                     +(1.-self.EPSB)*self.VFWG*(1.-2.*self.VFWS)
                     *self.EPSG*self.SIG*self.EPSG*self.TGP**4./60.
                     +self.EPSB*(1.-self.EPSB)*(1.-2.*self.VFWS)
                     *(1.-2.*self.VFWS)*self.SIG*self.TBP**4./60. )
      self.RG=RG1+RG2
      self.RB=RB1+RB2
      self.HB=self.RHO*self.CP*CHB*self.UC*(self.TBP-self.TCP)*100.
      self.ELEB=self.RHO*self.EL*CHB*self.UC*BETB*(QS0B-self.QCP)*100.
      self.G0B=self.SB+self.RB-self.HB-self.ELEB
      self.HG=self.RHO*self.CP*CHG*self.UC*(self.TGP-self.TCP)*100.
      self.ELEG=self.RHO*self.EL*CHG*self.UC*BETG*(QS0G-self.QCP)*100.
      self.G0G=self.SG+self.RG-self.HG-self.ELEG
      self.TB = force_restore(self.CAPB,self.AKSB,self.DELT,self.SB,self.RB,
                              self.HB,self.ELEB,self.TBLEND,self.TBP)
      self.TG = force_restore(self.CAPG,self.AKSG,self.DELT,self.SG,self.RG,
                              self.HG,self.ELEG,self.TGLEND,self.TGP)
      self.TBP=self.TB
      self.TGP=self.TG
      TC1=self.RW*ALPHAC+self.RW*ALPHAG+self.W*ALPHAB
      TC2=self.RW*ALPHAC*self.TA+self.RW*ALPHAG*self.TGP+self.W*ALPHAB*self.TBP
      self.TC=TC2/TC1
      QC1=self.RW*ALPHAC+self.RW*ALPHAG*BETG+self.W*ALPHAB*BETB
      QC2=self.RW*ALPHAC*self.QA+self.RW*ALPHAG*BETG*QS0G+self.W*ALPHAB*BETB*QS0B
      self.QC=QC2/QC1
      self.TCP=self.TC
      self.QCP=self.QC


  def total_fluxes_urban_canopy(self):
    FLXTHB=self.HB/self.RHO/self.CP/100.
    FLXHUMB=self.ELEB/self.RHO/self.EL/100.
    FLXTHG=self.HG/self.RHO/self.CP/100.
    FLXHUMG=self.ELEG/self.RHO/self.EL/100.
    # TODO: implement green roof option
    if(self.ahoption==1):
      FLXTH  = ( self.R*self.FLXTHR  + self.W*FLXTHB  + self.RW*FLXTHG ) + self.AH/self.RHOO/self.CPP
    else:
      FLXTH  = ( self.R*self.FLXTHR  + self.W*FLXTHB  + self.RW*FLXTHG )
    if(self.alhoption==1):
      FLXHUM = ( self.R*self.FLXHUMR + self.W*FLXHUMB + self.RW*FLXHUMG )+ self.ALH/self.RHOO/self.ELL
    else:
      FLXHUM = ( self.R*self.FLXHUMR + self.W*FLXHUMB + self.RW*FLXHUMG )
    FLXUV = ( self.R*self.CDR + self.RW*self.CDC )*self.UA*self.UA
    FLXG = ( self.R*self.G0R + self.W*self.G0B + self.RW*self.G0G )
    LNET = self.R*self.RR + self.W*self.RB + self.RW*self.RG

    # Convert Unit: FLUXES and u* T* q*  --> WRF
    SH    = FLXTH  * self.RHOO * self.CPP    # Sensible heat flux          [W/m/m]
    #SHC   = FLXTHC * self.RHOO * self.CPP    # Canyon Sensible heat flux   [W/m/m]
    LH    = FLXHUM * self.RHOO * self.ELL    # Latent heat flux            [W/m/m]
    LH_KINEMATIC = FLXHUM * self.RHOO   # Latent heat, Kinematic      [kg/m/m/s]
    LW    = self.LLG - (LNET*697.7*60.) # Upward longwave radiation   [W/m/m]
    SW    = self.SSG - (self.SNET*697.7*60.) # Upward shortwave radiation  [W/m/m]
    ALB   = numpy.zeros(numpy.shape(SW))
    boolean = ( abs(self.SSG) > 0.0001)
    ALB[boolean] = (SW/self.SSG)[boolean] # Effective albedo [-]
    G = -FLXG*697.7*60.            # [W/m/m]
    RN = (self.SNET+LNET)*697.7*60.     # Net radiation [W/m/m]

    UST = numpy.sqrt(FLXUV)              # u* [m/s]
    TST = -FLXTH/UST               # T* [K]
    QST = -FLXHUM/UST              # q* [-]

    # diagnostic GRID AVERAGED  PSIM  PSIH  TS QS --> WRF
    Z0 = self.Z0C
    Z0H = self.Z0HC
    Z = self.ZA - self.ZDC
    ZNT = Z0
    XXX = 0.4*9.81*Z*TST/self.TA/UST/UST

    # initialize
    PSIM = numpy.zeros(numpy.shape(SW))
    PSIH = numpy.zeros(numpy.shape(SW))
    PSIM2 = numpy.zeros(numpy.shape(SW))
    PSIH2 = numpy.zeros(numpy.shape(SW))
    PSIM10 = numpy.zeros(numpy.shape(SW))
    PSIH10 = numpy.zeros(numpy.shape(SW))
    PSIHZA = numpy.zeros(numpy.shape(SW))
    PSIH2M = numpy.zeros(numpy.shape(SW))

    boolean = ( XXX >= 1. )
    XXX[boolean] = 1.
    boolean = ( XXX <= -5. )
    XXX[boolean] = -5.
    boolean = ( XXX > 0 )
    PSIM[boolean] = (-5. * XXX)[boolean]
    PSIH[boolean] = (-5. * XXX)[boolean]
    X = (1.-16.*XXX)**0.25
    PSIM[~boolean] = (2.*numpy.log((1.+X)/2.) + numpy.log((1.+X*X)/2.) - 2.*numpy.arctan(X) + numpy.pi/2.)[~boolean]
    PSIH[~boolean] = (2.*numpy.log((1.+X*X)/2.))[~boolean]

    GZ1OZ0 = numpy.log(Z/Z0)
    CD = 0.4**2./(numpy.log(Z/Z0)-PSIM)**2.

    CH = 0.4**2./(numpy.log(Z/Z0)-PSIM)/(numpy.log(Z/Z0H)-PSIH)
    CHS = 0.4*UST/(numpy.log(Z/Z0H)-PSIH)

    TS = self.TA + FLXTH/CHS    # surface potential temp (flux temp)
    QS = self.QA + FLXHUM/CHS   # surface humidity

    # diagnostic  GRID AVERAGED  U10  V10  TH2  Q2 --> WRF

    XXX2 = (2./Z)*XXX
    boolean = (XXX2 >= 1. )
    XXX2[boolean] = 1.
    boolean = ( XXX2 <= -5. )
    XXX2[boolean] = -5.

    boolean = ( XXX2 > 0 )
    PSIM2[boolean] = (-5. * XXX2)[boolean]
    PSIH2[boolean] = (-5. * XXX2)[boolean]
    X = (1.-16.*XXX2)**0.25
    PSIM2[~boolean] = (2.*numpy.log((1.+X)/2.) + numpy.log((1.+X*X)/2.) - 2.*numpy.log(X) + 2.*numpy.log(1.))[~boolean]
    PSIH2[~boolean] = (2.*numpy.log((1.+X*X)/2.))[~boolean]

    CHS2 = 0.4*UST/(numpy.log(2./Z0H)-PSIH2)

    XXX10 = (10./Z)*XXX
    boolean = ( XXX10 >= 1. )
    XXX10[boolean] = 1.
    boolean = ( XXX10 <= -5. )
    XXX10[boolean] = -5.

    boolean =  ( XXX10 > 0 )
    PSIM10[boolean] = (-5. * XXX10)[boolean]
    PSIH10[boolean] = (-5. * XXX10)[boolean]
    X = (1.-16.*XXX10)**0.25
    PSIM10[~boolean] = (2.*numpy.log((1.+X)/2.) + numpy.log((1.+X*X)/2.) - 2.*numpy.arctan(X) + 2.*numpy.arctan(1.))[~boolean]
    PSIH10[~boolean] = (2.*numpy.log((1.+X*X)/2.))[~boolean]

    PSIX = numpy.log(Z/Z0) - PSIM
    PSIT = numpy.log(Z/Z0H) - PSIH

    PSIX2 = numpy.log(2./Z0) - PSIM2
    PSIT2 = numpy.log(2./Z0H) - PSIH2

    PSIX10 = numpy.log(10./Z0) - PSIM10
    PSIT10 = numpy.log(10./Z0H) - PSIH10

    #U10 = U1 * (PSIX10/PSIX)       # u at 10 m [m/s]
    #V10 = V1 * (PSIX10/PSIX)       # v at 10 m [m/s]

    TH2 = TS + (self.TA-TS) *(CHS/CHS2)

    Q2 = QS + (self.QA-QS)*(PSIT2/PSIT)     # humidity at 2 m       [-]

    # RJR calculation of 2m temperature in the canyon
    # RJR according to Theeuwes et al. (2014)
    XXXCZA = self.ZA*self.RLMO_CAN # ZA/MOL over the canyon
    XXXC2M = 2.*self.RLMO_CAN # 2m/MOL over the canyon

    boolean = ( XXXCZA >= 1. )
    XXXCZA[boolean] = 1
    boolean = ( XXXCZA<= -5. )
    XXXCZA[boolean] = -5
    boolean = ( XXXC2M >= 1. )
    XXXC2M[boolean] = 1
    boolean = ( XXXC2M <= -5. )
    XXXC2M[boolean] = -5

    boolean = ( self.RLMO_CAN > 0 )
    PSIHZA[boolean] = (-5. * XXXCZA)[boolean]
    PSIH2M[boolean] = (-5. * XXXC2M)[boolean]
    X = (1.-16.*XXXCZA)**0.25
    PSIHZA[~boolean] = (2.*numpy.log((1.+X*X)/2.))[~boolean]
    X = (1.-16.*XXXC2M)**0.25
    PSIH2M[~boolean] = (2.*numpy.log((1.+X*X)/2.))[~boolean]

    RAH = 1./(self.AK*numpy.sqrt(self.CDC*self.UA*self.UA))*(numpy.log(self.ZA/2.)-PSIHZA+PSIH2M)
    TC2M = self.TA + RAH*(self.W/self.RW*FLXTHB+FLXTHG)
    self.TC2M = self.return_original_shape(TC2M, self.urban, numpy.shape(self.TC2Min))
    #nmin = nanmin(self.TC2M[self.TC2M>0], axis=None)
    #nmax = nanmax(self.TC2M[self.TC2M>0], axis=None)
    #print(numpy.nanmean(SH), nmin, nmax)
    #contourf(range(0,120),range(0,120),self.TC2M, numpy.arange(288,292,0.1))
    #contourf(range(0,120),range(0,120),self.TC2Min, numpy.arange(288,292,0.1))
    #self.ZA = self.return_original_shape(self.ZA, self.urban, numpy.shape(self.TC2Min))
    #colorbar()
    #show()
