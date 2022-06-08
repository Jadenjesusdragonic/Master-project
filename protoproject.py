from inspect import Parameter
import pylab

from pycbc import waveform
from pycbc.types import TimeSeries as PTimeSeries, FrequencySeries
from gwpy.timeseries import TimeSeries
import h5py
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import math
from bajes.obs.gw import Series
import pycbc.psd

from pycbc.waveform import get_fd_waveform
from pycbc.filter import matched_filter
from pycbc import psd



#time of GW170817
time_of_event = 1187008882.4 

post_trigger_duration =4
duration = 8
analysis_start = time_of_event + post_trigger_duration - duration

# Use gwpy to fetch the open data
H1_analysis_data = TimeSeries.fetch_open_data(
    "H1", analysis_start, analysis_start + duration, sample_rate=4096*4, 
    cache=True)

t = H1_analysis_data.times
strain = H1_analysis_data.value



# set the data properties coherently
seglen = 8           # duration of the segment [s]
srate  = 4096*4        # sampling rate [Hz]
t_gps  = 0 # central value of time
f_max  = 1024*4 
f_min  = 20 

series = Series('time', strain, seglen=seglen, srate=srate, t_gps=t_gps, 
                f_min=f_min, f_max=f_max)

from bajes.obs.gw import Waveform

wave  = Waveform(series.freqs, srate, seglen, 'NRPM') #NRPM for postmergers

fn="/Users/jadenjesus/Documents/GW170817_GWTC-1.hdf5"
data = h5py.File(fn,'r')

#defining samples

posterior_samples = data['IMRPhenomPv2NRT_highSpin_posterior']
#print parameter names 

pnames=posterior_samples.dtype.names
print('\n\n List of Parameters:')
print(pnames,'\n')
#using pandas to set pnames

pnames = posterior_samples.dtype.names

#reading in mass 1 samples

ind = pnames.index('m1_detector_frame_Msun')

m1_samples = np.array([samp[ind] for samp in posterior_samples[()]])

#reading in mass 2 samples

ind = pnames.index('m2_detector_frame_Msun')

m2_samples = np.array([samp[ind] for samp in posterior_samples[()]])

#Defining lambda1 parameter

ind = pnames.index('lambda1')

lambda1_samples = np.array([samp[ind] for samp in posterior_samples[()]])

#defining lambda2 parameter
ind = pnames.index('lambda2')

lambda2_samples = np.array([samp[ind] for samp in posterior_samples[()]])

#defining distance parameter
ind=pnames.index('luminosity_distance_Mpc')

distance_samples = np.array([samp[ind] for samp in posterior_samples[()]])

#inclination parameter
ind=pnames.index('costheta_jn')

inclination_samples = np.array([samp[ind] for samp in posterior_samples[()]])

#GW170817 has array shape ~4000, chooses where in the array to start taking data
arr_index = 2000


#declaration of variables for easy adjustment
deltaT= 1./srate
c = "white"
transparency = 0.4
lwidth= 3
iterations= 10


#make 4 subplots so all simulations could be grouped together
#formatting all 4 subplots
plt.style.use('seaborn-pastel')
fig, (f1, f2, f3, f4, f5,f6,f7) = plt.subplots(7, figsize=(8, 13))

#make simulations
for num in range(iterations):
#get values from the GW170817 data
  m1 = m1_samples[arr_index]
  m2= m2_samples[arr_index]
  lambda1= lambda1_samples[arr_index]
  lambda2= lambda2_samples[arr_index]
  luminosity_distance_Mpc=distance_samples[arr_index]
  costheta_jn=inclination_samples[arr_index]
#NRPM approx is fickle and does not work if lambda values are lower than 350
#experimenting have found that sometimes it would work if a lambda1 = ~200 and 
#lambda2 = >1000 however, for ease of use, I just set the values for both lambda
#values to be 350 or up
  while lambda1 <= 350.0 or lambda2 <=350.0:
    arr_index = arr_index + 1
    m1 = m1_samples[arr_index]
    m2= m2_samples[arr_index]
    lambda1= lambda1_samples[arr_index]
    lambda2= lambda2_samples[arr_index]
    luminosity_distance_Mpc=distance_samples[arr_index]
    costheta_jn=inclination_samples[arr_index]
    


  #calculate chirp mass and mass ratio based on values received from previous data
  chirp_mass = (math.pow((m1 * m2), (3/5))) / (math.pow((m1 + m2), (1/5)))
  q = m1 / m2

#parameters for new waveform
  params = {'mchirp'       : chirp_mass,    # chirp mass [solar masses] 
              'q'          : q,      # mass ratio 
              's1x'        : 0.,      # primary spin parameter, x component
              's1y'        : 0.,      # primary spin parameter, y component
              's1z'        : 0.,      # primary spin parameter, z component
              's2x'        : 0.,      # secondary spin parameter, x component
              's2y'        : 0.,      # secondary spin parameter, y component
              's2z'        : 0.,      # secondary spin parameter, z component
              'lambda1'    : lambda1,    # primary tidal parameter 
              'lambda2'    : lambda2,    # secondary tidal parameter
              'distance'   : luminosity_distance_Mpc,    # distance [Mpc]   #this will change 100.8114416513031 for using samples
              'iota'       : costheta_jn,   # inclination [rad]   #this will change to using samples instead fixed value np.pi
              'ra'         : 0.,     # right ascension [rad]
              'dec'        : 0.,   # declination [rad]
              'psi'        : 0.,      # polarization angle [rad]
              'time_shift' : 0.419,   # time shift from GPS time [s]
              'phi_ref'    : 0.,      # phase shift [rad]
              'f_min'      : 20.,     # minimum frequency [Hz]
              'srate'      : srate,   # sampling rate [Hz]
              'seglen'     : seglen,  # segment duration [s] 
              'tukey'      : 0.1,     # parameter for tukey window
              't_gps'      : t_gps,   # GPS trigger time
              'lmax'       : 0., 
              'eccentricity' : 0.
             }  

  #calculates hp and hc
  hp, hc = wave.compute_hphc(params)
  
  #makes a plot of h+ amplitude vs time
 
  f1.plot(series.times, hp, alpha= transparency, linewidth=3, label= None)
  
  #uses the pcybc way to get amp and freq by turning hp to a time series 
  #then a frequency series
  hp_ts = PTimeSeries(hp, delta_t=deltaT)
  hp_fs = hp_ts.to_frequencyseries(delta_f=0.125) #HP Frequency Series

  #makes plot of h+ amplitude vs frequency

  f2.loglog(hp_fs.sample_frequencies, np.abs(hp_fs), 
    lw=lwidth, alpha= transparency, label= None)
  
  #makes power spectral density plot

  f3.loglog(hp_fs.sample_frequencies, 4* np.abs(hp_fs) *  np.abs(hp_fs) * hp_fs.sample_frequencies,  
     lw=lwidth, alpha= transparency, label=None)
  
  #makes amplitude vs time plot

  f4.plot(series.times, np.abs(hp))

  #iterates arr_index for next simulation
  arr_index = arr_index + 1
  

#formating our waveform graphs individually
f1.set_title('Postmerger h+ Strain vs. Time', fontsize=22, loc='center')
f1.set_xlabel('Duration [s]', fontsize=14)
f1.set_xlim(0.00, t_gps+0.025)
f1.set_facecolor(c)
f1.grid(False)

f2.set_title('Postmerger h+ Amplitude vs. Frequency', fontsize=22)
f2.set_ylabel("Amplitude", fontsize=14)
f2.set_xlabel("Frequency", fontsize=14)
f2.set_xlim(xmin=1000, xmax=10000)
f2.set_ylim(ymin=10**-29, ymax=10**-24)
f2.set_facecolor(c)
f2.grid(False)

f3.set_title('Power Spectral Density', fontsize=22)
f3.set_xlim(xmin=10**3, xmax=10**4)
f3.set_ylim(ymin=10**-54, ymax=10**-44)
f3.set_facecolor(c)
f3.grid(False)

f4.set_title("H+ Amplitude vs Time", fontsize=23 )
f4.set_xlabel("Duration[s]")
f4.set_ylabel("Amplitude")
f4.set_xlim(xmin=.0, xmax=0.02)
f4.set_facecolor(c)


#SNR-------------------------------------------------------------------
print(pycbc.psd.get_lalsim_psd_list())

delta_f = 1.0 / 4
flen = int(1024 / delta_f)
low_frequency_cutoff = 5

xvh, _ = waveform.get_fd_waveform(approximant="IMRPhenomD", 
                                  mass1=m1_samples[arr_index],
                                  mass2=m2_samples[arr_index],
                                  distance=luminosity_distance_Mpc,
                                  delta_f=1.0,
                                  f_lower=low_frequency_cutoff)
                      
p1 = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, low_frequency_cutoff)





f5.loglog(xvh.sample_frequencies,abs(xvh), label='xvh')
f5.loglog(p1.sample_frequencies, p1, label='LowPower')

#f5.set_xlim(20,10000)

def waveform_power(waveform_model):
  return 4*np.abs(waveform_model)*np.abs(waveform_model)*waveform_model.sample_frequencies

CosmicExplorerPSD=psd.from_string('CosmicExplorerP1600143',len(xvh),xvh.delta_f, low_freq_cutoff=low_frequency_cutoff)
AdvancedLIGOPlusPSD=psd.from_string('aLIGOAPlusDesignSensitivityT1800042',len(xvh),xvh.delta_f,low_freq_cutoff=low_frequency_cutoff)

max_freq=CosmicExplorerPSD.sample_frequencies[-1]
max_freq_APlus=AdvancedLIGOPlusPSD.sample_frequencies[-1]
print(max_freq, max_freq_APlus)

f6.loglog(CosmicExplorerPSD.sample_frequencies,CosmicExplorerPSD,label='CosmicExplorer psd')
f6.loglog(AdvancedLIGOPlusPSD.sample_frequencies,AdvancedLIGOPlusPSD,label='A+ psd')
f6.loglog(xvh.sample_frequencies, waveform_power(xvh), label='gravitational wave model PSD')
f6.legend()


from pycbc.filter.matchedfilter import sigma
reference_SNR = sigma(xvh, CosmicExplorerPSD,low_frequency_cutoff=low_frequency_cutoff, high_frequency_cutoff=max_freq)
reference_SNR_Aplus = sigma(xvh, AdvancedLIGOPlusPSD,low_frequency_cutoff=low_frequency_cutoff, high_frequency_cutoff=max_freq)



print ("SNR of reference signal is", reference_SNR, "compared to", reference_SNR_Aplus,"in Advanced LIGO Plus")
print ("Cosmic Explorer gives SNR",reference_SNR/reference_SNR_Aplus ,"times greater." )
#SNR-----------------------------------------------------------------------------
from pycbc.waveform import get_td_waveform 
from pycbc.waveform import td_approximants 
from pycbc.types import TimeSeries 
from pycbc import waveform 
from astropy.cosmology import WMAP9 as cosmo

DELTA_T= 1.0/4096 #sampling rate 
FLOW= 15.0 #low frequency cut off for waveforms


#Define waveform function
def generate_3g_waveform(m1,m2,distance,inclination,lambda1,lambda2): 
    hp, hc = get_td_waveform(approximant='TaylorT4', 
                                 mass1=m1_samples[arr_index], 
                                 mass2=m2_samples[arr_index], 
                                 distance = luminosity_distance_Mpc, 
                                 delta_t=DELTA_T, 
                                 f_lower=FLOW,
                                 inclination=costheta_jn,
                                 lambda1=lambda1,
                                 lambda2=lambda2) 
    




f7.plot(hp, hp, label='demo' )
f7.plot(hp, hc, label='demo1')
f7.plot(xvh.sample_frequencies, waveform_power(xvh), label='gravitational wave model PSD')
#saves the images 
fig.savefig('/Users/jadenjesus/Documents/Grad_project/test10.png')
