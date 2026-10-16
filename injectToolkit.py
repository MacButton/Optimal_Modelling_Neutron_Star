#This code is made to run with Dynesty 2.1.1, Bilby 2.6.0 and Numpy 1.26.4

import pmtoolkit as pmt
import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
import importlib
import bilby
import gwinc
from bilby.core.utils.random import seed
from scipy.signal.windows import tukey
import bajes
from bajes.obs.gw import Waveform
import json
from scipy import interpolate
import math
from scipy.fft import fft, fftfreq, ifft
from scipy import signal
from gwpy.timeseries import TimeSeries as gwpyTS
import matplotlib.ticker as ticker
import pycbc.psd
import pycbc.noise
from pycbc.types import TimeSeries

def read_timeseries(seed_num): #assumes duration 1
    filename = os.path.join("/your_path/segments", f"{seed_num}.txt")
    data = np.loadtxt(filename)
    amplitude = data[:, 1]
    return amplitude

def frequency_from_time(segment_timeseries, t_0, duration):
    N=len(segment_timeseries)
    sampling_frequency=16384
    fft_amp=fft(segment_timeseries)/N
    freq_array=fftfreq(N, 1 / sampling_frequency)

    mask = freq_array >= 0
    freq_array_pos = freq_array[mask]
    fourier_amp = fft_amp[mask]

    return freq_array_pos, fourier_amp

def time_from_frequency(freq_series_complex, t_0, duration):
    N = len(freq_series_complex)
    time_domain_signal = ifft(freq_series_complex)
    dt = duration / N
    time = t_0 + np.arange(N) * dt

    return time, time_domain_signal*N

def NRPMw_model(frequency, mtot, q, s1z, s2z, lambda1, lambda2, distance, cosi,ra, dec, psi, phi_ref, NRPMw_t_coll, NRPMw_df_2, NRPMw_phi_pm, **kwargs):
    tukey_duration=0.2
    pm_fmin=500 
    sampling_frequency=8192 
    wave = bajes.obs.gw.Waveform(freqs=frequency, seglen=1, srate=sampling_frequency, approx='NRPMw_pmonly') 

  pars = {'mtot': mtot,    # total mass in solar masses, you can also use 'mchirp' if you prefer
        'q': q,                  # mass ratio >= 1
        's1z': s1z,               # z-aligned spin component 1st object
        's2z': s2z,               # z-aligned spin component 2nd object
        'lambda1': lambda1,         # tidal deformability 1st object
        'lambda2': lambda2,         # tidal deformability 2nd object
        'distance': distance,           # luminosity distane [Mpc]
        'cos_iota': cosi,             # inclination angle
        'ra': ra,                 # right ascension
        'dec': dec,            # declination
        'psi': psi,               # polarization angle
        'time_shift': 0.,          # time shift from GPS time - set to zero
        'phi_ref': phi_ref,           # reference phase
        'NRPMw_t_coll':700,     # mass-scaled time of collapse, range: [1, 3000], fixed
        'NRPMw_df_2': NRPMw_df_2*1E-4,       # mass-scaled frequency drift, range: [-1e-4, 1e-4]
        'NRPMw_phi_pm': NRPMw_phi_pm,   # phase discontinuity at t_0
        'f_min': pm_fmin,
        'srate': sampling_frequency,
        'seglen': 1,
        'tukey': tukey_duration,
        # 't_gps': kwargs['geocent_time']
         't_gps': kwargs['t_0'] #1187008882.43, #GW170817
    }

    hpc=wave.compute_hphc(pars)
    return {'plus': hpc.plus, 'cross': hpc.cross}

def read_metadata(wf_path_self, extra=False):
    #reads the metadata of the waveform for the signal
    data = {}
    with open(wf_path_self, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                data[key] = value
    if extra==False:
        params_wo_distance={
            's1z':float(data['id_spin_starA'][0]),
            's2z':float(data['id_spin_starB'][0]),
            'mtot':float(data['id_mass']),
            'q':float(data['id_mass_ratio']),
            'lambda1':float(data['id_Lambdaell_starA'].split(',')[0]),
            'lambda2':float(data['id_Lambdaell_starB'].split(',')[0]),
            'cosi':1,
            'NRPMw_df_2':0,
            'NRPMw_phi_pm':np.pi
        }
    if extra==True:
        params_wo_distance={
            's1z':float(data['id_spin_starA'][0]),
            's2z':float(data['id_spin_starB'][0]),
            'cosi':1,
            'NRPMw_phi_pm':np.pi
        }
    return params_wo_distance

def inverse(coords):
    x1=coords[0][0]
    y1=coords[0][1]
    a=y1*x1
    return a

def inverse_func(vals, SNR):
    a=vals
    return a/SNR

def distance_from_SNR(frequency_array, SNR, sampling_frequency, params, injection_parameters_wo_distance, duration, type_noise, strain_data_only_noise=None, timesegment=None):

    start_time=params['t_0']

    #making the detectors to inject into for the optimal_SNR
    detector_info={
        'CEHanford': dict(
            minimum_frequency = 0,
            maximum_frequency = 8192,
            length = 40,
            latitude = 46 + 27. / 60 + 18.528 / 3600,
            longitude = -(119 + 24. / 60 + 27.5657 / 3600),
            elevation = 142.554,
            xarm_azimuth = 125.9994,
            yarm_azimuth = 215.994,
            xarm_tilt = -6.195e-4,
            yarm_tilt = 1.25e-5
        ),
        'CELivingston': dict(
            minimum_frequency = 0,
            maximum_frequency = 8192,
            length = 40,
            latitude = 30 + 33. / 60 + 46.4196 / 3600,
            longitude = -(90 + 46. / 60 + 27.2654 / 3600),
            elevation = 142.554,
            xarm_azimuth = 125.9994,
            yarm_azimuth = 215.994,
            xarm_tilt = -6.195e-4,
            yarm_tilt = 1.25e-5
        )
    }

    def make_detector(name, metadata, frequency_array, start_time, type_noise, sampling_frequency, duration, strain_data_only_noise=None, timesegment=timesegment):
        if type_noise=='GWOSC':
            psd_array_freqs_welch, psd_array_before_colour=psd_from_welch_pycbc(timesegment, sampling_frequency*2) #the times two is fine because timeseries sampled at 16kHz
            psd_array=recolour_arr(psd_array_before_colour, psd_array_freqs_welch)
            psd_freq_array=psd_array_freqs_welch
        if type_noise=='GWINC':
            curve_ce = gwinc.load_budget("CE1").run()
            frequency_array_ce=curve_ce.freq
            psd_array_ce=curve_ce.psd
            psd_array=upsample(frequency_array_ce, psd_array_ce, sampling_frequency, duration)
            asd_array_ce=curve_ce.asd
            asd_array=upsample(frequency_array_ce, asd_array_ce, sampling_frequency, duration)
            psd_freq_array=np.linspace(0, sampling_frequency, sampling_frequency)
        ifo = bilby.gw.detector.Interferometer(name=name,
            power_spectral_density=bilby.gw.detector.PowerSpectralDensity(
                frequency_array=psd_freq_array,
                psd_array=psd_array),
                **metadata
                )
        # if type_noise=='GWOSC':
        #   ifo.strain_data.set_from_frequency_domain_strain(
        #       strain_data_only_noise,
        #       start_time=start_time,
        #       frequency_array=frequency_array,
        #   )
        #   # # ifo.strain_data.set_from_time_domain_strain(
        #   # #     time_domain_strain=timesegment,
        #   # #     sampling_frequency=sampling_frequency*2,
        #   # #     duration=duration,
        #   # #     start_time=start_time
        #   # # )
        #   # ifo.set_strain_data_from_power_spectral_density(
        #   #   sampling_frequency=sampling_frequency*2,
        #   #   duration=1,
        #   #   start_time=start_time,
        #   # )
        return ifo

    CEHanford = make_detector("CEHanford", detector_info["CEHanford"], frequency_array, start_time, type_noise, sampling_frequency, duration, strain_data_only_noise=strain_data_only_noise, timesegment=timesegment)
    CELivingston = make_detector("CELivingston", detector_info["CELivingston"], frequency_array, start_time, type_noise, sampling_frequency, duration, strain_data_only_noise=strain_data_only_noise, timesegment=timesegment)

    ifos = [CEHanford, CELivingston]
    min_freq=500
    max_freq=6000

    #the waveform injecting to get the optimal_snr
    waveform_generator_MB = bilby.gw.waveform_generator.WaveformGenerator(
            duration=duration,
            sampling_frequency=16384-2/duration,
            frequency_domain_source_model = NRPMw_model,
            waveform_arguments=params
            )

    #actually injecting that waveform to get optimal_snr
    coords=[]

    for test_distance in {5}: #Mpc
        injection_parameters=injection_parameters_wo_distance|{'distance': test_distance}
        for ifo in ifos:
            ifo.minimum_frequency = min_freq
            ifo.maximum_frequency = max_freq
            if type_noise=='GWOSC':
                ifo.strain_data.set_from_frequency_domain_strain(
                    strain_data_only_noise,
                    start_time=start_time,
                    frequency_array=frequency_array,
                )
                # ifo.set_strain_data_from_power_spectral_density(
                #   sampling_frequency=sampling_frequency*2-2,
                #   duration=1,
                #   start_time=start_time,
                # )
            if type_noise=='GWINC':
                ifo.set_strain_data_from_power_spectral_density(
                    sampling_frequency=sampling_frequency*2-2,
                    duration=1,
                    start_time=start_time,
                )
            ifo.inject_signal(
                waveform_generator=waveform_generator_MB,
                parameters=params|injection_parameters,
                raise_error=False
            )
            ifo.minimum_frequency = min_freq #again just to be safe
            ifo.maximum_frequency = max_freq
        opt_snr0=ifos[0].meta_data['optimal_SNR']
        opt_snr1=ifos[1].meta_data['optimal_SNR']
        coords.append([test_distance, (opt_snr0**2+opt_snr1**2)**(1/2)])
    func_vals=inverse(coords)
    distance=inverse_func(func_vals, SNR)
    print('the injected distance is '+str(distance) +' for a SNR of '+str(SNR))
    return (distance, ifos)

def core_waveform_model(frequency, **inject_params):
    wf_path_core = inject_params['wf_path_core']
    loudness = inject_params['loudness']
    t_0 = inject_params['t_0']
    duration=inject_params['duration']
    timeseries_segment=waveform_core_upsampled(wf_path_core, loudness)
    frequency_array, strain_data_freqseries=frequency_from_time(timeseries_segment, t_0, duration)

    #we want this at 8192 Hz samples, so it must be interpolated
    interp_strain_data_freqseries=upsample(frequency_array, strain_data_freqseries, 8192, duration)
    plus=np.real(interp_strain_data_freqseries)
    cross=np.imag(interp_strain_data_freqseries)
    return {'plus': plus, 'cross': cross}

def linear_loudness(test_SNR, SNR):
    return SNR/test_SNR

def amplitude_from_SNR(frequency_array, SNR, sampling_frequency, params, injection_parameters_wo_amplitude, duration, type_noise, waveform_arguments, strain_data_only_noise=None, timesegment=None):

    start_time=params['t_0']

    #making the detectors to inject into for the optimal_SNR
    detector_info={
        'CEHanford': dict(
            minimum_frequency = 0,
            maximum_frequency = 8192,
            length = 40,
            latitude = 46 + 27. / 60 + 18.528 / 3600,
            longitude = -(119 + 24. / 60 + 27.5657 / 3600),
            elevation = 142.554,
            xarm_azimuth = 125.9994,
            yarm_azimuth = 215.994,
            xarm_tilt = -6.195e-4,
            yarm_tilt = 1.25e-5
        ),
        'CELivingston': dict(
            minimum_frequency = 0,
            maximum_frequency = 8192,
            length = 40,
            latitude = 30 + 33. / 60 + 46.4196 / 3600,
            longitude = -(90 + 46. / 60 + 27.2654 / 3600),
            elevation = 142.554,
            xarm_azimuth = 125.9994,
            yarm_azimuth = 215.994,
            xarm_tilt = -6.195e-4,
            yarm_tilt = 1.25e-5
        )
    }

    def make_detector(name, metadata, frequency_array, start_time, type_noise, sampling_frequency, duration, strain_data_only_noise=None, timesegment=timesegment):
        if type_noise=='GWOSC':
            psd_array_freqs_welch, psd_array_before_colour=psd_from_welch_pycbc(timesegment, sampling_frequency*2)
            psd_array=recolour_arr(psd_array_before_colour, psd_array_freqs_welch)
            psd_freq_array=psd_array_freqs_welch
        if type_noise=='GWINC':
            curve_ce = gwinc.load_budget("CE1").run()
            frequency_array_ce=curve_ce.freq
            psd_array_ce=curve_ce.psd
            psd_array=upsample(frequency_array_ce, psd_array_ce, sampling_frequency, duration)
            asd_array_ce=curve_ce.asd
            asd_array=upsample(frequency_array_ce, asd_array_ce, sampling_frequency, duration)
            psd_freq_array=np.linspace(0, sampling_frequency, sampling_frequency)
        ifo = bilby.gw.detector.Interferometer(name=name,
            power_spectral_density=bilby.gw.detector.PowerSpectralDensity(
                frequency_array=psd_freq_array,
                psd_array=psd_array),
                **metadata
                )
        # if type_noise=='GWOSC':
        #   ifo.strain_data.set_from_frequency_domain_strain(
        #       strain_data_only_noise,
        #       start_time=start_time,
        #       frequency_array=frequency_array,
        #   )
        #   # # ifo.strain_data.set_from_time_domain_strain(
        #   # #     time_domain_strain=timesegment,
        #   # #     sampling_frequency=sampling_frequency*2,
        #   # #     duration=duration,
        #   # #     start_time=start_time
        #   # # )
        #   # ifo.set_strain_data_from_power_spectral_density(
        #   #   sampling_frequency=sampling_frequency*2,
        #   #   duration=1,
        #   #   start_time=start_time,
        #   # )
        return ifo

    CEHanford = make_detector("CEHanford", detector_info["CEHanford"], frequency_array, start_time, type_noise, sampling_frequency, duration, strain_data_only_noise=strain_data_only_noise, timesegment=timesegment)
    CELivingston = make_detector("CELivingston", detector_info["CELivingston"], frequency_array, start_time, type_noise, sampling_frequency, duration, strain_data_only_noise=strain_data_only_noise, timesegment=timesegment)

    ifos = [CEHanford, CELivingston]
    min_freq=500
    max_freq=6000

    #actually injecting that waveform to get optimal_snr
    coords=[]

    for logB in {-20}:
        inject_params=injection_parameters_wo_amplitude

        waveform_generator_easter = bilby.gw.waveform_generator.WaveformGenerator(
            duration=duration,
            sampling_frequency=16384-2/duration, #this might be wrong lowkey
            time_domain_source_model = easter_model,
            parameter_conversion = lambda x: (x, None),
            start_time=start_time
            )
        for ifo in ifos:
            ifo.minimum_frequency = min_freq
            ifo.maximum_frequency = max_freq
            if type_noise=='GWOSC':
                ifo.strain_data.set_from_frequency_domain_strain(
                    strain_data_only_noise,
                    start_time=start_time,
                    frequency_array=frequency_array,
                )
            if type_noise=='GWINC':
                ifo.set_strain_data_from_power_spectral_density(
                    sampling_frequency=sampling_frequency*2-2,
                    duration=1,
                    start_time=start_time,
                )
            parameters_new=inject_params|{'logB':logB}|waveform_arguments
            ifo.inject_signal(
                waveform_generator=waveform_generator_easter,
                parameters=parameters_new,
                raise_error=False,
            )
            ifo.minimum_frequency = min_freq #again just to be safe
            ifo.maximum_frequency = max_freq
        opt_snr0=ifos[0].meta_data['optimal_SNR']
        opt_snr1=ifos[1].meta_data['optimal_SNR']
        test_SNR=(opt_snr0**2+opt_snr1**2)**(1/2)
    logB_optimal=logB+np.log10(SNR)-np.log10(test_SNR)
    print('the injected logB is '+str(logB_optimal) +' for a SNR of '+str(SNR))
    return (logB_optimal, ifos)

def gwinc_ce():
    curve_ce = gwinc.load_budget("CE1").run()
    curve_ce_freq=curve_ce.freq
    curve_ce_freq_max=np.max(curve_ce_freq)
    curve_ce_freq_min=np.min(curve_ce_freq)
    psd_ce=bilby.gw.detector.PowerSpectralDensity(frequency_array=curve_ce_freq, psd_array=curve_ce.psd)
    return psd_ce

def recolour_arr(input_arr, frequency_array, type_recolour='psd'):
    curve_aLIGO = gwinc.load_budget("aLIGO").run()
    curve_ce = gwinc.load_budget("CE1").run()

    interp_aLIGO = interpolate.interp1d(curve_aLIGO.freq, curve_aLIGO.psd,
                                    bounds_error=False, fill_value=np.mean(curve_aLIGO.psd))
    interp_ce = interpolate.interp1d(curve_ce.freq, curve_ce.psd,
                                     bounds_error=False, fill_value=np.mean(curve_ce.psd))
    psd_aLIGO = interp_aLIGO(frequency_array)
    psd_ce = interp_ce(frequency_array)

    mask = (frequency_array >= 500) & (frequency_array <= 6000)

    output_arr = np.zeros_like(input_arr)

    ratio = np.zeros_like(input_arr, dtype=float)
    if type_recolour!='psd':
        ratio[mask] = np.sqrt(psd_ce[mask] / psd_aLIGO[mask])
    else:
        ratio[mask] = psd_ce[mask] / psd_aLIGO[mask]

    output_arr[mask] = input_arr[mask] * ratio[mask]

    return output_arr

# def psd_from_complex_strain(strain_freq, frequency_array, duration):
#   psd =  np.abs(strain_freq)**2 / duration
#   asd = np.sqrt(psd)
#   return psd, asd

def psd_from_welch_pycbc(ts, sampling_freq):
    timeseries_welch=TimeSeries(ts, delta_t=1/sampling_freq)
    delta_t=1/sampling_freq
    seg_len = int(1 / delta_t)
    seg_stride = int(seg_len //2)
    estimated_psd = pycbc.psd.welch(timeseries_welch,
                        seg_len=seg_len,
                        seg_stride=seg_stride,
                        window='hann',
                        avg_method='median',
                        num_segments=None,
                        require_exact_data_fit=True
                            )
    freqs = estimated_psd.sample_frequencies
    psd_vals = np.array(estimated_psd)
    new_freqs = np.linspace(freqs[0], freqs[-1], sampling_freq)
    new_psd = np.interp(new_freqs, freqs, psd_vals)

    return new_freqs, new_psd


def recolour_timeseries(time_series, sampling_frequency):
    N = len(time_series)
    dt = 1.0 / sampling_frequency
    freq_series = np.fft.rfft(time_series)
    freqs = np.fft.rfftfreq(N, dt)
    freq_series_recoloured = recolour_arr(freq_series, freqs, type_recolour='strain')
    time_series_new = np.fft.irfft(freq_series_recoloured, n=N)
    return time_series_new

def inject_signal_itk(outdir, frequency_array, sampling_frequency, duration, t_0, waveform_generator_inject, parameters, type_noise, strain_data_only_noise=None, timesegment=None, inject='yes'):

    min_freq=500
    max_freq=6000

    start_time=t_0

    #making the detectors to inject into for the optimal_SNR
    detector_info={
        'CEHanford': dict(
            minimum_frequency = 0,
            maximum_frequency = 8192,
            length = 40,
            latitude = 46 + 27. / 60 + 18.528 / 3600,
            longitude = -(119 + 24. / 60 + 27.5657 / 3600),
            elevation = 142.554,
            xarm_azimuth = 125.9994,
            yarm_azimuth = 215.994,
            xarm_tilt = -6.195e-4,
            yarm_tilt = 1.25e-5
        ),
        'CELivingston': dict(
            minimum_frequency = 0,
            maximum_frequency = 8192,
            length = 40,
            latitude = 30 + 33. / 60 + 46.4196 / 3600,
            longitude = -(90 + 46. / 60 + 27.2654 / 3600),
            elevation = 142.554,
            xarm_azimuth = 125.9994,
            yarm_azimuth = 215.994,
            xarm_tilt = -6.195e-4,
            yarm_tilt = 1.25e-5
        )
    }

    def make_detector_inject(outdir, name, metadata, frequency_array, start_time, type_noise, sampling_frequency, duration, strain_data_only_noise=None, timesegment=timesegment):
        if type_noise=='GWOSC':
            psd_array_freqs_welch, psd_array_before_colour=psd_from_welch_pycbc(timesegment, sampling_frequency*2)
            psd_array=recolour_arr(psd_array_before_colour, psd_array_freqs_welch)
            psd_freq_array=psd_array_freqs_welch
        if type_noise=='GWINC':
            curve_ce = gwinc.load_budget("CE1").run()
            frequency_array_ce=curve_ce.freq
            psd_array_ce=curve_ce.psd
            psd_array=upsample(frequency_array_ce, psd_array_ce, sampling_frequency, duration)
            asd_array_ce=curve_ce.asd
            asd_array=upsample(frequency_array_ce, asd_array_ce, sampling_frequency, duration)
            psd_freq_array=np.linspace(0, sampling_frequency, sampling_frequency)
        ifo = bilby.gw.detector.Interferometer(name=name,
                                               power_spectral_density=bilby.gw.detector.PowerSpectralDensity(
                                                    frequency_array=psd_freq_array,
                                                    psd_array=psd_array),
                                                    **metadata
                                                )
        ifo.minimum_frequency = min_freq
        ifo.maximum_frequency = max_freq
        if type_noise=='GWOSC':
            timesegment_recoloured=recolour_timeseries(timesegment, sampling_frequency*2)
            time_series=gwpyTS(timesegment_recoloured,
                                  sample_rate=sampling_frequency*2)
            ifo.set_strain_data_from_gwpy_timeseries(
                time_series=time_series
            )
  if type_noise=='GWINC':
            ifo.set_strain_data_from_power_spectral_density(
                sampling_frequency=sampling_frequency*2-2,
                duration=1,
                start_time=start_time,
            ) #this has the 'white noise' generation in it
        if inject=='yes':
            ifo.inject_signal(
                waveform_generator=waveform_generator_inject,
                parameters=parameters,
                raise_error=False
            )
        ifo.minimum_frequency = min_freq #again just to be safe
        ifo.maximum_frequency = max_freq
        ######################
        polarisations = waveform_generator_inject.frequency_domain_strain(
            {**parameters}
        )
        signal = ifo.get_detector_response(
            polarisations,
            {**parameters}
        )
        #outdir='/home/macbutton/Masters/NR waveforms and noise/Plotting/splitting_code'
        ifo.plot_data(signal=signal, outdir=outdir, label="strain_plot")

        ###############
        return ifo
    CEHanford = make_detector_inject(outdir, "CEHanford_inject", detector_info["CEHanford"], frequency_array, start_time, type_noise, sampling_frequency, duration, strain_data_only_noise=strain_data_only_noise, timesegment=timesegment)
    CELivingston = make_detector_inject(outdir, "CELivingston_inject", detector_info["CELivingston"], frequency_array, start_time, type_noise, sampling_frequency, duration, strain_data_only_noise=strain_data_only_noise, timesegment=timesegment)

    ifos = [CEHanford, CELivingston]

    return ifos, frequency_array

def convert_masses(parameters_dict):
    mtot = parameters_dict['mtot']
    q = parameters_dict['q']

    converted_parameters = parameters_dict.copy()
    converted_parameters['mass1'] = mtot / (1 + 1/q)
    converted_parameters['mass2'] = mtot / (1 + q)
    return converted_parameters

def prior_generate_NRPMw(injection_parameters, NR_constant_parameters, waveform_arguments, type_wf,distance_percent=0.2, distance_guess=None):
    prior = bilby.core.prior.PriorDict(conversion_function = convert_masses)
    params=injection_parameters|NR_constant_parameters|waveform_arguments
    params.pop('loudness')
    for param in params:
        if type_wf=='NRPMw':
            if param not in ['mtot', 'q', 'lambda1', 'lambda2', 'NRPMw_df_2', 'wf_path_core']:
                prior[param]=params[param]
        if type_wf=='Easter':
            prior['distance']=injection_parameters['equivalent_distance']
            prior['s1z']=0
            prior['s2z']=0
            prior['cosi']=0
            prior['psi']=NR_constant_parameters['psi']
            prior['ra']=NR_constant_parameters['ra']
            prior['dec']=NR_constant_parameters['dec']
            prior['phi_ref']=waveform_arguments['phi_ref']
            prior['NRPMw_phi_pm']=np.pi
            prior['t_0']=NR_constant_parameters['t_0']
            prior['NRPMw_t_coll']=700
            prior['geocent_time']=NR_constant_parameters['t_0']
            prior['ra']=NR_constant_parameters['ra']
            prior['dec']=NR_constant_parameters['dec']
            #add the fixed distance prior when you have it
        prior['mass1']=bilby.core.prior.Constraint(minimum=1, maximum=2.5, name = 'mass1')
        prior['mass2']=bilby.core.prior.Constraint(minimum=1, maximum=2.5, name = 'mass2')
        prior['mtot'] = bilby.core.prior.Uniform(minimum=2, maximum = 5, name ='mtot')
        prior['lambda1']=bilby.core.prior.Uniform(minimum=100, maximum=3000, name='lambda1')
        prior['lambda2']=bilby.core.prior.Uniform(minimum=100, maximum=3000, name='lambda2')
        prior['q']=bilby.core.prior.Uniform(minimum=1, maximum = 1.5, name='q')
        prior['NRPMw_df_2']=bilby.core.prior.Uniform(minimum=-1, maximum=1, name='NRPMw_df_2')
    return prior

def prior_generate_Easter(injection_params):
    prior = bilby.core.prior.PriorDict()
    for param in injection_params:
        prior[param]=injection_params[param] #has logB, t_0, and waveform number
    prior['w_0']=bilby.core.prior.Uniform(minimum=0, maximum=1, name = 'w_0')
    prior['w_1']=bilby.core.prior.Uniform(minimum=0, maximum=1, name = 'w_1')
    #dont need a prior of w_2 as its defined to be 1-w_0-w_1

    prior['f_0']=bilby.core.prior.Uniform(minimum=1000, maximum=5000, name = 'f_0')
    prior['f_1']=bilby.core.prior.Uniform(minimum=1000, maximum=5000, name = 'f_1')
    prior['f_2']=bilby.core.prior.Uniform(minimum=1000, maximum=5000, name = 'f_2')

    prior['T_0']=bilby.core.prior.Uniform(minimum=10**(-4), maximum=10**(0.3), name = 'T_0')
    prior['T_1']=bilby.core.prior.Uniform(minimum=10**(-4), maximum=10**(0.3), name = 'T_1')
    prior['T_2']=bilby.core.prior.Uniform(minimum=10**(-4), maximum=10**(0.3), name = 'T_2')

    prior['phi_0']=bilby.core.prior.Uniform(minimum=-1*np.pi, maximum=np.pi, name = 'phi_0')
    prior['phi_1']=bilby.core.prior.Uniform(minimum=-1*np.pi, maximum=np.pi, name = 'phi_1')
    prior['phi_2']=bilby.core.prior.Uniform(minimum=-1*np.pi, maximum=np.pi, name = 'phi_2')

    prior['alpha_0']=bilby.core.prior.Uniform(minimum=-6.4, maximum=6.4, name = 'alpha_0')
    prior['alpha_1']=bilby.core.prior.Uniform(minimum=-6.4, maximum=6.4, name = 'alpha_1')
    prior['alpha_2']=bilby.core.prior.Uniform(minimum=-6.4, maximum=6.4, name = 'alpha_2')

    return prior

def upsample(xvals, yvals, sampling_frequency, duration=None):
    interp= interpolate.interp1d(xvals, yvals,
                                     bounds_error=False, fill_value=yvals[-1])
    interpolated = interp(np.linspace(0, sampling_frequency, sampling_frequency))
    return interpolated

def waveform_core_upsampled(wf_path_core, loudness=None, duration=1.0, fs_target=16384):

    loaded_wf = pmt.NRWaveform(wf_path_core, loudness=loudness)
    t_orig = loaded_wf.tnew
    h_re = loaded_wf.hrenew
    h_im = loaded_wf.himnew

    duration_orig = t_orig[-1] - t_orig[0]
    N_new = int(duration_orig * fs_target)
    t_new = np.linspace(t_orig[0], t_orig[-1], N_new)

    # Interpolate
    interp_re = interpolate.interp1d(t_orig, h_re, kind='cubic', bounds_error=False, fill_value=0.0)
    interp_im = interpolate.interp1d(t_orig, h_im, kind='cubic', bounds_error=False, fill_value=0.0)

    h_new = interp_re(t_new) - 1j*interp_im(t_new)
    return h_new

def calculate_mode_waveform(f, T, phi, alpha, A, tpos, Two_pi_dt, dplus, dcross):

    Amp = np.exp(-tpos / T) * A
    angle = Two_pi_dt * f * (1 + alpha * tpos) + phi
    if dplus is not None:
        dplus = Amp * np.sin(angle)
    if dcross is not None:
        dcross = Amp * np.cos(angle)
    return dplus, dcross

def easter_model(
        time,
        t_0,
        logB,
        w_0,
        w_1,
        f_0, f_1, f_2,
        T_0, T_1, T_2,
        phi_0, phi_1, phi_2,
        alpha_0, alpha_1, alpha_2,
        ra=0, #dont look at this
        dec=0, #bilbys gives these variables to PE but easter model doenst want them
        geocent_time=0, #shh
        psi=0 #the last one i promise
):
    """
    Three damped sinusoidal waveforms, h+:cos, hx:sin.
    waveform_order is 3 for 3 damped sinusoids
    """
    waveform_order = 3

    w_2 = 1 - w_0 - w_1
    w_list = [w_0, w_1, w_2]

    f_list = [f_0, f_1, f_2]
    T_list = [T_0, T_1, T_2]
    phi_list = [phi_0, phi_1, phi_2]
    alpha_list = [alpha_0, alpha_1, alpha_2]

    # waveform_order = search_waveform.waveform_arguments['waveform_order']
    # tukey_rolloff = search_waveform.waveform_arguments['tukey_rolloff']

    tukey_rolloff=0.2 #check this

    plus = np.zeros(len(time))
    cross = plus.copy()
    #t_0 = t_0 / 1000.0
    #    print(t_0)
    dt = time# - t_0

    # tpos = dt[dt >= 0]
    tpos = time - time[0]
    dplus = np.zeros(len(tpos))
    dcross = dplus.copy()
    Two_pi_dt = 2 * np.pi * tpos
    dtpos = tpos[-1] - tpos[0]
    window = tukey(len(tpos), 2 * tukey_rolloff / dtpos)  # 0.2ms tukey
    for waveform_number in range(waveform_order):
        if waveform_number == 2:
            A = (1 - w_list[0] - w_list[1]) * 10 ** logB
        else:
            A = w_list[waveform_number] * 10 ** logB
        f=f_list[waveform_number]
        T = T_list[waveform_number] / 1000.0
        phi = phi_list[waveform_number]
        alpha = alpha_list[waveform_number]
        Amp = np.exp(-tpos / T) * A
        # note alpha is multiplied by (t-t_0)**2
        angle = Two_pi_dt * f * (1 + alpha * tpos) + phi
        ddplus, ddcross = calculate_mode_waveform(f_list[waveform_number],
                                                    T_list[waveform_number] / 1000.0,
                                                    phi_list[waveform_number],
                                                    alpha_list[waveform_number],
                                                    A, tpos, Two_pi_dt, dplus, dcross)
        dplus += ddplus
        dcross += ddcross
    plus[dt >= 0] = dplus * window
    cross[dt >= 0] = dcross * window
    #    print(sum(plus))
    # plt.figure()
    # plt.plot(time,plus)
    # plt.xlim(0.032, 0.042)
    # plt.show()
    # 1/0

    return {'plus': plus, 'cross': cross}

def main_inject(SNR, seed_number, t_0, noise_path, wf_path, type_noise, type_wf, distance_percent, inject='yes'):
    wf_path_self=os.path.join(wf_path, 'metadata_main.txt')
    wf_path_core=os.path.join(wf_path, 'R01/data.h5')


    if t_0==1240215503:
        ra=np.radians(-165.45)
        dec=np.radians(51.82)
        if inject=='no':
            outdir=os.path.join(
            'your_path/resultsBaseline190425',
            str(type_noise)+'_'+str(type_wf)+'_SNR'+str(int(SNR))+'_SEED'+str(int(seed_number))+'_'+str(100)
                )
        else:
            if type_wf=='Easter':
                outdir=os.path.join(
                'your_path/resultsEaster190425',
                str(type_noise)+'_'+str(type_wf)+'_SNR'+str(int(SNR))+'_SEED'+str(int(seed_number))+'_'+str(100)
                    )
            else:
                outdir=os.path.join(
                'your_path/results190425',
                str(type_noise)+'_'+str(type_wf)+'_SNR'+str(int(SNR))+'_SEED'+str(int(seed_number))+'_'+str(100)
                    )
    if t_0==1187008882:
        ra=np.radians(12.73)
        dec=np.radians(51.82)
        if inject=='no':
            outdir=os.path.join(
            'your_path/resultsBaseline170817',
            str(type_noise)+'_'+str(type_wf)+'_SNR'+str(int(SNR))+'_SEED'+str(int(seed_number))+'_'+str(100)
                )
        else:
            if type_wf=='Easter':
                outdir=os.path.join(
                'your_path/resultsEaster170817',
                str(type_noise)+'_'+str(type_wf)+'_SNR'+str(int(SNR))+'_SEED'+str(int(seed_number))+'_'+str(100)
                    )
            else:
                outdir=os.path.join(
                'your_path/results170817',
                str(type_noise)+'_'+str(type_wf)+'_SNR'+str(int(SNR))+'_SEED'+str(int(seed_number))+'_'+str(100)
                    )
    print('The injection is a '+type_noise+' noise and a '+type_wf+' signal.')


    duration=1 #sec 
    sampling_frequency=8192 #Hz

    #getting the timeseries from and FFT it
    if type_noise=='GWOSC':
        timeseries_segment=read_timeseries(seed_number)
        frequency_array, frequency_domain_noise_aLIGO=frequency_from_time(timeseries_segment, t_0, duration)
        frequency_domain_noise=recolour_arr(frequency_domain_noise_aLIGO, frequency_array, type_recolour='strain')

    if type_noise=='GWINC':
        curve_ce = gwinc.load_budget("CE1").run()
        curve_ce_freq=curve_ce.freq
        curve_ce_freq_max=np.max(curve_ce_freq)
        curve_ce_freq_min=np.min(curve_ce_freq)
        frequency_array_ce=curve_ce_freq
        psd_array=upsample(frequency_array_ce, curve_ce.psd, sampling_frequency, duration)
        asd_array=upsample(frequency_array_ce, curve_ce.asd, sampling_frequency, duration)
        asd_noise_only=asd_array #redefining for later
        frequency_array=np.linspace(0, sampling_frequency, sampling_frequency)
        psd_gwinc_ce=bilby.gw.detector.PowerSpectralDensity(frequency_array=frequency_array, psd_array=psd_array)
        #missing all the frequency and time domain strain

    #getting the frequency domain signal to inject
    NR_constant_parameters = dict(
        geocent_time=t_0, ra=ra, dec=dec, psi=np.pi/2,
        t_0=t_0, loudness=1
    )

    waveform_arguments =  dict(
        t_0=t_0,
        geocent_time=t_0,
        ra=ra,
        dec=dec,
        psi=np.pi/2,
        phi_ref=np.pi,
        NRPMw_t_coll=700,
        loudness=NR_constant_parameters['loudness'],
    )
    injection_parameters_wo_distance=read_metadata(wf_path_self)
    params=NR_constant_parameters|waveform_arguments

#damped sinusoid
    w_0=0.45 #sum of w_i=1
    w_1=0.45
    w_2=1-w_0-w_1
    f_0=3050 #peak frequency, uniform prior between 1000, 5000
    f_1=2400
    f_2=4000
    T_0=10**(0.1) #damping time , uniform log10T prior between -4, 0.3
    T_1=10**(-0.3)
    T_2=10**(-0.3)
    phi_0=0 #initial phases, uniform prior between -pi and pi
    phi_1=np.pi/2
    phi_2=2*np.pi/3
    alpha_0=-1.6 #frequency drift terms, uniform prior between -6.4 and 6.4
    alpha_1=-3.3
    alpha_2=5

    injection_parameters_wo_amplitude = {
            "t_0": waveform_arguments["t_0"],

            "w_0": w_0,
            "w_1": w_1,

            "f_0": f_0,
            "f_1": f_1,
            "f_2": f_2,

            "T_0": T_0,
            "T_1": T_1,
            "T_2": T_2,

            "phi_0": phi_0,
            "phi_1": phi_1,
            "phi_2": phi_2,

            "alpha_0": alpha_0,
            "alpha_1": alpha_1,
            "alpha_2": alpha_2
    }
    ###

    if type_noise=='GWOSC':
        if type_wf=='NRPMw':
            optimal_distance, ifos= distance_from_SNR(frequency_array, SNR, sampling_frequency, params, injection_parameters_wo_distance, duration, type_noise, strain_data_only_noise=frequency_domain_noise, timesegment=timeseries_segment)
        if type_wf=='Easter':
            #get the injection parameters and the optimal amplitude for the SNR
            optimal_logB, ifos=amplitude_from_SNR(frequency_array, SNR, sampling_frequency, params, injection_parameters_wo_amplitude, duration, type_noise, waveform_arguments, strain_data_only_noise=frequency_domain_noise, timesegment=timeseries_segment)
            equivalent_distance, ifos_dummy=distance_from_SNR(frequency_array, SNR, sampling_frequency, params, injection_parameters_wo_distance, duration, type_noise, strain_data_only_noise=frequency_domain_noise, timesegment=timeseries_segment)
            print("For SNR: "+str(SNR)+" the optimal logB is "+str(round(optimal_logB, 4))+" and the equivalent distance is "+str(round(equivalent_distance,4)))

    if type_noise=='GWINC':
        if type_wf=='NRPMw':
            optimal_distance, ifos=distance_from_SNR(frequency_array, SNR, sampling_frequency, params, injection_parameters_wo_distance, duration, type_noise, strain_data_only_noise=None)
        if type_wf=='Easter':
            #get the injection parameters and the optimal amplitude for the SNR
            optimal_logB, ifos=amplitude_from_SNR(frequency_array, SNR, sampling_frequency, params, injection_parameters_wo_amplitude, duration, type_noise, waveform_arguments, strain_data_only_noise=None)
            equivalent_distance, ifos_dummy=distance_from_SNR(frequency_array, SNR, sampling_frequency, params, injection_parameters_wo_distance, duration, type_noise, strain_data_only_noise=None)
            print("For SNR: "+str(SNR)+" the optimal logB is "+str(round(optimal_logB, 4))+" and the equivalent distance is "+str(round(equivalent_distance,4)))
    if type_wf=='NRPMw':
        injection_parameters=injection_parameters_wo_distance|{'distance': optimal_distance}
        dict_NRPMw=params|injection_parameters
        # frequency_array_16=np.linspace(0, 16384, 16384)
        # frequency_domain_NRPMw_signal=NRPMw_model(frequency_array_16, **dict_NRPMw)
        # strain_signal=(frequency_domain_NRPMw_signal['plus']+1j*frequency_domain_NRPMw_signal['cross'])#**(2)
        # strain_signal_label='NRPMw signal'
    if type_wf=='Easter':
        #get the easter model waveform as a strain complex function in the frequency domain based on the injection parameters
        injection_parameters=injection_parameters_wo_amplitude|{'logB': optimal_logB}
        # t_0=injection_parameters['t_0']
        # frequency_array_16=np.linspace(0, 16384, 16384)
        # time_array_16=np.linspace(t_0/1000, (t_0)/1000+duration, 16384)
        # time_domain_easter_signal=easter_model(time_array_16, **injection_parameters)
        # time_signal=(time_domain_easter_signal['plus']-1j*time_domain_easter_signal['cross'])
        # frequency_arr, frequency_domain_easter_signal=frequency_from_time(time_signal, t_0, duration)
        # strain_signal=frequency_domain_easter_signal
        # strain_signal_label='Damped sinusoid signal'

    #adding the signals together

    if type_noise=='GWOSC':
        # frequency_strain_noise=frequency_domain_noise
        # time_array, time_noise=time_from_frequency(frequency_strain_noise, t_0, duration)
        # welch_freqs, psd_welch=psd_from_welch_pycbc(np.real(time_noise), len(time_noise))
        welch_freqs, psd_welch=psd_from_welch_pycbc(timeseries_segment, 16384)
        recolour_welch_psd=recolour_arr(psd_welch, welch_freqs)

    #injecting the signal into some new ifos
    if type_noise=='GWINC': 
        samp_freq=16384-2/duration
    if type_noise=='GWOSC':
        samp_freq=16384/duration
    if type_wf=='NRPMw':
        waveform_generator_inject = bilby.gw.waveform_generator.WaveformGenerator(
            duration=duration,
            sampling_frequency=samp_freq, #the model should be sampled at 16kHz
            frequency_domain_source_model = NRPMw_model,
            waveform_arguments=params #params is everything NOT being parameterestimated ie the fixed priors
        )
    if type_wf=='Easter':
        #to inject the signal you need the easter model as a waveformgenerator object
        waveform_generator_inject = bilby.gw.waveform_generator.WaveformGenerator(
            duration=duration,
            sampling_frequency=samp_freq, 
            time_domain_source_model = easter_model,
            waveform_arguments={'logB': optimal_logB, 't_0':injection_parameters['t_0']},
            start_time=t_0
        )
    if type_wf=='NRPMw':
        inject_params=dict_NRPMw
    if type_wf=='Easter':
        #the easter injection parameters ie w_i etc
        inject_params=injection_parameters_wo_amplitude|{'logB':optimal_logB}|NR_constant_parameters
    if type_noise=='GWOSC':
        ifos_injected, crop_freq_array=inject_signal_itk(outdir, frequency_array, sampling_frequency, duration, params['t_0'], waveform_generator_inject, inject_params, type_noise, strain_data_only_noise=frequency_domain_noise, timesegment=timeseries_segment, inject=inject)
    if type_noise=='GWINC':
        ifos_injected, crop_freq_array=inject_signal_itk(outdir, frequency_array, sampling_frequency, duration, params['t_0'], waveform_generator_inject, inject_params, type_noise, inject=inject)

    #the final arrays must be cropped to between 500 and 6000 Hz
    #they get cropped in the injection in the new ifos

    if type_wf=='NRPMw':
        prior_dict=prior_generate_NRPMw(injection_parameters, NR_constant_parameters, waveform_arguments, type_wf)
        inject_params.pop('loudness')
        inject_params.pop('mtot')
        inject_params.pop('q')
        inject_params.pop('NRPMw_df_2')
        inject_params.pop('lambda1')
        inject_params.pop('lambda2')

        filtered_params=inject_params
    if type_wf=='Easter':
        #the easter model is being recovered by the NRPMw model so the priors need to be the NRPMw parameters
        '''
        prior_dict=prior_generate_NRPMw(injection_parameters|{'equivalent_distance' : equivalent_distance}, NR_constant_parameters, waveform_arguments, type_wf)
        filtered_params={
            'distance':equivalent_distance,
            's1z':0,
            's2z':0,
            'cosi':1,
            'psi':NR_constant_parameters['psi'],
            'ra':NR_constant_parameters['ra'],
            'dec':NR_constant_parameters['dec'],
            'phi_ref':waveform_arguments['phi_ref'],
            'NRPMw_phi_pm':np.pi,
            't_0':NR_constant_parameters['t_0'],
            'NRPMw_t_coll':700
        }
        '''
        filtered_params={'logB':optimal_logB, 't_0':NR_constant_parameters['t_0'], 'ra':NR_constant_parameters['ra'], 'dec':NR_constant_parameters['dec'], 'geocent_time':NR_constant_parameters['t_0'], 'psi':NR_constant_parameters['psi']}
        prior_dict=prior_generate_Easter(filtered_params)
        print(prior_dict)
        # filtered_params=inject_params #only the waveform order and logB is in here
        # prior_dict=prior_generate_Easter(inject_params)
        #the priors used to recover the model but leave them the same as the NRPmw model as thats what we are recovering
        #will need to define the fixed ones that come from the NRPMw model as they wont be defined for the Easter waveform previously

    if type_noise=='GWINC':
        frequency_strain_noise_signal=None

    waveform_generator_recover=bilby.gw.waveform_generator.WaveformGenerator(
            duration=duration,
            sampling_frequency=16384-2/duration, #this might be wrong lowkey
            frequency_domain_source_model = NRPMw_model,
            #time_domain_source_model=easter_model,
            waveform_arguments=filtered_params
    )

    return {
        # 'strain_w_signal': frequency_strain_noise_signal,
        'freq_array': crop_freq_array,
        'prior_dict': prior_dict,
        'ifos':ifos_injected,
        'waveform_generator_MB': waveform_generator_recover,
        'injection_parameters': filtered_params
    }
                  
