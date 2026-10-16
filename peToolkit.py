#this code will take the ASD from injectToolkit.py and the do PE on it to recover the signal

#This code is made to run with Dynesty 2.1.1, Bilby 2.6.0 and Numpy 1.26.4. May be compatible with other versions. 

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
from scipy.fft import fft, fftfreq
from scipy import signal
from gwpy.timeseries import TimeSeries
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import importlib

def do_pe(SNR,
                seed,
                t_0,
                outdir,
                label,
                noise_path,
                wf_path,
                type_noise,
                type_wf, distance_percent,
                inject,
                itk_script):
    itk = importlib.import_module(itk_script)
    output_injectToolkit=itk.main_inject(SNR, seed, t_0, noise_path, wf_path, type_noise, type_wf, distance_percent/100, inject)

    prior=output_injectToolkit['prior_dict']
    ifos=output_injectToolkit['ifos']
    waveform_generator_MB=output_injectToolkit['waveform_generator_MB']
    injection_parameters=output_injectToolkit['injection_parameters']

    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(ifos, waveform_generator_MB)

    result = bilby.run_sampler(
        likelihood,
        prior,
        sampler="dynesty",
        npoints=500,
        walks=5,
        nact=3,
        injection_parameters=injection_parameters,
        outdir=outdir,
        label=label,
        result_class=bilby.gw.result.CBCResult,
        npool=2
     )

def do_rand(
                seed,
                t_0,
                outdir,
                label,
                noise_path,
                type_noise,
                type_wf, distance_percent):
    output_injectToolkit=rtk.main_inject(seed, t_0, noise_path, type_noise, type_wf, distance_percent/100)

    prior=output_injectToolkit['prior_dict']
    if prior=="no":
        return
    ifos=output_injectToolkit['ifos']
    waveform_generator_MB=output_injectToolkit['waveform_generator_MB']
    injection_parameters=output_injectToolkit['injection_parameters']

    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(ifos, waveform_generator_MB)

    result = bilby.run_sampler(
        likelihood,
        prior,
        sampler="dynesty",
        npoints=500,
        walks=5,
        nact=3,
        injection_parameters=injection_parameters,
        outdir=outdir,
        label=label,
        result_class=bilby.gw.result.CBCResult,
        npool=2,
        seed=int(seed_number)
     )
