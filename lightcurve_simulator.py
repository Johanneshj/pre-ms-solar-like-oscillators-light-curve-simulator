import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from astropy.io import ascii
import os
import pandas as pd
import butterpy as bp
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Process 
import scipy.signal
from scipy.interpolate import interp1d
from astropy.timeseries import LombScargle
from astropy.coordinates import SkyCoord
from scipy.stats import skewnorm
import random
from numba import njit
import time as chronos
from scipy.signal import lfilter
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import seaborn as sns
sns.set(style="ticks", palette="muted", 
        rc={"xtick.bottom" : True, "ytick.left" : True})
plt.style.use(r'../low_mass_episodic_accretion/python_scripts/matplotlibrc.txt')

def signal(t, ell, νnl, mass, teff, lum):
    '''Based on formalism in De Ridder+ (2006).
    Simulates an exponentially dampened, but continously reexcited signal.
    Signal given as
    f(t) = exp(-η(t-nΔt)) [Bn sin(2π ν0 t) + Cn cos(2 π ν0 t)]
    and calculated as
    amp = A0 * decay + np.cumsum(kicks)
    where kicks are randomly generated numbers following a normal distribution

    Parameters
    ----------
    t : array 
        array of time values structured as
        np.arange(start, end, cadence) [days]
    η : float
        inverse dampening time [days]
    ν0 : float
        frequency of mode [1/days]
    cos : bool
        if True returns cosine wave, else sine
        (essentially to get rid of phase)
    Δt : float
        time between reexcitation (or kicks) [days]
        given here as η/100 (De Ridder+ 2006)
    rng : function
        numpy random number generator
    
    Returns
    -------
    amp*cos or amp*sin : list
        the simulated signal with time varying amplitude
    
    '''

    Δt = t[1] - t[0]

    η = calculate_linewidth(νnl, mass, teff, lum)

    A0 = mode_amplitude(spherical_degree=ell, νnl=νnl, lum=lum, mass=mass, teff=teff)     
    #print(A0) 

    sqrt_factor = np.sqrt(η*Δt)
    sig = A0 * sqrt_factor

    exp_factor = np.exp(-η * Δt)

    signal = np.zeros((len(t), len(sig)), dtype=float)

    for i in range(2):
        noise = np.random.normal(np.zeros_like(sig), sig, (len(t), len(sig)))

        amp = np.empty_like(noise)
        for j in range(noise.shape[1]):
            amp[:, j] = lfilter([1], [1, -exp_factor[j]], noise[:, j])

        if i == 0:
            cos = np.cos(2 * np.pi * t[:, None] * νnl[None, :]) 
            signal += amp*cos
        else:
            sin = np.sin(2 * np.pi * t[:, None] * νnl[None, :]) 
            signal += amp*sin
    #t1 = chronos.time()
    for j in range(signal.shape[1]):
        signal[:, j] = signal[:, j] / np.std(signal[:, j]) * A0[j]
    #t2 = chronos.time()
    #print('norm time:', t2-t1)

    signal -= np.median(signal, axis=0)
    signal = np.sum(signal, axis=1)
    
    return signal

def calculate_linewidth(νnl, mass, teff, lum):
    μHz_to_cpd = 1e-6 * 86400

    νnl /= ( np.ones_like(νnl) * μHz_to_cpd )
    νmax = calculate_νmax(mass, teff, lum) / μHz_to_cpd   

    α = -3.71e0 + 1.073e-3*teff + 1.883e-4*νmax
    Γα = -7.209e1 + 1.543e-2*teff + 9.101e-4*νmax
    ΔΓdip = -2.266e-1 + 5.083e-5*teff + 2.715e-6*νmax
    νdip = -2.190e3 + 4.302e-1*teff + 8.427e-1*νmax
    Wdip = -5.639e-1 + 1.138e-4*teff + 1.312e-4*νmax

    if α < 0 or Γα < 0 or ΔΓdip < 0 or νdip < 0 or Wdip < 0:
        Γ0 = 1.02 # muHz
        T0 = 436 # Kelvin
        lw = Γ0 * np.exp( (teff - 5777)/T0 ) * μHz_to_cpd
        line_widths = np.ones_like(νnl) * lw
    else:
        lnΓ = α*np.log(νnl/νmax) + np.log(Γα) + ( np.log(ΔΓdip) / ( 1 + ( 2*np.log(νnl/νdip) / np.log(Wdip/νmax) )**2 ) )
        line_widths = np.exp(lnΓ) * μHz_to_cpd

    νnl *=  np.ones_like(νnl) * μHz_to_cpd

    return line_widths

def granulation(t, lum, mass, teff):
    """
    Simulates a two-component granulation signal.
    Based on the formalism in De Ridder+ (2006) and empirical evidence from Kallinger+ (2014).

    Parameters
    ----------
    t : array
        Array of time values structured as np.arange(start, end, cadence) [days].
    lum : float
        Stellar luminosity in solar units.
    mass : float
        Stellar mass in solar units.
    teff : float
        Stellar effective temperature [K].
    
    Returns
    -------
    granulation : ndarray
        Relative flux variation of two-component granulation.
    """

    mass_Sun = 1.0

    μHz_to_cpd = 1e-6 * 86400 
    
    Δt = (t[1] - t[0])

    νmax = calculate_νmax(mass=mass, teff=teff, lum=lum)
    
    # Granulation timescales
    tau = 1 / (np.array([0.317, 0.948]) * (νmax / μHz_to_cpd)**np.array([0.970, 0.992]) * μHz_to_cpd * 2 * np.pi)  # shape (2,)

    # Granulation amplitude
    Agran = 3710 * (νmax / μHz_to_cpd)**-0.613 * (mass / mass_Sun)**-0.26

    sqrt_factor = Agran * np.sqrt(Δt / tau)  # shape (2,)
    exp_factor = np.exp(-Δt / tau)           # shape (2,)

    N = len(t)
    noise = np.random.normal(0, 1, size=(N, 2)) * sqrt_factor  # shape (N, 2)

    gran = np.empty_like(noise)
    for i in range(2):
        gran[:, i] = lfilter([1], [1, -exp_factor[i]], noise[:, i])

    gran -= np.median(gran, axis=0)
    granulation = np.sum(gran, axis=1)
        
    return granulation  

def calculate_νmax(mass, teff, lum):
    '''Calculate νmax from the scaling relations
    νmax = (M/M_sun) * (Teff/Teff_Sun)^3.5 * (L/L_sun)^(-1) * νmax_sun

    Parameters
    ----------
    mass : float
        mass in units Msun
    teff : float
        effective temperatute in units Kelvin
    lum : float
        luminosity in units Lsun   
    μHz_to_cpd : float
        conversion factor from μHz to cycles per day

    Returns
    -------
    νmax : float
        frequency at maximum power from the scaling relations
    '''

    νmax_Sun = 3090 # muHz
    mass_Sun = 1
    teff_Sun = 5777
    lum_Sun = 1
    
    νmax = (mass/mass_Sun) * (teff/teff_Sun)**(3.5) / (lum/lum_Sun) * νmax_Sun

    μHz_to_cpd = 1e-6 * 86400
    νmax *= μHz_to_cpd

    return νmax

def calculate_amplitude(mass, lum):
    '''Calculate amplitude at νmax
    A = (L/M)^1.5 * A_sun

    Parameters
    ----------
    mass : float
        mass in units Msun
    lum : float
        luminosity in units Lsun   
    A_Sun : float
        amplitude of the Sun in ppm (Ball+ 2018)

    Returns
    -------
    amplitude : float
        amplitude at νmax
    '''
    A_Sun = 2.1

    exponent = np.random.uniform(0.7, 1.5)
    #exponent = np.linspace(0.7, 1.5, 100)

    #amp = np.max((lum/mass)**exponent * A_Sun)
    amp = (lum/mass)**exponent * A_Sun

    return amp

def calculate_envelope_width(νmax):
    '''Calculate envelope width (Mosser+ 2012)
    width = 0.66 μHz * (νmax/μHz)*0.88
    width = 0.66 * μHz_to_cpd * (νmax/cpd)**0.88

    Parameters
    ----------
    mass : float
        mass in units Msun
    lum : float
        luminosity in units Lsun   
    μHz_to_cpd : float
        conversion factor from μHz to cycles per day

    Returns
    -------
    Γ : float
        width of mode envelope
    '''
    μHz_to_cpd = 1e-6 * 86400

    Γ = 0.66 * (νmax/μHz_to_cpd)**0.88 / (2 * np.sqrt(2*np.log(2))) * μHz_to_cpd
    return Γ

def mode_amplitude(spherical_degree, νnl, lum, mass, teff):
    '''Calculate ampliutde of single mode ν_nl
    assuming Gaussian shaped envelope

    Parameters
    ----------
    νmax : float
        frequency at maximum power
    amp : float
        amplitude at νmax
    Γ : float
        width of mode envelope
    spherical_degree : float
        spherical degree
    '''

    μHz_to_cpd = 1e-6 * 86400

    νmax = calculate_νmax(mass, teff, lum)
    amp = calculate_amplitude(mass, lum)
    Γ = calculate_envelope_width(νmax)

    amp_νnl_squared = amp**2 * np.exp( -(1/2) * ( (νnl - νmax)/ Γ)**2 )

    scaling = np.array([1, 1.505, 0.620, 0.075])
    scaling = np.array([1, 1, 1, 1])
    unique_ell = np.unique(spherical_degree)
    scaling_array = np.concatenate([
            np.ones_like(spherical_degree[spherical_degree == ell]) * scaling[ell]
            for ell in unique_ell
        ])
    
    amp_νnl = np.sqrt(amp_νnl_squared) * scaling_array
    return amp_νnl

def calculate_noise(t, teff, position : bool):
    # -- Noise calc from TESS ATL -- #
    # Hey et al. 2024 #
    def calc_noise(
        imag,
        exptime,
        teff,
        e_lng=0,
        e_lat=30,
        g_lng=96,
        g_lat=-30,
        subexptime=2.0,
        npix_aper=4,
        frac_aper=0.76,
        e_pix_ro=10,
        geom_area=60.0,
        pix_scale=21.1,
        sys_limit=0,
    ):  

        omega_pix = pix_scale**2.0
        n_exposures = exptime / subexptime

        # electrons from the star
        megaph_s_cm2_0mag = 1.6301336 + 0.14733937 * (teff - 5000.0) / 5000.0
        e_star = (
            10.0 ** (-0.4 * imag)
            * 10.0**6
            * megaph_s_cm2_0mag
            * geom_area
            * exptime
            * frac_aper
        )
        e_star_sub = e_star * subexptime / exptime
        dlat = (abs(e_lat) - 90.0) / 90.0
        vmag_zodi = 23.345 - (1.148 * dlat**2.0)
        e_pix_zodi = (
            10.0 ** (-0.4 * (vmag_zodi - 22.8))
            * (2.39 * 10.0**-3)
            * geom_area
            * omega_pix
            * exptime
        )

        # e/pix from background stars
        dlat = abs(g_lat) / 40.0 * 10.0**0

        dlon = g_lng
        q = np.where(np.atleast_1d(dlon) > 180.0)
        if len(q) > 0:
            dlon = 360.0 - dlon

        dlon = abs(dlon) / 180.0 * 10.0**0
        p = [18.97338 * 10.0**0, 8.833 * 10.0**0, 4.007 * 10.0**0, 0.805 * 10.0**0]
        imag_bgstars = p[0] + p[1] * dlat + p[2] * dlon ** (p[3])
        e_pix_bgstars = (
            10.0 ** (-0.4 * imag_bgstars)
            * 1.7
            * 10.0**6
            * geom_area
            * omega_pix
            * exptime
        )

        # compute noise sources
        noise_star = np.sqrt(e_star) / e_star
        noise_sky = np.sqrt(npix_aper * (e_pix_zodi + e_pix_bgstars)) / e_star
        noise_ro = np.sqrt(npix_aper * n_exposures) * e_pix_ro / e_star
        noise_sys = 0.0 * noise_star + sys_limit / (1 * 10.0**6) / np.sqrt(
            exptime / 3600.0
        )
        noise2 = np.sqrt(
            noise_star**2.0 + noise_sky**2.0 + noise_ro**2.0 + noise_sys**2.0
        )
        # noise1 = np.sqrt(noise_star**2.0 + noise_sky**2.0 + noise_ro**2.0)
        return noise2
    # ------------------------------ #

    Δt = (t[1] - t[0]) * 24 * 60 * 60

    #rng = np.random.default_rng()

    # Draw magnitude from distribution resembling the magnitude distribution
    # of Elizabethson+ (2021)
    mag = skewnorm.rvs(-2.578251317682087, 
                       loc=15.281604353398574, 
                       scale=2.2959939442219195)
    
    mag = 4

    if position:
        ra = 287.21607602027086  # cepheus
        dec = 31.36334026749376
        coordinate = SkyCoord(ra, dec, unit='deg', frame='icrs')
        ecl = coordinate.geocentrictrueecliptic
        gal = coordinate.galactic

        noise = calc_noise(
            imag=mag,
            teff=teff,
            exptime=Δt,
            e_lng=ecl.lon.value,
            e_lat=ecl.lat.value,
            g_lng=gal.l.value,
            g_lat=gal.b.value,
            )
    else:
        noise = calc_noise(
            imag=mag, 
            exptime=Δt, 
            teff=teff
            )

    noise = noise*1e6

    noise_list = np.random.normal(0, noise, len(t))

    return noise_list, mag

def calculate_activity(t):
    ds = 40

    surface = bp.Surface()

    rng = np.random.default_rng()

    def log_uniform(low, high):
        return np.exp(rng.uniform(np.log(low), np.log(high)))

    Peq = np.exp(rng.normal(1.2302251754700018, 0.8748162388411769)) # Elizabethson
    lambda_min = rng.uniform(0, 40)
    lambda_max = lambda_min + rng.uniform(5, 80)
    inclination = rng.uniform(0, 90)

    a_lvl = rng.uniform(1, 10)
    Tcycle = log_uniform(1, 40)
    Toverlap = log_uniform(0.1, Tcycle) 

    prob = rng.uniform(0, 1)
    if prob <= 0.5:
        shear = log_uniform(0.1, 1)
    else:
        shear = 0.0

    ndays = t[-1] + 1 # To get rid of UserWarning: `time` array exceeds duration of regions computation.
    # No new spots will emerge after this time, and the light curve will relax back to unity.

    regions = surface.emerge_regions(
        ndays=ndays,
        activity_level=a_lvl, 
        cycle_period=Tcycle,
        cycle_overlap=Toverlap,
        max_lat = lambda_max,
        min_lat = lambda_min
        )

    lightcurve = surface.evolve_spots(
        time=t[::ds],
        inclination=inclination,
        period=Peq,
        shear=shear,
        )
    
    amp = 1e6
    
    wl = int(len(lightcurve.flux)/3) if int(len(lightcurve.flux)/3) % 2 == 1 else int(len(lightcurve.flux)/3) + 1


    lc_savgol = savgol_filter(lightcurve.flux, wl, 2)

    #fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    #ax.scatter(t[::ds], lightcurve.flux, s=4, c='k', label='lightcurve')
    #ax.scatter(t[::ds], lc_savgol, s=1, c='r', label='savgol')
    

    activity = (lightcurve.flux - lc_savgol)/lc_savgol * amp 

    activity = interp1d(t[::ds], activity, kind='linear', fill_value='extrapolate')(t)

    #activity = gaussian_filter1d(activity, sigma=ds/2)
    #ax.scatter(t, activity, s=4, c='magenta', label='lightcurve', zorder=-1)
    #ax.set_xlim(50,75)
    #plt.show()
    #fig.savefig('activity.png')

    return activity, Peq

def fft(t, s):
    ls = LombScargle(t, s, normalization='psd')

    dt = (t[1] - t[0])
    f_Nyq = 1/(2*dt)

    freq = np.arange(1/t[-1], f_Nyq, 1/t[-1])
    psd = ls.power(freq, normalization='psd', method='fast', 
                   assume_regular_frequency=True)
    
    d_nu = freq[1] - freq[0]
    normfactor = np.sum((s - np.nanmean(s))**2) / len(t) / sum(psd)
    
    psd *= normfactor / (d_nu)

    """frequency_domain_energy = np.sum(psd)
    time_domain_energy = np.sum(np.abs(s)**2) * dt

    print(time_domain_energy, frequency_domain_energy, time_domain_energy/frequency_domain_energy)
    if np.isclose(time_domain_energy, frequency_domain_energy):
        print("Parseval's theorem is verified.")
    else:
        print("Parseval's theorem is not verified.")
    """
    freq_muHz = freq * (1e6/86400)

    return freq_muHz, psd

def full_signal(length, cadence, final_mass, acchist, profile_number, ells, 
                gran : bool, noise : bool, activity : bool, do_plot : bool, save_lc : bool, 
                delta_mag : bool, dest_folder, ID=0):

    
    freq_data_folder = 'freq_files'
    model = f"mass-{final_mass}Msun_acchist-{acchist}"
    data_folder = f'../data_may_2025/{model}'
    freq_data_model_folder = os.path.join(data_folder, freq_data_folder)
    freq_file_path = os.path.join(freq_data_model_folder, f'profile{profile_number}-freqs.dat')

    hist_model_numbers = pd.read_csv(os.path.join(data_folder, f"{model}_value_model_number.txt"), header=None).values
    Teffs = pd.read_csv(os.path.join(data_folder, f"{model}_value_log_Teff.txt"), header=None).values
    Lums = pd.read_csv(os.path.join(data_folder, f"{model}_value_log_L.txt"), header=None).values
    Ms = pd.read_csv(os.path.join(data_folder, f"{model}_value_star_mass.txt"), header=None).values
    Loggs = pd.read_csv(os.path.join(data_folder, f"{model}_value_log_g.txt"), header=None).values

    profile_folder = os.path.join(data_folder, 'profile_folder')
    index_file = np.genfromtxt(os.path.join(profile_folder, f"index_file.txt"))
    index_model_number = np.array(index_file[:, 1], dtype=int)
    profile_numbers = np.array(index_file[:, 0], dtype=int)

    model_number = index_model_number[np.where(profile_numbers == profile_number)[0]]
    #print(model_number)

    teff = 10**Teffs[np.where(hist_model_numbers == model_number)[0]][0]
    lum = 10**Lums[np.where(hist_model_numbers == model_number)[0]][0]
    mass = Ms[np.where(hist_model_numbers == model_number)[0]][0]
    g = 10**Loggs[np.where(hist_model_numbers == model_number)[0]][0]

    #freq_file_path = os.path.join(freq_data_model_folder, freq_file_name)
    freq_file = pd.read_table(freq_file_path, skiprows=5, sep='\s+')

# ---------------- calculate signal ---------------- #
    t = np.arange(0, length+6, cadence)
    μHz_to_cpd = 1e-6 * 86400
    s = 0

    #print(f'start {model}')

    if activity == True:
        #t1 = chronos.time()
        activity, P_rot = calculate_activity(t)
        s += activity
        #t2 = chronos.time()
        #print('rot. calc. time:', t2-t1)
    else:
        P_rot = 0
    
    νnl = np.array(freq_file['Re(freq)'].values * (μHz_to_cpd))
    ell = np.array(freq_file['l'].values)

    nbins = 30
    bin_edges = np.linspace(0, len(t), nbins+1, dtype=int)
    F = np.zeros_like(t, dtype=float)    

    for i in range(nbins):
        #t1 = chronos.time()   
        i_start = bin_edges[i]
        i_end = bin_edges[i+1]
        t_bin = t[i_start:i_end]

        Fbin = signal(t_bin, ell, νnl, mass, teff, lum)

        if P_rot > 0:
            P_rot_sun = 28 # 1/days
            frac = P_rot/P_rot_sun
            rotation_scaling = (1 + frac)**(-1.2)
            #print(f'rotation scaling: {rotation_scaling}')
            Fbin *= rotation_scaling
        
        F[i_start:i_end] = Fbin

        #t2 = chronos.time()
        #print('signal calc. time:', t2-t1)
    s += F

    if gran == True:
        #t1 = chronos.time()
        g = granulation(t, lum, mass, teff)
        s += g
        #t2 = chronos.time()
        #print('gran. calc. time:', t2-t1)
    
    if noise == True:
        noise, mag = calculate_noise(t=t, teff=teff, position=False)
        s += noise
        #print(f'noise added')
    else:
        mag = None
        #print(f'no noise')
        
    s -= np.median(s)

    relax_time = 6
    idx = np.where( (t >= relax_time) )

    final_scaling = 1e0
    t, s = t[idx]-relax_time, s[idx] * final_scaling
    F = F[idx]

    if do_plot == True:
        plot(t, s, g, F, 
             final_mass, acchist, profile_number, 
             delta_mag=delta_mag, dest_folder=dest_folder, ID=ID)

    if save_lc == True:
        μHz_to_cpd = 1e-6 * 86400
        νmax = calculate_νmax(mass=mass, teff=teff, lum=lum) / μHz_to_cpd
        model = f'{model}_prof-{profile_number}'
        if not os.path.exists(f'{dest_folder}'):
            os.mkdir(f'{dest_folder}')
        

        if ID > 0:
            if not os.path.exists(f'{dest_folder}/data'):
                os.mkdir(f'{dest_folder}/data')
            filename = f'{dest_folder}/data/varsource_{ID:09d}.txt'
        else:
            if not os.path.exists(f'{dest_folder}/{model}'):
                os.mkdir(f'{dest_folder}/{model}')
            filename = f'{dest_folder}/{model}/{model}.txt'

        with open(filename, 'w') as f:
            '''f.write("#---Simulated pre-ms lightcurve---#\n")
            f.write(f"# Observation length : {int(t[-1]+1)} days\n")
            f.write(f"# Teff : {int(teff[0])} Kelvin\n")
            f.write(f"# numax : {int(νmax[0])} microHz\n")

            if P_rot:
                f.write(f"# Rotation period : {P_rot} days\n")
            else:
                f.write(f"# Rotation period : 0 days\n")
            
            if mag:
                f.write(f"# Magnitude : {mag}\n")
            else:
                f.write(f"# Magnitude : None\n")

            f.write("#\n")'''
            #f.write("# Column 1: Time (days)\n")

            #if delta_mag:
            #    f.write("# Column 2: Relative mag\n")
            #else:
            #    f.write("# Column 2: Relative flux (ppm)\n")
            #f.write("#----------------------------------\n")

            if delta_mag:
                s = -2.5*np.log10(1.0 + s/1e6)
            
            t *= 24 * 60 * 60 # convert to seconds
            for time, flux in zip(t, s):
                f.write(f"{time}\t{flux}\n")   

    #print(f'finish {model}')

def plot(t, s, g, F, final_mass, acchist, profile_number, delta_mag, dest_folder, ID=0):
    model = f"mass-{final_mass}Msun_acchist-{acchist}_prof-{profile_number}"

    def smooth(y, width):
        return gaussian_filter1d(y, width)
    
    fig, axs = plt.subplots(2, 1, figsize=(6,8), dpi=500)

    if delta_mag==True:
        axs[0].plot(t, -2.5*np.log10(1+s/1e6), '.', ms=1, c='k')
        axs[0].set_xlabel('time [d]')
        axs[0].set_ylabel(r'amplitude [$\Delta$mag]')
        axs[0].invert_yaxis()
    else:
        axs[0].plot(t, s, '.', ms=1, c='k')
        axs[0].set_xlabel('time [days]')
        axs[0].set_ylabel('amplitude [ppm]')
        axs[0].legend(frameon=False, loc='upper left')

    freq, psd = fft(t, s)
    
    axs[1].plot(freq, psd, c='grey', lw=1, zorder=0, label='full PSD')
    axs[1].plot(freq, smooth(psd,10), c='k', lw=1, zorder=2, label='smoothed PSD')

    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_xlim(0.1, max(freq))
    axs[1].set_ylim(min(psd)/10, max(psd)*10)
    #axs[1].set_ylim(, max(psd)*10)

    freq, psd = fft(t, F)
    
    axs[1].plot(freq, psd, c='green', lw=1, zorder=1, label='oscillations')
    axs[1].legend(frameon=False, loc='upper right')
    
    axs[1].set_xlabel(r'frequency $[\mu\rm{Hz}]$')
    axs[1].set_ylabel(r'PSD [$\rm{ppm}^2$ $\mu$$\rm{Hz}^{-1}$]')

    if not os.path.exists(os.path.join(dest_folder, 'figures')):
        os.mkdir(f'{dest_folder}/figures')

    if ID > 0:
        fig.savefig(f'{dest_folder}/figures/varsource_{ID:09d}.png')
    else:
        fig.savefig(f'{dest_folder}/figures/{model}.png')

    plt.close()

full_signal(
        length=2.1*365, cadence=25/(60*60*24), 
        final_mass=1.4, 
        acchist=15, 
        profile_number=500, 
        ells=[0,1,2,3], 
        gran=True, noise=False, activity=True, do_plot=True, save_lc=True, delta_mag=False,
        dest_folder=f'./', ID=1
    )


