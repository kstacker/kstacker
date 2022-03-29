"""
This module is used to create and manipulate phase masks. The creation of the
masks is handled by the IDL code SIMUL_FOURIER_SCAO (T. Fusco et al.), which is
called through an IDL session opened via pexpect.  IMPORTANT: this code calls
idl. Make sure to change the command and the PATH values according to your own
System.
"""

__author__ = "Mathias Nowak"
__email__ = "mathias.nowak@ens-cachan.fr"
__status__ = "Testing"


import math

import numpy as np


def create_ao_mask(n, d, dpix, wav, wind, t_ao, seeing, mag_gs, name):
    """
    This function creates a phase mask that represents the residual atmospheric phase errors after AO correction, using the IDL code. The mask is written in
    a file located in /simu_idl/outputs
    @param int n: desired size of the returned array
    @param int d: telescope diameter (in m)
    @param int dpix: telescope diameter (in pixels). Used for scaling.
    @param float wind: wind speed (m/s)
    @param float t_ao: integration time for each single image (decorrelation time for the AO)
    @param float seeing: seeing value (in arcsecond, at 0.5 microns)
    @param float mag_gs: magnitude of the guide star for the AO system (for wafefront sensor band)
    @param string name: a name for the mask that will be used as an identifier to create and read the files
    @return None
    """
    # name of the txt file that will be created
    filename = name + ".txt"

    # first part is to open an IDL session and tun the AO code
    command = (
        "idl -args "
        + str(n)
        + " "
        + str(d)
        + " "
        + str(dpix)
        + " "
        + str(wav)
        + " "
        + str(wind)
        + " "
        + str(t_ao)
        + " "
        + str(seeing)
        + " "
        + str(mag_gs)
        + " "
        + "'"
        + filename
        + "'"
    )
    #    command="gdl -args "+str(n)+" "+str(d)+" "+str(dpix)+" "+str(seeing)+" "+str(mag_gs)+" "+"'"+filename+"'"
    import pexpect  # to open an IDL session and run the OA code
    child = pexpect.spawn(command)  # this lauches IDL with some arguments for later use
    child.expect("IDL>")  # wait for IDL prompt
    child.sendline(
        '!PATH=!PATH+":/home/mnowak/coronagraph/simu_idl/SIMUL_FOURIER_SCAO/PRO"'
    )
    child.expect("IDL>")  # wait for IDL prompt
    child.sendline(
        '!PATH=!PATH+":/home/mnowak/coronagraph/simu_idl/SIMUL_FOURIER_SCAO"'
    )
    child.expect("IDL>")  # wait for IDL prompt
    child.sendline(
        ".r 'coronagraph/simu_idl/residu_caller.pro'"
    )  # run the caller. arg given when opening IDL session are passed to and processed by residu_caller
    child.expect("IDL>")  # wait until it's done
    print((child.before))
    print(
        (child.after)
    )  # very important to print that in order to detect a possible anomaly in the execution of the IDL code
    child.sendline("exit")  # exit IDL
    child.close()  # close the session

    return None


def my_static(wfe, m, wav):
    """
    My very own function to generate a static phase mask, following a 1/f^2 PSD.
    @param float wfe: wafefront error (in nm rms)
    @param int m: size (in pixel) of the mask
    @param float wav: wavelength of observation (in m), used to convert nm to rad
    @return float[m, m]: static phase mask (in rad)
    """
    n = (
        2 * m
    )  # we need to generate a mask of size 2*m to cut it in half at the end (avoid symetry created by FFT)
    # create the 2d power spectral density (PSD)
    psd = np.zeros([n, n])
    for k in range(n):
        for l in range(n):
            if (l != n // 2) or (k != n // 2):
                psd[k, l] = 1.0 / ((k - n // 2) ** 2 + (l - n // 2) ** 2)
    psd = np.fft.fftshift(psd)
    # multiply by gaussian white noise, and compute fft
    phase_noise = np.fft.ifft2(
        np.random.normal(size=[n, n]) * np.sqrt(np.sqrt(psd))
    )  # sqrt because psd is energy
    phase_noise = np.fft.fftshift(phase_noise)
    phases = np.imag(
        phase_noise[0 : n // 2, 0 : n // 2]
    )  # take only the first quadrant to avoid symetry
    # normalize to std=wfe
    phases = phases / np.std(phases) * wfe
    # convert to radians
    phases = phases * 2 * math.pi / wav

    return phases


def create_static_mask(wfe, n, d, wav, name):
    """
    This function creates a static phase mask that represents the phase errors induced by the mirror surface, using the IDL code. The mask is written in
    a file located in /simu_idl/outputs
    @param float wfe: wave front error (m rms)
    @param int n: desired size for the mask (in pixel)
    @param int d: pupil diameter (in pixel)
    @param float wav: wavelength (m)
    @param string name: a name for the mask that will be used as an identifier to create and read the files
    @return None
    """
    # name of the txt file that will be created
    filename = name + ".txt"

    # first part is to open an IDL session and tun the AO code
    command = (
        "/usr/local/itt/idl/idl/bin/idl -args "
        + str(wfe)
        + " "
        + str(n)
        + " "
        + str(d)
        + " "
        + str(wav)
        + " "
        + "'"
        + filename
        + "'"
    )
    #    command="gdl -args "+str(wfe)+" "+str(n)+" "+str(d)+" "+str(wav)+" "+"'"+filename+"'"
    import pexpect  # to open an IDL session and run the OA code
    child = pexpect.spawn(command)  # this lauches IDL with some arguments for later use
    child.expect("IDL>")  # wait for IDL prompt
    child.sendline(
        '!PATH=!PATH+":/home/mnowak/coronagraph/simu_idl/SIMUL_FOURIER_SCAO/PRO"'
    )
    child.expect("IDL>")  # wait for IDL prompt
    child.sendline(
        '!PATH=!PATH+":/home/mnowak/coronagraph/simu_idl/SIMUL_FOURIER_SCAO"'
    )
    child.expect("IDL>")  # wait for IDL prompt
    print((child.before))
    print(
        (child.after)
    )  # very important to print that in order to detect a possible anomaly in the execution of the IDL code
    child.sendline(
        ".r 'coronagraph/simu_idl/phase_statique_caller.pro'"
    )  # run the caller
    child.expect("IDL>")  # wait until it's done
    print((child.before))
    print(
        (child.after)
    )  # very important to print that in order to detect a possible anomaly in the execution of the IDL code
    child.sendline("exit")  # exit IDL
    child.close()  # close the session

    return None


def get_mask(name):
    """
    This functions is complementary to create_ao_mask and create_static_mask. It looks for a file corresponding the mask whose name is given as an argument,
    and load it as a python numpy array.
    @param string name: the name of the mask to load
    @return float[n, n] phase_mask: the phase mask, loaded as a numpy array. n should be given in the first line of the mask file.
    """
    # name of thetxt file
    filename = name + ".txt"

    # load the mask and reshape it
    data = np.loadtxt("./coronagraph/simu_idl/outputs/" + filename)
    phase_mask = data[1:].reshape(
        [data[0], data[0]]
    )  # mask is written as a 1 column file. First line is the size of the array

    return phase_mask
