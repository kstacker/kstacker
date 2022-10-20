# cython: language_level=3

import numpy as np

cimport cython
cimport numpy as np
from cython.parallel import prange
from libc.math cimport floor, sqrt
from libc.stdio cimport printf

np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
def photometry_preprocessed(double[:,:] image, double[:] x, double[:] y,
                            int upsampling_factor):
    """
    Return the flux at given positions on an image that has already been
    convoled and upsampled.

    Parameters
    ----------
    image : array
        Image containing the fluxes.
    x, y : array
        x/y arrays (in pixels).
    upsampling_factor : int
        Factor used for the upsampling.

    """
    cdef:
        size_t i, nx, ny, norbits, noutside=0, xpix, ypix

    nx = image.shape[0]
    ny = image.shape[1]
    norbits = x.shape[0]

    resarr = np.zeros(norbits, dtype=np.float64, order='C')
    cdef double [:] res = resarr

    for i in range(norbits):
        # grid for photutils is centered on pixels hence the - 0.5
        # position = np.array(position) - 0.5
        # xpix, ypix = ((position + 0.5) * upsampling_factor - 0.5).astype(int)
        xpix = int(x[i] * upsampling_factor - 0.5)
        ypix = int(y[i] * upsampling_factor - 0.5)

        if (xpix < 0) | (xpix >= nx) | (ypix < 0) | (ypix >= ny):
            noutside += 1
        else:
            res[i] = image[xpix, ypix]

    if noutside > 0:
        print(noutside, "values outside of the image")

    return resarr


cdef inline double interp(double arr[], double x, size_t size) nogil:
    cdef size_t x1 = <size_t>floor(x)
    cdef size_t x2 = x1 + 1
    # return  arr[x1] + ((arr[x2] - arr[x1]) / (x2 - x1)) * (x - x1)
    if x2 >= size - 1:
        return arr[size - 1]
    else:
        return  arr[x1] + (arr[x2] - arr[x1]) * (x - x1)


@cython.boundscheck(False)
@cython.wraparound(False)
def cy_compute_snr(double[:,:,::1] images,
                   double[:,::1] positions,
                   double[:,::1] bkg_profiles,
                   double[:,::1] noise_profiles,
                   float[:,:,::1] proj_matrices,
                   double r_mask,
                   double r_mask_ext,
                   double scale,
                   int size,
                   int upsampling_factor,
                   double[:,::1] out,
                   int num_threads):

    cdef:
        double x, y, xproj, yproj, signal, noise, snr, temp_d, sk, zk, sum_sk_on_zk2, sum_sk2_on_zk2, sum_one_on_zk2
        ssize_t i, k
        size_t xpix, ypix
        size_t half_size = size // 2
        size_t nx = images.shape[1]
        size_t ny = images.shape[2]

    for i in prange(proj_matrices.shape[0], nogil=True, schedule='static',
                    num_threads=num_threads):
        signal = 0.
        noise = 0.
        sum_sk_on_zk2 = 0.  # Sum of signal on variance in image k
        sum_sk2_on_zk2 = 0.  # Sum of signal**2 on variance in image k
        sum_one_on_zk2 = 0.  # Sum of one on variance in image k

        for k in range(images.shape[0]):
            x = positions[k, 0]
            y = positions[k, 1]

            # project positions
            xproj = proj_matrices[i, 0, 0]*x + proj_matrices[i, 0, 1]*y
            yproj = proj_matrices[i, 1, 0]*x + proj_matrices[i, 1, 1]*y

            xproj = xproj * scale
            yproj = yproj * scale

            # distance to the center
            temp_d = sqrt(xproj**2 + yproj**2)

            if temp_d <= r_mask or temp_d >= r_mask_ext:
                continue

            # convert position into pixel in the image
            xproj = xproj + half_size
            yproj = yproj + half_size

            # get signal value from the convolved image
            # grid is centered on pixels hence the - 0.5
            xpix = int(xproj * upsampling_factor - 0.5)
            ypix = int(yproj * upsampling_factor - 0.5)

            if (xpix >= 0) and (xpix < nx) and (ypix >= 0) and (ypix < ny):
                zk = interp(&noise_profiles[k, 0], temp_d, half_size)
                sk = images[k, xpix, ypix] - interp(&bkg_profiles[k, 0], temp_d, half_size)
                if zk != 0.:
                    sum_sk_on_zk2 = sum_sk_on_zk2 + sk / zk ** 2.
                    sum_one_on_zk2 = sum_one_on_zk2 + 1. / zk ** 2.
                    sum_sk2_on_zk2 = sum_sk2_on_zk2 + sk ** 2. / zk ** 2.

        # Signal that maximize the probability of detection in a gaussian noise (eq. 18 of Zackey)

        if sum_one_on_zk2 != 0:
            signal =  sum_sk_on_zk2 / sum_one_on_zk2
        else:
            signal = 0.
        # Minimization of the signal
        snr = - signal

        out[i, 0] = signal
        out[i, 1] = sum_one_on_zk2 # noise, for the classical SNR
        out[i, 2] = snr
