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
def compute_snr(double[:,:,::1] images,
                double[:,::1] positions,
                double[:,::1] bkg_profiles,
                double[:,::1] noise_profiles,
                float[:,:,::1] proj_matrices,
                double r_mask,
                double scale,
                int size,
                int upsampling_factor,
                double[:,::1] out,
                int num_threads=0,
                int debug=0):

    cdef:
        double x, y, xproj, yproj, signal, noise, temp_d
        ssize_t i, k
        size_t xpix, ypix
        size_t half_size = size // 2
        size_t nx = images.shape[1]
        size_t ny = images.shape[2]

    for i in prange(proj_matrices.shape[0], nogil=True, schedule='static',
                    num_threads=num_threads):
        signal = 0
        noise = 0

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

            if temp_d <= r_mask:
                continue

            # convert position into pixel in the image
            xproj = xproj + half_size
            yproj = yproj + half_size

            # get signal value from the convolved image
            # grid is centered on pixels hence the - 0.5
            xpix = int(xproj * upsampling_factor - 0.5)
            ypix = int(yproj * upsampling_factor - 0.5)

            if (xpix >= 0) and (xpix < nx) and (ypix >= 0) and (ypix < ny):
                # add signal and correct for background (using pre-computed
                # background profile)
                signal = (signal + images[k, xpix, ypix] -
                          interp(&bkg_profiles[k, 0], temp_d, half_size))

                # add noise using pre-computed radial noise profile
                noise = noise + interp(&noise_profiles[k, 0], temp_d, half_size)**2

            if debug:
                printf("image %ld: xproj=%.2f, yproj=%.2f, xpix=%ld, ypix=%ld, "
                       "dist=%f, signal=%f, noise=%.2f\n",
                       k, xproj, yproj, xpix, ypix, temp_d, signal, noise)

        noise = sqrt(noise)
        out[i, 0] = signal
        out[i, 1] = noise
        if noise == 0:
            # if the value of total noise is 0 (i.e. all values of noise are 0,
            # i.e. the orbit is completely out of the image) then snr=0
            out[i, 2] = 0
        else:
            out[i, 2] = - signal / noise
