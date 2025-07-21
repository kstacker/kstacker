class MCMCCstData:
    """
    Class to manage the constante parameters used in the MCMC.
    ts : list
        the 
    size : int
        numbers of images in the data set
    scale : float
        scale of one pixel
    fwhm : float
        radius of the aperture studied
    bounds : list
        searching bounds define on all 7 orbital param
    treated_image : numpy.ndarray
        all 5 pretreated images
    r_mask : float
        value of the inner radius of the mask 
    r_mask_ext: float
        value of the outer radius of the mask 
    r_vals : list
        x value for the bessel shape.
    j0_vals : list
        y value for the bessel shape.
    fixed_params : dict 
        fixe some orbital params, typical shape is {"a":50,"e":0.1}, the orbital param names must be : ["a", "e", "t0", "m0", "omega", "i", "theta_0"]. The default is None.
    cste_part_Likelihood : float
        store the constant component of the likelihood.
    
    Attributes
    ----------
    """
    def __init__(self):
        self.ts = None
        self.size = None
        self.scale = None
        self.fwhm = None
        self.bounds = None
        self.treated_image = None
        self.r_mask = None
        self.r_mask_ext = None
        self.r_vals = None
        self.j0_vals = None
        self.fixed_params = None
        self.PSF_shape = None
        self.cste_part_Likelihood = None
        self.data = None