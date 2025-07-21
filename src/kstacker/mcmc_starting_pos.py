import h5py
import numpy as np

from astropy.io import ascii

def read_starting_file(params):
    work_dir = params.get_path("work_dir")
    p0 = ascii.read(f"{work_dir}/init_pos_mcmc.txt")
    bounds = ascii.read(f"{work_dir}/init_bounds_mcmc.txt")
    
    return p0.to_pandas().to_numpy(), list(zip(*bounds))

def build_mcmc_starting_position(params, n_walkers=28,  fixed_params=None, nbr_psf=1., fix_bounds=None):
    values_dir = params.get_path("values_dir")
    work_dir = params.get_path("work_dir")
    bounds = params.grid.bounds()
    
    with h5py.File(f"{values_dir}/res_grid.h5") as f:
        # note: results are already sorted by decreasing SNR
        results = f["Best solutions"][:]
    
    n_walkers = min(n_walkers, results.shape[0])
    
    # Define search range
    param_names = ["a", "e", "t0", "m0", "omega", "i", "theta_0"]
    if fixed_params == None: 
        final_param_names = param_names
        delta_param = {key: None for key in final_param_names}
        
        unfixed_param_indices = np.linspace(0,6,7,dtype=int)
    else :
        final_param_names = [key for key in param_names if key not in fixed_params]
        delta_param = {key: None for key in final_param_names}
        
        unfixed_param_indices = [index for index, keys in enumerate(param_names) if keys not in fixed_params]
    
    p0 = results[:n_walkers, unfixed_param_indices].copy()
    
    for i in range(len(delta_param)):
        index = param_names.index(final_param_names[i])
        delta_param[final_param_names[i]] = nbr_psf * (bounds[index][1]-bounds[index][0]) / params.grid.limits(final_param_names[i])[2]
    
    # Loop over each walker and each parameter to add
    # a small random perturbation to create independence between walkers
    for walker in range(n_walkers):
        for i in range(len(delta_param)):
            # Set up the initial flag for checking bounds
            in_bounds = False
    
            # Loop until the random perturbation is within the bounds
            while not in_bounds:
                # Generate random factor in range [-1, 1]
                random_factor = (np.random.rand() - 0.5) * 2
                perturbation = random_factor * delta_param[final_param_names[i]]
    
                # Add the perturbation to the parameter
                new_value = p0[5, i] + perturbation # This line use only one of the best output of  brute-force+gradiant (line 5)
                # new_value = p0[walker, param_index] + perturbation # This line use all the outputs of brute-force+gradiant (Doesn't work! before using this line, take into account modulos pi on Omega and omega)

                # Check if new values are in the bounds
                if bounds[param_names.index(final_param_names[i])][0] <= new_value <= bounds[param_names.index(final_param_names[i])][1]:
                    p0[walker, i] = new_value
                    in_bounds = True
    
    # mcmc configuration
    
    # Calculate the mean for each column
    means = np.mean(p0, axis=0)
    bounds = []
    if fix_bounds == None: 
        for  i in range(len(delta_param)):
            index = param_names.index(final_param_names[i])
            bounds.append((means[i] - delta_param[final_param_names[i]], means[i] + delta_param[final_param_names[i]]))
    else :
        fixed_bound_name = list(fix_bounds.keys())
        for  i in range(len(delta_param)):
            index = param_names.index(final_param_names[i])
            lower_bound = means[i] - delta_param[final_param_names[i]]
            upper_bound = means[i] + delta_param[final_param_names[i]]
            if final_param_names[i] in fixed_bound_name:
                if fix_bounds[final_param_names[i]]['bounds'] == 'both':
                    lower_bound = max(means[i] - delta_param[final_param_names[i]],fix_bounds[final_param_names[i]]['value'][0])
                    upper_bound = min(means[i] + delta_param[final_param_names[i]],fix_bounds[final_param_names[i]]['value'][1])
                if fix_bounds[final_param_names[i]]['bounds'] == 'lower':
                    lower_bound = max(means[i] - delta_param[final_param_names[i]],fix_bounds[final_param_names[i]]['value'])
                if fix_bounds[final_param_names[i]]['bounds'] == 'upper':
                    upper_bound = min(means[i] + delta_param[final_param_names[i]],fix_bounds[final_param_names[i]]['value'])
            bounds.append((lower_bound, upper_bound))
                
    p0 = np.array(p0)

    names = [param_names[unfixed_param_indices[i]] for i in range(len(unfixed_param_indices))]
    ascii.write(
        p0,
        f"{work_dir}/init_pos_mcmc.txt",
        names=names,
        format="fixed_width_two_line",
        overwrite=True,
    )
    
    ascii.write(
        bounds,
        f"{work_dir}/init_bounds_mcmc.txt",
        names=names,
        format="fixed_width_two_line",
        overwrite=True,
    )