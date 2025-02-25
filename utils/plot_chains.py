import sys
import numpy as np
import corner
import matplotlib.pyplot as plt

def plot_corner(output_basename, pmin, pmax, derived_params=False, labels=None):
    """
    Generate a corner plot from the flattened samples, transforming parameters back to original scale
    based on specific wrapping conditions.
    
    Parameters:
        output_basename : str
            The base filename used for saving outputs.
        pmin : list or array
            Minimum values for each parameter's original range.
        pmax : list or array
            Maximum values for each parameter's original range.
        derived_params : bool, optional
            If True, include derived parameters in the corner plot.
        labels : list of str, optional
            Labels for each parameter in the plot.
    """
    # Load the flattened samples
    flat_samples = np.loadtxt(f"{output_basename}_chain.txt")
   
    
 
    # Unscale parameters according to the transformations used in log_probability
    original_samples = np.zeros_like(flat_samples)
    for i in range(flat_samples.shape[1]):
        if i == 4:  # Assuming parameter 4 is wrapped to [0, 2 * pi]
            original_samples[:, i] = flat_samples[:, i] * (pmax[i] - pmin[i]) % (2 * np.pi) + pmin[i]
        elif i == 0:  # Assuming parameter 0 is wrapped to [0, 1]
            original_samples[:, i] = flat_samples[:, i] * (pmax[i] - pmin[i]) % 1 - 0.5  + pmin[i] 
        else:  # Other parameters with a direct transformation
            original_samples[:, i] = flat_samples[:, i] * (pmax[i] - pmin[i]) + pmin[i]
    
    # Load derived parameters if specified
    if derived_params:
        derived = np.loadtxt(f"{output_basename}_derived_params.txt")
        # Combine original parameters and derived parameters
        combined_data = np.hstack((original_samples, derived))
    else:
        combined_data = original_samples
    
    raw_log_prob = np.load(f'{output_basename}_raw_log_prob.npy')
    raw_log_prob_reshaped = raw_log_prob.reshape(-1,6)
    raw_chains = np.load(f'{output_basename}_raw_chain.npy')
    raw_chains_reshaped = raw_chains.reshape(-1,6)
    derived_params = np.load(f'{output_basename}_raw_derived_params.npy')
    derived_params_reshaped = derived_params.reshape(-1,6)
    # Get the best derived parameters based on highest logL value
    best_derived_parameters = derived_params_reshaped[np.argmax(raw_log_prob_reshaped)]
    best_sampling_parameters = raw_chains_reshaped[np.argmax(raw_log_prob_reshaped)]

    print(best_derived_parameters)
    print(best_sampling_parameters)


    # Generate the corner plot
    fig = corner.corner(combined_data, labels=labels, show_titles=True, title_fmt='.12f')
    
    # Display the plot
    plt.show()


with open(sys.argv[2]) as f:
    data = f.readlines()


pmin = np.zeros(6, dtype=float)
pmax = np.zeros(6, dtype=float)

# Set default phase range
pmin[0] = -0.5
pmax[0] = 0.5


# Set default width range
pmin[1] = -2
pmax[1] = 0

# Set period range

P = float(data[0].strip(' \n').split(' ')[1])
dP = float(data[0].strip(' \n').split(' ')[2])
pmin[2] = P - dP 
pmax[2] = P + dP

# Set x range
pmin[3] = float(data[1].strip(' \n').split(' ')[3])
pmax[3] = float(data[1].strip(' \n').split(' ')[4])

# Set default binary phase range
pmin[4] = 0
pmax[4] = 2*np.pi

# Set Pb range
pmin[5] = float(data[1].strip(' \n').split(' ')[1])
pmax[5] = float(data[1].strip(' \n').split(' ')[2])

labels = ["$\phi$","$log_{10} W$", "P(s)","$log_{10} x(s)$", "$\psi$", "$log_{10} P_b(d)$"]  # Replace with actual parameter names
labels_derived = ["$\phi$","$log_{10} W$", "P(s)","$log_{10} x(s)$", "$\psi$", "$log_{10} P_b(d)$","$\phi_{True}$","$log_{10} W","$P_{True}$","$x_{True}$ (lt-s)","\psi(0-1)","$log_{10} P_{b(True})(d)"]  # Replace with actual parameter names
if sys.argv[3] == 'derived':
    plot_corner(sys.argv[1], pmin, pmax, derived_params=True, labels=labels_derived)
elif sys.argv[3] == 'sampled':
    plot_corner(sys.argv[1], pmin, pmax, derived_params=False, labels=labels)    
