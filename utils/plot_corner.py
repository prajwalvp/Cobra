import argparse
import numpy as np
import corner
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate a corner plot from MCMC samples.")
    parser.add_argument("output_basename", type=str, help="Base filename for input/output files.")
    parser.add_argument("param_file", type=str, help="File containing parameter limits.")
    parser.add_argument("mode", choices=["derived", "sampled"], help="Plot mode: 'derived' or 'sampled'.")
    return parser.parse_args()

def load_parameter_limits(param_file):
    with open(param_file) as f:
        data = f.readlines()
    
    pmin = np.zeros(6, dtype=float)
    pmax = np.zeros(6, dtype=float)
   
    pmin[0], pmax[0] = map(float, data[0].strip().split()[1:3])
    pmin[1], pmax[1] = map(float, data[1].strip().split()[1:3])
    P, dP = map(float, data[2].strip().split()[1:3])
    pmin[2], pmax[2] = P - dP, P + dP
    pmin[3], pmax[3] = map(float, data[3].strip().split()[3:5])
    pmin[4], pmax[4] = 0, 2 * np.pi
    pmin[5], pmax[5] = map(float, data[3].strip().split()[1:3])
    
    return pmin, pmax

def plot_corner(output_basename, pmin, pmax, derived_params=False, labels=None, nwalkers=30, ndims=6):
    initial_flat_samples = np.loadtxt(f"{output_basename}_chain.txt")
    raw_chains = initial_flat_samples.reshape(-1, nwalkers, ndims)
    logL_values = np.loadtxt(f'{output_basename}_log_prob.txt').reshape(-1, nwalkers)
    cut = max(np.mean(logL_values, axis=0)) - 5.0 
    new_chains = raw_chains[:, np.mean(logL_values, axis=0) > cut, :]
    new_chains_flattened = new_chains.reshape(-1, ndims)

    original_samples = np.zeros_like(new_chains_flattened)
    for i in range(ndims):
        if i == 4:
            original_samples[:, i] = new_chains_flattened[:, i] * (pmax[i] - pmin[i]) % (2 * np.pi) + pmin[i]
        elif i == 0:
            original_samples[:, i] = new_chains_flattened[:, i] * (pmax[i] - pmin[i]) % 1 - 0.5 + pmin[i]
        else:
            original_samples[:, i] = new_chains_flattened[:, i] * (pmax[i] - pmin[i]) + pmin[i]
    
    if derived_params:
        derived = np.loadtxt(f"{output_basename}_derived_params.txt")
        combined_data = np.hstack((original_samples, derived))
    else:
        combined_data = original_samples
    
    fig = corner.corner(combined_data, labels=labels, show_titles=True, title_fmt='.12f')
    plt.show()

def main():
    args = parse_arguments()
    pmin, pmax = load_parameter_limits(args.param_file)
    labels_sampled = ["$\phi$", "$log_{10} W$", "P(s)", "$log_{10} x(s)$", "$\psi$", "$log_{10} P_b(d)$"]
    labels_derived = labels_sampled + ["$\phi_{True}$", "$log_{10} W$", "$P_{True}$", "$x_{True}$ (lt-s)", "$\psi(0-1)$", "$log_{10} P_{b(True)}(d)$"]
    labels = labels_derived if args.mode == "derived" else labels_sampled
    
    plot_corner(args.output_basename, pmin, pmax, derived_params=(args.mode == "derived"), labels=labels, nwalkers=30)

if __name__ == "__main__":
    main()

