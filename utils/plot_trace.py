import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt


parameters = ["$\phi$","$log_{10} W$", "P(s)","$log_{10} x(s)$", "$\psi$", "$log_{10} P_b(d)$"]

def plot_trace(chain_file, logL_file):
    """
    Plot all chains and logL 
    """
    data = np.load(chain_file)
    logL_data = np.load(logL_file)
    
    # Number of parameters
    num_params = data.shape[2]
    
    # Create subplots
    fig, axes = plt.subplots(num_params+1, 1, figsize=(10, 12), sharex=True)
    
    for i in range(num_params):
        for walker in range(data.shape[1]):
            axes[i].plot(data[:, walker, i], alpha=0.6, lw=0.8)  # Plot each walker's trace
            
        axes[i].set_ylabel('{}'.format(parameters[i]))
        axes[i].grid(True, linestyle='--', alpha=0.5)
    
    
    for walker in range(data.shape[1]):
        axes[-1].set_ylabel('logL')
        axes[-1].plot(logL_data[:, walker], alpha=0.6, lw=0.8)  # Plot each walker's trace
    
    axes[-1].set_xlabel("Steps")
    fig.suptitle("MCMC Chains for {} Walkers".format(data.shape[1]), fontsize=14)
    plt.tight_layout()
    plt.show()


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Plot the chains of all walkers combined for each parameter')
    parser.add_argument('--raw_chains', required=True, help='input raw chain file')
    parser.add_argument('--raw_logL', required=True, help='input raw file with log probabilities')
    #parser.add_argument('--nwalkers', required=True, help='Number of walkers')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    plot_trace(args.raw_chains, args.raw_logL)
