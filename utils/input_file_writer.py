import argparse
import Cobra
import numpy as np



def read_par(par_file):
    """
    Read a pulsar par file and return a dictionary of P0, PB and A1 values
    """
    # Initialize an empty dictionary to store the values
    data_dict = {}

    # Read the file line by line
    with open(par_file, "r") as file:
        for line in file:
            # Split each line by whitespace
            parts = line.split()

            # Check if the line has the expected format
            if len(parts) >= 2:
                key = parts[0]  # The first item is the key
                value = parts[1]  # The second item is the value

                # Store PB and A1 values directly
                if key == "PB":
                    data_dict["PB"] = float(value)
                elif key == "A1":
                    data_dict["A1"] = float(value)
                # Compute P0 from F0
                elif key == "F0":
                    data_dict["P0"] = 1.0 / float(value)

    return data_dict


def read_bestprof(bestprof_file):

    # Initialize an empty dictionary to store the values
    data_dict = {}

    # Read the file line by line
    with open(bestprof_file, "r") as file:
        for line in file:
            # Split each line by whitespace
            parts = line.split('=')
    
            # Check if the line has the expected format
            if len(parts) >= 2:
                key = parts[0].strip()  # The part before '=' is the key
                value = parts[1].split()[0]  # The part after '=' is the value (take only the first value if +/- exists)

                #print(key)
                # Store P_orb(s) and asin(i)/c directly
                if key == "# P_orb (s)":
                    data_dict["PB"] = float(value) / 86400.0 # Convert from seconds to days
                elif key == "# asin(i)/c (s)":
                    data_dict["A1"] = float(value)
                # Compute P0 from P_bary (ms), converting to seconds
                elif key == "# P_bary (ms)":
                    P_bary_seconds = float(value) * 1e-3  # Convert ms to seconds
                    data_dict["P0"] = P_bary_seconds

    print(data_dict)
    # Display the dictionary
    return data_dict



def makeCandidate(input_files, param_file, time_span, file_tag):
    """
    This function creates a Cobra candidate file with default priors set based on initial parameters obtained either from a par file or bestprof file
    """

    if '.bestprof' in param_file:
        initial_params = read_bestprof(param_file)
    elif '.par' in param_file:
        initial_params = read_par(param_file)


    PB = initial_params['PB'] 
    A1 = initial_params['A1']
    P0 = initial_params['P0']


    #Get all dat files
    dat_files = input_files.split(' ')

    # Initiate Cobra features
    s = Cobra.Search("double")
    for dat_file in dat_files:
        s.addDatFile(dat_file[:-4])

    # Calculate intermediate parameters
    sinOrbit = np.sin(2*np.pi*np.linspace(0, 1, 10001))
    
    BinaryPeriod = initial_params['PB']*24*60*60


    blins = []
    bstds = []
    for i in range(100):
        BinaryPhase = np.random.uniform(0, 2 * np.pi)
        BinaryPhase -= 2*np.pi * (s.DatFiles[0].BaseTime[0])/BinaryPeriod
        BinaryPhase = BinaryPhase % (2*np.pi)
        bsum, blin, bstd = s.CircSum(sinOrbit, BinaryPeriod, BinaryPhase, interpstep=1024)
        blins.append(blin)
        bstds.append(bstd)

    print("Mean bstd value: {}".format(np.mean(np.asarray(bstds, dtype=float))))
    print("Mean blin value: {}".format(np.mean(np.asarray(blin, dtype=float))))

    A1_input = A1 * np.mean(np.asarray(bstds, dtype=float)) 
    P0_input = P0 - A1*P0*np.mean(np.asarray(blins, dtype=float))

    print(np.log10(PB), np.log10(A1_input), P0_input)
    A1_ip = np.log10(A1_input) 
    PB_ip = np.log10(PB) 

    #Calculate uncertainties to use

    dF0 = 1.0/(float(time_span)) # Uncertainty of up to 1 rotation in the given time span
    dP0 = dF0 * float(initial_params['P0'])
    dP0_input = dP0

    # Uncertainties on Pb and A1 are hard-coded
    dA1_input = 0.5
    dPb_input = 0.5
     
    with open("{}_cobra_input_{}.dat".format(file_tag, time_span),"w") as f:
        f.write("Phase -0.5 0.5\n")
        f.write("Width -2 0\n")
        f.write("Period {} {}\n".format(P0_input, dP0_input))
        f.write("CircBinary {} {} {} {}\n".format(PB_ip - dPb_input, PB_ip + dPb_input, A1_ip - dA1_input, A1_ip + dA1_input))

    f.close()


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Write out a candidate file for COBRA based on given input dedispersed files and a pulsar par or PRESTO bestprof file')
    parser.add_argument('--dat_files', required=True, help='List of all time ordered dedispersed files (spaced)')
    parser.add_argument('--param_file', required=True, help='Par file or bestprof file to use for guessing initial parameters')
    parser.add_argument('--time_span', required=True, help='Time span of all dat files input in seconds')
    parser.add_argument('--file_tag', required=True, help='Output candidate file tag')
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_arguments()
    makeCandidate(args.dat_files, args.param_file, args.time_span, args.file_tag) 
      
