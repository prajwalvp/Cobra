import File
import Candidate
import DatClass
import time

import pymultinest
import numpy as np
import pylab as la
import matplotlib.pyplot as plt
import numpy as np
import corner
import scipy.interpolate as interp
from scipy.optimize import fmin
import importlib.resources

import cupy as cp


class Search(object):

    def __init__(self):
        '''
        Typical usecase Scenario for Cobra:

        MySearch = Cobra.Search()
        MySearch.addDatFile('FileRoot', bary = True) #bary = True is the default
        MySearch.addCandidate('CandidateFile')
        MySearch.sample()
        '''

        self.SECDAY = 24*60*60
        self.pepoch = None
        self.length = None
        self.Sim = False

        self.Cand = None
        self.DatFiles = []

        self.CosOrbit = None
        self.SinOrbit = None
        self.TrueAnomaly = None
        self.CPUCosOrbit = None
        self.CPUSinOrbit = None
        self.CPUTrueAnomaly = None

        self.MinInterpEcc = 0
        self.MaxInterpEcc = 1
        self.InterpEccStepSize = 0.01
        self.NumInterpEccSteps = 100

        self.InterpBinarySteps = 10000
        self.doplot = False
        self.ChainRoot = None
        self.phys = None
        self.post = None
        self.ML = None

        self.MakeSignal = None
        self.AddAcceleration = None
        self.AddCircBinary = None
        self.Scatter = None
        self.subtractPhase = None
        self.GetPhaseBins = None
        self.RotatePhase = None
        self.MultNoise = None
        self.addInterpCircBinary = None
        self.addInterpEccBinary = None
        self.addInterpGRBinary = None

        self.AverageProf = None
        self.AverageBins = None


        kernel_code = importlib.resources.files("Cobra").joinpath("kernels.c").read_text()
        kernel_names = ["AddAcceleration", "AddInterpCircBinary", "AddInterpEccBinary", "AddInterpGRBinary","AddInterpCircBinary2","AddCircBinary","MakeSignal","GetPhaseBins","Scatter"]

        mod = cp.RawModule(
                code=kernel_code, name_expressions=kernel_names
                )

        self.MakeSignal = mod.get_function("MakeSignal")
        self.AddAcceleration = mod.get_function("AddAcceleration")
        self.AddCircBinary = mod.get_function("AddCircBinary")
        self.Scatter = mod.get_function("Scatter")
        self.GetPhaseBins = mod.get_function("GetPhaseBins")
        self.addInterpCircBinary = mod.get_function("AddInterpCircBinary")
        self.addInterpEccBinary = mod.get_function("AddInterpEccBinary")
        self.addInterpGRBinary = mod.get_function("AddInterpGRBinary")

        self.MultNoise = cp.ElementwiseKernel(
            "complex128 a, float64 b",
            "complex128 c",
            "c = a*b",
            "MultNoise")
        # self.MultNoise = mod.get_function("MultNoise")

    def addCandidate(self, filename):
        '''
        filename - name of the Candidate file.

        Candidate file has the following minimum content:

                Period  P fP

        Where P is the candidate period, and fP is the fractional error.
        
        Note that this is only for P. Any additional parameters need the 
        lower and upper limits of the prior as an input
 
        Optional lines are:

        Phase  ph  d_ph
        Width log10_w dlog10_w
        Acceleration  min_a max_a
        CircBinary min_log10_bp max_log10_bp min_log10_ba max_log10d_ba 
        Scattering min_log10_s max_log10_s
        DM min_dm max_dm  
        '''
        self.Cand = Candidate.Candidate(filename)

        if (self.Cand.FitCircBinary == True):
            self.CosOrbit = cp.empty(
                self.InterpBinarySteps+1, np.float64)
            self.SinOrbit = cp.empty(
                self.InterpBinarySteps+1, np.float64)

            self.CPUCosOrbit, self.CPUSinOrbit = self.KeplersOrbit(0)

            #self.CosOrbit = cp.asarray(np.float64(self.CPUCosOrbit))
            self.CosOrbit = cp.asarray(np.float64(self.CPUCosOrbit))
            self.SinOrbit = cp.asarray(np.float64(self.CPUSinOrbit))

        if (self.Cand.FitEccBinary == True):

            print(self.Cand.pmin[7], self.Cand.pmax[7])
            self.MinInterpEcc = self.Cand.pmin[7]
            self.MaxInterpEcc = self.Cand.pmax[7]
            self.InterpEccStepSize = 1
            self.NumInterpEccSteps = 1
            if (self.MaxInterpEcc - self.MinInterpEcc > 10.0**-10):
                self.NumInterpEccSteps = 100
                self.InterpEccStepSize = (
                    self.MaxInterpEcc - self.MinInterpEcc)/self.NumInterpEccSteps

            print("Interp details:", self.MinInterpEcc, self.MaxInterpEcc, 10.0**self.MinInterpEcc,
                  10.0**self.MaxInterpEcc, self.NumInterpEccSteps, self.InterpEccStepSize)

            self.CosOrbit = []
            self.SinOrbit = []

            self.CPUCosOrbit = []
            self.CPUSinOrbit = []

            for i in range(self.NumInterpEccSteps):

                Ecc = 10.0**(self.MinInterpEcc + i*self.InterpEccStepSize)
                print("Computing Ecc: ", i, self.MinInterpEcc +
                      i*self.InterpEccStepSize, Ecc)
                COrbit, SOrbit = self.KeplersOrbit(Ecc)

                self.CPUCosOrbit.append(COrbit)
                self.CPUSinOrbit.append(SOrbit)

                self.CosOrbit.append(cp.empty(
                    self.InterpBinarySteps+1, np.float64))
                self.SinOrbit.append(cp.empty(
                    self.InterpBinarySteps+1, np.float64))
                self.CosOrbit[i] = cp.asarray(
                    np.float64(self.CPUCosOrbit[i]))
                self.SinOrbit[i] = cp.asarray(
                    np.float64(self.CPUSinOrbit[i]))

        if (self.Cand.FitGRBinary == True or self.Cand.FitPKBinary == True):

            print(self.Cand.pmin[7], self.Cand.pmax[7])

            self.MinInterpEcc = self.Cand.pmin[7]
            self.MaxInterpEcc = self.Cand.pmax[7]

            self.NumInterpEccSteps = 1
            self.InterpEccStepSize = 1
            if (self.MaxInterpEcc - self.MinInterpEcc > 10.0**-10):
                self.NumInterpEccSteps = 100
                self.InterpEccStepSize = (
                    self.MaxInterpEcc - self.MinInterpEcc)/self.NumInterpEccSteps

            print("Interp details:", self.MinInterpEcc, self.MaxInterpEcc, 10.0**self.MinInterpEcc,
                  10.0**self.MaxInterpEcc, self.NumInterpEccSteps, self.InterpEccStepSize)

            self.CosOrbit = []
            self.SinOrbit = []
            self.TrueAnomaly = []

            self.CPUCosOrbit = []
            self.CPUSinOrbit = []
            self.CPUTrueAnomaly = []

            for i in range(self.NumInterpEccSteps):

                Ecc = 10.0**(self.MinInterpEcc + i*self.InterpEccStepSize)

                print("Computing Ecc: ", i, self.MinInterpEcc +
                      i*self.InterpEccStepSize, Ecc)
                COrbit, SOrbit = self.KeplersOrbit(Ecc)

                self.CPUCosOrbit.append(COrbit)
                self.CPUSinOrbit.append(SOrbit)

                self.CosOrbit.append(cp.empty(
                    self.InterpBinarySteps+1, np.float64))
                self.SinOrbit.append(cp.empty(
                    self.InterpBinarySteps+1, np.float64))

                self.CosOrbit[i] = cp.asarray(
                    np.float64(self.CPUCosOrbit[i]))
                self.SinOrbit[i] = cp.asarray(
                    np.float64(self.CPUSinOrbit[i]))

                # double sqr1me2 = sqrt(1-Ecc*Ecc);
                # double cume = CosBinarySignal-Ecc;
                # double onemecu = 1.0-Ecc*CosBinarySignal;

                # //double sae = sqr1me2*SinBinarySignal/onemecu;
                # //double cae = cume/onemecu;

                # double ae = TrueAnomaly; //atan2(sae, cae);
                # //ae = ae + 2*M_PI - trunc((ae+2*M_PI)/(2*M_PI))*2*M_PI;
                sae = np.sqrt(1.0 - Ecc*Ecc)*SOrbit/(1.0 - Ecc*COrbit)
                cae = (COrbit - Ecc)/(1.0 - Ecc*COrbit)
                self.CPUTrueAnomaly.append(np.arctan2(sae, cae) % (2*np.pi))

                self.TrueAnomaly.append(cp.empty(
                    self.InterpBinarySteps+1, np.float64))
                self.TrueAnomaly[i] = cp.asarray(
                    np.float64(self.CPUTrueAnomaly[i]))

    def addDatFile(self, root, bary=False, powerofTwo=True, setRefMJD=None, FromPickle=False, doFFT=True):
        '''
        Add dat file to the search with root 'root'.  Requires root.dat and root.inf to be present in directory
        bary - perform barycentering using Tempo2 to scale the time axis for the model (default = True)

        '''
        if (len(self.DatFiles) == 0):
            RefMJD = 0
            if (setRefMJD != None):
                print("setting ref:", setRefMJD)
                RefMJD = setRefMJD
            self.DatFiles.append(DatClass.DatFile(
                root, RefMJD, bary, powerofTwo, FromPickle, doFFT))
            self.pepoch = self.DatFiles[0].pepoch
            self.length = self.DatFiles[0].BaseTime[-1] - \
                self.DatFiles[0].BaseTime[0]
        else:
            RefMJD = self.DatFiles[0].RefMJD
            if (setRefMJD != None):
                print("setting ref:", setRefMJD)
                RefMJD = setRefMJD
            self.DatFiles.append(DatClass.DatFile(
                root, RefMJD, bary, powerofTwo, FromPickle, doFFT))
            self.pepoch = ((len(self.DatFiles) - 1)*self.pepoch +
                           self.DatFiles[-1].pepoch)/len(self.DatFiles)
            self.length = self.DatFiles[-1].BaseTime[-1] - \
                self.DatFiles[0].BaseTime[0]
            print('RefMJD:', self.DatFiles[0].RefMJD, self.DatFiles[-1].RefMJD)
            # self.pepoch = (self.DatFiles[-1].BaseTime[-1] - self.DatFiles[0].BaseTime[0])/2

    def gaussGPULike(self, x):

        like = 0
        uniformprior = 0

        phase = x[0]
        width = 10.0**x[1]  # Width
        period = x[2]

        pcount = 3
        if (self.Cand.FitBigP == True):
            BigP = np.floor(x[4])  # 3.21142857142857e-12
            pcount = pcount + 1
            period = period + BigP*2.248e-11

        # phase = phase + (0.5*self.length + self.DatFiles[0].BaseTime[0])/period

        if (self.Cand.FitAcceleration == True):

            Acceleration = x[3]
            # if(self.Cand.FitBigP == True):
            # Acceleration -= BigP*5.2e-18

            pcount = 4

            asum, alin = self.AccSum(Acceleration)

            # print asum, alin*self.pepoch, alin
            # asum=0
            # alin=0

            # phase += (asum - alin*self.pepoch)/period
            period += alin*period

            x[pcount] = phase % 1
            x[pcount+1] = period
            x[pcount+2] = Acceleration

        elif (self.Cand.FitCircBinary == True):

            BinaryAmp = 10.0**x[3]
            BinaryPhase = x[4]
            BinaryPeriod = (10.0**x[5])*24*60*60

            # BinaryPhase -= 2*np.pi*self.DatFiles[0].BaseTime[0]/BinaryPeriod
            BinaryPhase -= 2*np.pi * \
                (0.0*self.length + self.DatFiles[0].BaseTime[0])/BinaryPeriod
            BinaryPhase = BinaryPhase % (2*np.pi)

            bsum, blin, bstd = self.CircSum(
                self.CPUSinOrbit, BinaryPeriod, BinaryPhase, interpstep=1024)

            # bsum=0
            # blin=0
            # bstd=1

            BinaryAmp = BinaryAmp/bstd

            phase += -BinaryAmp*bsum/period + BinaryAmp*blin*self.pepoch/period

            BinaryPhase = BinaryPhase/(2*np.pi)

            # period += BinaryAmp*blin*period
            # print bsum, blin, bstd
            x[6] = phase % 1
            x[7] = period+BinaryAmp*blin*period
            x[8] = BinaryAmp
            x[9] = BinaryPhase % 1
            x[10] = BinaryPeriod/24/60/60

        elif (self.Cand.FitEccBinary == True):

            BinaryAmp = 10.0**x[3]
            BinaryPhase = x[4]
            BinaryPeriod = (10.0**x[5])*24*60*60
            Omega = x[6]
            LogEcc = x[7]
            Ecc = 10.0**LogEcc

            EccBin = np.int(
                np.floor((LogEcc-self.MinInterpEcc)/self.InterpEccStepSize))
            if (EccBin < 0):
                EccBin = 0
            if (EccBin >= self.NumInterpEccSteps):
                EccBin = self.NumInterpEccSteps-1
            # BinaryPhase -= 2*np.pi*self.DatFiles[0].BaseTime[0]/BinaryPeriod
            # BinaryPhase = BinaryPhase%(2*np.pi)

            uniformprior += np.log(Ecc)

            bsum, blin, bstd = self.EccSum(np.sin(Omega)*self.CPUCosOrbit[EccBin]+np.cos(
                Omega)*self.CPUSinOrbit[EccBin], BinaryPeriod, BinaryPhase, interpstep=128)

            # bsum, blin, bstd = self.CircSum(self.CPUCosOrbit, BinaryPeriod, BinaryPhase, interpstep=1024)

            BinaryAmp = BinaryAmp/bstd

            phase += -BinaryAmp*bsum/period + BinaryAmp*blin*self.pepoch/period

            BinaryPhase = BinaryPhase/(2*np.pi)

            x[8] = phase % 1
            x[9] = period+BinaryAmp*blin*period
            x[10] = BinaryAmp
            x[11] = BinaryPhase % 1
            x[12] = BinaryPeriod/24/60/60
            x[13] = Omega
            x[14] = Ecc

        elif (self.Cand.FitGRBinary == True):

            BinaryAmp = 10.0**x[3]
            BinaryPhase = x[4]
            BinaryPeriod = (10.0**x[5])*24*60*60
            Omega = x[6]

            LogEcc = x[7]
            Ecc = 10.0**LogEcc

            EccBin = np.int(
                np.floor((LogEcc-self.MinInterpEcc)/self.InterpEccStepSize))
            if (EccBin < 0):
                EccBin = 0
            if (EccBin >= self.NumInterpEccSteps):
                EccBin = self.NumInterpEccSteps-1

            # print "Check Bin:", LogEcc, self.MinInterpEcc, self.InterpEccStepSize, EccBin
            M1 = 10.0**x[8]
            M2 = 10.0**x[9]

            arr, ar, OMDot, SINI, Gamma, PBDot, DTheta, Dr = self.mass2dd(
                M1+M2, M2, BinaryAmp, Ecc, BinaryPeriod)

            # Check minimum brace
            args = (SINI, Omega, Ecc,)
            StartPoint = [0]
            MinBraceU = fmin(self.BraceFunc, StartPoint, args=(
                SINI, Omega, Ecc,), xtol=1e-8, disp=False)[0]
            MinBrace = self.BraceFunc(MinBraceU, *args)
            # print "min brace: ", MinBrace
            if (MinBrace < 1e-8):
                return -np.inf, x

            bsum = 0
            blin = 0
            bstd = 1

            BinaryAmp = BinaryAmp/bstd
            phase += -BinaryAmp*bsum/period + BinaryAmp*blin*self.pepoch/period
            BinaryPhase = BinaryPhase/(2*np.pi)

            x[10] = phase % 1
            x[11] = period+BinaryAmp*blin*period
            x[12] = BinaryAmp
            x[13] = BinaryPhase % 1
            x[14] = BinaryPeriod/24/60/60
            x[15] = Omega
            x[16] = Ecc
            x[17] = M1
            x[18] = M2
            x[19] = OMDot*(180.0/np.pi)*365.25*86400.0*2.0*np.pi/BinaryPeriod
            x[20] = SINI
            x[21] = Gamma
            x[22] = PBDot
            x[23] = DTheta
            x[24] = Dr

        elif (self.Cand.FitPKBinary == True):

            BinaryAmp = 10.0**x[3]
            BinaryPhase = x[4]
            BinaryPeriod = (10.0**x[5])*24*60*60
            Omega = x[6]

            LogEcc = x[7]
            Ecc = 10.0**LogEcc

            EccBin = np.int(
                np.floor((LogEcc-self.MinInterpEcc)/self.InterpEccStepSize))
            if (EccBin < 0):
                EccBin = 0
            if (EccBin >= self.NumInterpEccSteps):
                EccBin = self.NumInterpEccSteps-1

            # print "Check Bin:", LogEcc, self.MinInterpEcc, self.InterpEccStepSize, EccBin

            arr = np.float64(0.0)
            ar = np.float64(0.0)
            OMDot = x[8]/((180.0/np.pi)*365.25*86400.0*2.0*np.pi/BinaryPeriod)
            SINI = x[9]
            Gamma = x[10]
            PBDot = x[11]
            M2 = x[12]

            DTheta = np.float64(0.0)
            Dr = np.float64(0.0)

            bsum = 0
            blin = 0
            bstd = 1

            BinaryAmp = BinaryAmp/bstd
            phase += -BinaryAmp*bsum/period + BinaryAmp*blin*self.pepoch/period
            BinaryPhase = BinaryPhase/(2*np.pi)

            # x[12] = phase%1
            # x[13] = period+BinaryAmp*blin*period
            # x[14] = BinaryAmp
            # x[15] = BinaryPhase%1
            # x[16] = BinaryPeriod/24/60/60
            # x[17] = Omega
            # x[18] = Ecc
            # x[19] = M1
            # x[20] = M2
            # x[21] = OMDot*(180.0/np.pi)*365.25*86400.0*2.0*np.pi/BinaryPeriod
            # x[22] = SINI
            # x[23] = Gamma
            # x[24] = PBDot
            # x[25] = DTheta
            # x[26] = Dr

        else:

            x[3] = phase % 1
            x[4] = period

        for i in range(len(self.DatFiles)):

            if (self.Cand.FitEccBinary == True):

                CosOmega = np.float64(np.cos(Omega))
                SinOmega = np.float64(np.sin(Omega))

                Alpha = np.float64(BinaryAmp*SinOmega)
                Beta = np.float64(BinaryAmp*np.sqrt(1 - Ecc*Ecc)*CosOmega)

              
                self.addInterpEccBinary((self.DatFiles[i].Tblocks, 1), (self.DatFiles[i].block_size, 1, 1),
                        (self.DatFiles[i].gpu_pulsar_signal,  self.DatFiles[i].gpu_time, self.CosOrbit[EccBin], self.SinOrbit[EccBin], BinaryPeriod, BinaryPhase, BinaryAmp,
                                        CosOmega, SinOmega, Ecc, phase,  period, width**2, BinaryAmp*blin, Alpha, Beta))

            elif (self.Cand.FitGRBinary == True or self.Cand.FitPKBinary == True):

                SUNMASS = 4.925490947e-6
                M2 = M2*SUNMASS
                Ecc_r = Ecc*(1 + Dr)
                Ecc_th = Ecc*(1 + DTheta)
                SqEcc_th = np.sqrt(1.0-Ecc_th*Ecc_th)

                # print "GR parameters: ", OMDot, SINI, Gamma, PBDot, DTheta, Dr

                self.addInterpGRBinary((self.DatFiles[i].Tblocks, 1), (self.DatFiles[i].block_size, 1, 1),
                        (self.DatFiles[i].gpu_pulsar_signal,  self.DatFiles[i].gpu_time, self.CosOrbit[EccBin], self.SinOrbit[EccBin], self.TrueAnomaly[EccBin], BinaryPeriod, BinaryPhase, BinaryAmp, Omega, Ecc, M2, OMDot, SINI, Gamma, PBDot, SqEcc_th, Ecc_r, arr, ar, phase,  period, width**2, BinaryAmp*blin, self.pepoch))
                        

            elif (self.Cand.FitCircBinary == True):

                eta = np.float64(2*np.pi/BinaryPeriod)
                Beta = np.float64(eta*BinaryAmp)
                H2Beta = np.float64(0.5*Beta*Beta)

                self.addInterpCircBinary((self.DatFiles[i].Tblocks, 1), (self.DatFiles[i].block_size, 1, 1),
                        (self.DatFiles[i].gpu_pulsar_signal,  self.DatFiles[i].gpu_time, self.CosOrbit, self.SinOrbit, BinaryPeriod, BinaryPhase, BinaryAmp, phase,  period, width**2, BinaryAmp*blin,  eta, Beta, H2Beta))
                                         




            elif (self.Cand.FitAcceleration == True):

                self.AddAcceleration((self.DatFiles[i].Tblocks, 1), (self.DatFiles[i].block_size, 1, 1),
                        (self.DatFiles[i].gpu_pulsar_signal,  self.DatFiles[i].gpu_time, Acceleration, period,
                                     phase, width**2))  

            else:
                self.MakeSignal((self.DatFiles[i].Tblocks, 1), (self.DatFiles[i].block_size, 1, 1), (self.DatFiles[i].gpu_pulsar_signal, 
                                                                                                     self.DatFiles[i].gpu_time, 
                                                                                                     period, 
                                                                                                     width**2, 
                                                                                                     phase))

            self.DatFiles[i].gpu_pulsar_fft = cp.fft.rfft(self.DatFiles[i].gpu_pulsar_signal)

            if (self.Cand.FitScatter == True):
                ChanScale = (
                    (self.DatFiles[i].LowChan*10.0**6)**4)/(10.0**(9.0*4.0))

                tau = (10.0**x[3])/ChanScale
                self.Scatter((self.DatFiles[i].Fblocks, 1), (self.block_size, 1, 1), (rsig, isig, tau, self.DatFiles[i].SampleFreqs)) 
            output_array = cp.empty_like(self.DatFiles[i].gpu_pulsar_fft[1:-1])    
            self.MultNoise(
                self.DatFiles[i].gpu_pulsar_fft[1:-1], self.DatFiles[i].Noise, output_array)
            self.DatFiles[i].gpu_pulsar_fft[1:-1] = output_array


            # Compute the conjugate dot product for mcdot
            mcdot = cp.dot(self.DatFiles[i].gpu_pulsar_fft[1:-1].conj(), self.DatFiles[i].gpu_pulsar_fft[1:-1]).real
            #print(mcdot)

            # Calculate the norm
            norm = cp.sqrt(mcdot / 2 / self.DatFiles[i].FSamps)

            # Compute the conjugate dot product for cdot
            cdot = cp.dot(self.DatFiles[i].gpu_fft_data.conj(), self.DatFiles[i].gpu_pulsar_fft[1:-1]).real
            #print(cdot)
            MLAmp = cdot/mcdot
            MarginLike = MLAmp*cdot
            logdetMNM = np.log(mcdot) - 2*np.log(norm)

            like += -0.5*(logdetMNM - MarginLike)

            if (self.doplot == True):

                ZeroMLike = cdot*cdot/(mcdot + 10.0**20)
                ZerologdetMNM = np.log(mcdot + 10.0**20)
                ZerodetP = np.log(10.0**-20)
                ZeroLike = -0.5*(ZerologdetMNM - ZeroMLike + ZerodetP)
                '''	
				fig, (ax1, ax2) = plt.subplots(2,1)

				ax1.plot(np.arange(1,len(rsig)+1), self.DatFiles[i].Real.get(), color='black')
				ax1.plot(np.arange(1,len(rsig)+1), MLAmp*rsig.get(), color='red', alpha=0.6)

				ax2.plot(np.arange(1,len(rsig)+1), self.DatFiles[i].Imag.get(), color='black')
				ax2.plot(np.arange(1,len(rsig)+1), MLAmp*isig.get(), color='red', alpha=0.6)
				fig.show()
				'''
                # np.savetxt(self.ChainRoot+"Real_"+str(i)+".dat", zip(self.DatFiles[i].SampleFreqs.get(), self.DatFiles[i].Real.get(), (MLAmp*self.DatFiles[i].gpu_pulsar_fft.get()).real[1:-1]))
                # np.savetxt(self.ChainRoot+"Imag_"+str(i)+".dat", zip(self.DatFiles[i].SampleFreqs.get(), self.DatFiles[i].Imag.get(), (MLAmp*self.DatFiles[i].gpu_pulsar_fft.get()).imag[1:-1]))

                # self.DatFiles[i].gpu_pulsar_signal = self.DatFiles[i].gpu_time - phase*period
            # self.GetPhaseBins(self.DatFiles[i].gpu_pulsar_signal, period, grid=(self.DatFiles[i].Tblocks,1), block=(self.DatFiles[i].block_size,1,1))

                # phasebins=self.DatFiles[i].gpu_pulsar_signal.get()
                # floorbins=np.floor((phasebins+0.5)*self.AverageBins)
                weight = MLAmp/np.sqrt(1.0/mcdot)
                print("weight", MLAmp, np.sqrt(1.0/mcdot), weight, ZeroLike)
                '''
				if(self.AverageProf == None):
					self.AverageProf = np.zeros(self.AverageBins)
					OneProf = np.zeros(self.AverageBins)
					for bin in range(len(OneProf)):
						OneProf[bin] = np.sum(self.DatFiles[i].Data[floorbins==bin])/np.sum(floorbins==bin)
						#print i, bin, OneProf[bin], np.sum(floorbins==bin)
					self.AverageProf += weight*weight*OneProf/np.abs(np.std(OneProf))
					np.savetxt(self.ChainRoot+"AverageProf_"+str(i)+".dat", self.AverageProf/np.abs(np.std(self.AverageProf)))
					np.savetxt(self.ChainRoot+"OneProf_"+str(i)+".dat", OneProf/np.abs(np.std(OneProf)))
					
				else:
					OneProf = np.zeros(self.AverageBins)
					for bin in range(len(OneProf)):
						OneProf[bin] = np.sum(self.DatFiles[i].Data[floorbins==bin])
						#print i, bin, OneProf[bin]
					self.AverageProf += weight*weight*OneProf/np.abs(np.std(OneProf))
					np.savetxt(self.ChainRoot+"AverageProf_"+str(i)+".dat", self.AverageProf/np.abs(np.std(self.AverageProf)))
                                        np.savetxt(self.ChainRoot+"OneProf_"+str(i)+".dat", OneProf/np.abs(np.std(OneProf)))
				#plt.plot(np.linspace(0,1,len(self.AverageProf)), self.AverageProf)
				#plt.show()
				'''
        # like += uniformprior
        return like, x

    def Simulate(self, period, width):

        period = np.float64(period)
        width = np.float64(width)

        for i in range(len(self.DatFiles)):
            self.DatFiles[i].gpu_pulsar_signal = self.DatFiles[i].gpu_time - 0*period
            #self.MakeSignal(self.DatFiles[i].gpu_pulsar_signal, period, ((
            #    period*width)**2), grid=(self.DatFiles[i].Tblocks, 1), block=(self.DatFiles[i].block_size, 1, 1))
            self.MakeSignal(
                    (self.DatFiles[i].Tblocks, 1), (self.DatFiles[i].block_size, 1, 1),
                    (self.DatFiles[i].gpu_pulsar_signal, period, ((period*width)**2)) 
                    )

            s = self.DatFiles[i].gpu_pulsar_signal.get()
            np.savetxt("realsig.dat", list(
                zip(np.arange(0, 10000), s[:10000])))


            self.DatFiles[i].gpu_pulsar_fft = cp.fft.rfft(self.DatFiles[i].gpu_pulsar_signal)
            
            ranPhases = np.random.uniform(
                0, 1, len(self.DatFiles[i].gpu_pulsar_fft))
            CompRan = np.cos(2*np.pi*ranPhases) + 1j*np.sin(2*np.pi*ranPhases)
            CompRan[0] = 1 + 0j
            OComp = self.DatFiles[i].gpu_pulsar_fft.get()
            NComp = OComp*CompRan
            s = np.fft.irfft(NComp)
            np.savetxt("ransig.dat", list(zip(np.arange(0, 10000), s[:10000])))

    def MNprior(self, cube, ndim, nparams):
        for i in range(ndim):
            cube[i] = (self.Cand.pmax[i] - self.Cand.pmin[i]) * \
                cube[i] + self.Cand.pmin[i]

    def GaussGPULikeWrap(self, cube, ndim, nparams):

        x = np.zeros(nparams)
        for i in range(ndim):
            x[i] = cube[i]
        
        start = time.time()
        like, dp = self.gaussGPULike(x)
        end = time.time()
       # print("Likelihood evaluation time: {} s".format(end - start))

        for i in range(ndim, nparams):
            cube[i] = dp[i]

        return like

    def loadChains(self):
        self.phys = np.loadtxt(self.ChainRoot+'phys_live.points')
        self.post = np.loadtxt(self.ChainRoot+'post_equal_weights.dat')
        self.ML = self.phys[np.argmax(self.phys.T[-2])][:self.Cand.n_params]

    def plotResult(self, AverageBins=128):
        self.AverageBins = AverageBins
        self.doplot = True
        self.gaussGPULike(self.ML)
        self.doplot = False
        print(self.Cand.params)
        print(self.post.shape)
        #figure = corner.corner((self.post.T[:self.Cand.n_params]).T, labels=self.Cand.params,quantiles=[0.16, 0.5, 0.84],
				       #show_titles=True, title_kwargs={"fontsize": 12})
        #figure.savefig("test.png")

        '''
		figure = corner.corner((self.post.T[:self.Cand.n_params]).T, labels=self.Cand.params,
				       quantiles=[0.16, 0.5, 0.84],
				       show_titles=True, title_kwargs={"fontsize": 12})
		figure.show()
		'''

    def sample(self, nlive=500, ceff=False, efr=0.2, resume=False, doplot=False, sample=True):
        '''
        Function to begin sampling with model defined in Candidate File.

        nlive - number of live points for multinest (default = 500)
        ceff - use constant efficient mode (default = False)
        efr - efficiency rate (default = 0.2)
        resume - resume sampling if restarted (default = False)
        doplot - make plots after sampling (default = False)
        '''

        if (sample == True):
            pymultinest.run(self.GaussGPULikeWrap, self.MNprior, self.Cand.n_dims, n_params=self.Cand.n_params, importance_nested_sampling=False, resume=resume, verbose=True,
                            sampling_efficiency=efr, multimodal=False, const_efficiency_mode=ceff, n_live_points=nlive, init_MPI=True, outputfiles_basename=self.ChainRoot, wrapped_params=self.Cand.wrapped)

        self.loadChains()
        if (doplot == True):
            self.plotResult()

    def AccSum(self, Acceleration):

        mean = 0
        totsamps = 0
        for i in range(len(self.DatFiles)):
            min = self.DatFiles[i].BaseTime[0]
            max = self.DatFiles[i].BaseTime[-1]
            mean += (Acceleration/3.0)*(max**3 - min**3)
            totsamps += (max - min)

        bsum = mean/totsamps

        lin = 0
        min = self.DatFiles[0].BaseTime[0]
        max = self.DatFiles[-1].BaseTime[-1]
        lin = Acceleration*(max**2 - min**2)/(max-min)
        # print (max**2 - min**2)/(max-min), min, (max-min)
        return bsum, lin

    def CircSum(self, Orbit, BinaryPeriod, BinaryPhase, interpstep):

        if (self.Sim == True):
            return 0.0, 0.0, 1.0

        mean = 0
        totsamps = 0
        for i in range(len(self.DatFiles)):
            min = self.DatFiles[i].BaseTime[0]
            max = self.DatFiles[i].BaseTime[-1]
            mean += ((BinaryPeriod*(np.cos(BinaryPhase + (2*min*np.pi)/BinaryPeriod) -
                     np.cos(BinaryPhase + (2*max*np.pi)/BinaryPeriod)))/(2*np.pi))
            totsamps += max - min

        bsum = mean/totsamps

        lin = 0
        min = self.DatFiles[0].BaseTime[0]
        max = self.DatFiles[-1].BaseTime[-1]
        lin = (np.sin(BinaryPhase + (2*max*np.pi)/BinaryPeriod) -
               np.sin(BinaryPhase + (2*min*np.pi)/BinaryPeriod))/(max-min)

        totsamps = 0
        bstd = 0
        for i in range(len(self.DatFiles)):

            NInterp = len(self.DatFiles[i].BaseTime[::interpstep])
            ITime = np.zeros(NInterp+1)
            ITime[:NInterp] = self.DatFiles[i].BaseTime[::interpstep]
            ITime[-1] = self.DatFiles[i].BaseTime[-1]
            Sig = interp.griddata(np.linspace(0, BinaryPeriod, self.InterpBinarySteps+1),
                                  Orbit,  (ITime+BinaryPeriod*BinaryPhase/2.0/np.pi) % BinaryPeriod)
            totsamps += NInterp+1

            Sig = Sig - bsum - lin*(ITime-self.pepoch)

            bstd += np.dot(Sig, Sig)

        bstd = np.sqrt(bstd/totsamps)

        return bsum, lin, bstd

    def EccSum(self, Orbit, Period, BinaryPhase, interpstep):

        blinMin = 0
        blinMax = 0
        bstd = 0
        bsum = 0
        blin = 0
        totsamps = 0

        Sigs = []
        Times = []

        for i in range(len(self.DatFiles)):
            NInterp = len(self.DatFiles[i].BaseTime[::interpstep])
            ITime = np.zeros(NInterp+1)
            ITime[:NInterp] = self.DatFiles[i].BaseTime[::interpstep]
            ITime[-1] = self.DatFiles[i].BaseTime[-1]
            Sig = interp.griddata(np.linspace(0, Period, self.InterpBinarySteps+1),
                                  Orbit,  (ITime+Period*BinaryPhase/2.0/np.pi) % Period)

            totsamps += NInterp+1
            bsum += np.sum(Sig)

            if (i == 0):
                mintime = self.DatFiles[i].BaseTime[0]
                blinMin = Sig[0]
            if (i == len(self.DatFiles)-1):
                maxtime = self.DatFiles[i].BaseTime[-1]
                blinMax = Sig[-1]

            Sigs.append(Sig)
            Times.append(ITime)

        bsum = bsum/totsamps
        blin = (blinMax-blinMin)/(maxtime-mintime)

        for i in range(len(self.DatFiles)):
            Sigs[i] = Sigs[i] - bsum - blin*(Times[i]-self.pepoch)
            bstd += np.dot(Sigs[i], Sigs[i])

        bstd = np.sqrt(bstd/totsamps)

        return bsum, blin, bstd

    def KeplersOrbit(self, ecc):

        time = np.linspace(0, 1, self.InterpBinarySteps+1)

        MeanAnomoly = 2*np.pi*time

        if (ecc == 0):
            return np.cos(MeanAnomoly), np.sin(MeanAnomoly)

        EccAnomoly = MeanAnomoly+ecc * \
            np.sin(MeanAnomoly)/np.sqrt(1.0-2*ecc*np.cos(MeanAnomoly)+ecc*ecc)
        for i in range(5):
            dE = (EccAnomoly - ecc*np.sin(EccAnomoly) -
                  MeanAnomoly)/(1.0-ecc*np.cos(EccAnomoly))
            EccAnomoly -= dE

        CosEccAnomoly = np.cos(EccAnomoly)
        SinEccAnomoly = np.sin(EccAnomoly)

        return CosEccAnomoly, SinEccAnomoly

        CosTrueAnomoly = (CosEccAnomoly - ecc)/(1-ecc*CosEccAnomoly)
        SinTrueAnomoly = np.sqrt(1.0 - ecc*ecc) * \
            SinEccAnomoly/(1-ecc*CosEccAnomoly)

        TrueAnomoly = np.arctan2(SinTrueAnomoly, CosTrueAnomoly)

        return np.cos(TrueAnomoly), np.sin(TrueAnomoly)

    def mass2dd(self, TotalMass, CompMass, BinaryAmp, BinaryEcc, BinaryPeriod):

        SUNMASS = 4.925490947e-6
        ARRTOL = 1.0e-10

        an = 2*np.pi/BinaryPeriod

        arr = 0
        ar = 0
        OMDot = 0
        SINI = 0
        Gamma = 0
        PBDot = 0
        DTheta = 0
        Dr = 0

        m = TotalMass*SUNMASS
        m2 = CompMass*SUNMASS
        m1 = m-m2

        arr0 = (m/(an*an))**(1.0/3.0)
        arr = arr0
        arrold = 0

        while (np.abs((arr-arrold)/arr) > ARRTOL):
            arrold = arr
            arr = arr0*(1.0+(m1*m2/m/m - 9.0)*0.5*m/arr)**(2.0/3.0)
            # print arr

        ar = arr*m2/m

        SINI = BinaryAmp/ar
        OMDot = 3.0*m/(arr*(1.0-BinaryEcc*BinaryEcc))
        Gamma = BinaryEcc*m2*(m1+2*m2)/(an*arr*m)
        PBDot = -(96.0*2.0*np.pi/5.0)*an**(5.0/3.0)*(1.0-BinaryEcc*BinaryEcc)**(-3.5) * \
            (1+(73.0/24)*BinaryEcc*BinaryEcc + (37.0/96)
             * BinaryEcc**4)*m1*m2*m**(-1.0/3.0)

        Dr = (3.0*m1*m1 + 6.0*m1*m2 + 2.0*m2*m2)/(arr*m)
        DTheta = (3.5*m1*m1 + 6*m1*m2 + 2*m2*m2)/(arr*m)

        return arr, ar, OMDot, SINI, Gamma, PBDot, DTheta, Dr

    def Circmass2dd(self, TotalMass, CompMass, BinaryAmp, BinaryPeriod):

        SUNMASS = 4.925490947e-6
        ARRTOL = 1.0e-10

        an = 2*np.pi/BinaryPeriod

        arr = 0
        ar = 0
        OMDot = 0
        SINI = 0
        Gamma = 0
        PBDot = 0
        DTheta = 0
        Dr = 0

        m = TotalMass*SUNMASS
        m2 = CompMass*SUNMASS
        m1 = m-m2

        arr0 = (m/(an*an))**(1.0/3.0)
        arr = arr0
        arrold = 0

        while (np.abs((arr-arrold)/arr) > ARRTOL):
            arrold = arr
            arr = arr0*(1.0+(m1*m2/m/m - 9.0)*0.5*m/arr)**(2.0/3.0)

        ar = arr*m2/m

        SINI = BinaryAmp/ar
        OMDot = 3.0*m/arr
        PBDot = -(96.0*2.0*np.pi/5.0)*an**(5.0/3.0)*m1*m2*m**(-1.0/3.0)

        Dr = (3.0*m1*m1 + 6.0*m1*m2 + 2.0*m2*m2)/(arr*m)
        DTheta = (3.5*m1*m1 + 6*m1*m2 + 2*m2*m2)/(arr*m)

        return arr, ar, OMDot, SINI, Gamma, PBDot, DTheta, Dr

    def BraceFunc(self, u, *args):

        SINI = args[0]
        Omega = args[1]
        BinaryEcc = args[2]

        sqr1me2 = np.sqrt(1-BinaryEcc*BinaryEcc)
        cume = np.cos(u)-BinaryEcc
        onemecu = 1.0-BinaryEcc*np.cos(u)
        brace = onemecu-SINI*(np.sin(Omega)*cume +
                              sqr1me2*np.cos(Omega)*np.sin(u))
        # print u, brace
        return brace

    def CircBraceFunc(self, u, *args):

        SINI = args[0]

        brace = 1 - SINI  # *np.sin(u)

        return brace
