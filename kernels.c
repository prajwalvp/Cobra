#include <cupy/complex.cuh>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


		    extern "C" __global__ void AddAcceleration(double *a, double *orig, double accel, double period, double phase, double width)
		  {
			const int i = blockDim.x*blockIdx.x + threadIdx.x;

			a[i] = orig[i] + accel*orig[i]*orig[i];


                        a[i] = a[i]/period - phase - trunc(a[i] / period - phase);
                        a[i] = (a[i]+1) - trunc(a[i]+1);
                        a[i] = a[i]-0.5;
                        a[i] = exp(-0.5*a[i]*a[i]/width);
		       
		     
		   }



		    extern "C" __global__ void AddInterpCircBinary(double *a, double *orig, double *InterpCosBinary, double *InterpSinBinary, double BinaryPeriod, double BinaryPhase, double BinaryAmp, double phase, double period, double width, double blin, double eta, double etaB, double Heta2B)
		  {
			const int i = blockDim.x*blockIdx.x + threadIdx.x;
			

			double BPhase = orig[i]/BinaryPeriod + BinaryPhase;
			BPhase = BPhase - trunc(BPhase);
                        BPhase = 10000*((BPhase + 1) - trunc((BPhase+1)));

			int LowBin = floor(BPhase);
			int HighBin = LowBin+1;
			double BinaryCosSignal = InterpCosBinary[LowBin]+(InterpCosBinary[HighBin] - InterpCosBinary[LowBin])*(BPhase-LowBin);
			double BinarySinSignal = InterpSinBinary[LowBin]+(InterpSinBinary[HighBin] - InterpSinBinary[LowBin])*(BPhase-LowBin);

			double BinarySignal =  BinaryAmp*BinarySinSignal*(1 - etaB*BinaryCosSignal + Heta2B*BinarySinSignal*BinarySinSignal);
			
			a[i] = orig[i] - BinarySignal + blin*orig[i];

                        a[i] = a[i]/period - phase - trunc(a[i] / period - phase);
                        a[i] = a[i] + 0.5 - trunc(a[i]+1);
                        a[i] = exp(-0.5*a[i]*a[i]/width);
		       
		   }

		    extern "C" __global__ void AddInterpEccBinary(double *a, double *orig, double *InterpCosBinary, double *InterpSinBinary, double BinaryPeriod, double BinaryPhase, double BinaryAmp, double BinaryCosW, double BinarySinW, double Ecc, double phase, double period, double width, double blin, double Alpha, double Beta)
		  {
			const int i = blockDim.x*blockIdx.x + threadIdx.x;
			
			double BPhase = orig[i]/BinaryPeriod + BinaryPhase;
			BPhase = BPhase - trunc(BPhase);
                        BPhase = 10000*((BPhase + 1) - trunc((BPhase+1)));

			int LowBin = floor(BPhase);
			int HighBin = LowBin+1;
			double CosBinarySignal = InterpCosBinary[LowBin]+(InterpCosBinary[HighBin] - InterpCosBinary[LowBin])*(BPhase-LowBin);
			double SinBinarySignal = InterpSinBinary[LowBin]+(InterpSinBinary[HighBin] - InterpSinBinary[LowBin])*(BPhase-LowBin);


			double eta = 2*M_PI/BinaryPeriod/(1-Ecc*CosBinarySignal);

			double Dre = Alpha*(CosBinarySignal - Ecc) + Beta*SinBinarySignal;
			double Drep = -Alpha*SinBinarySignal + Beta*CosBinarySignal;
			double Drepp = -Alpha*CosBinarySignal - Beta*SinBinarySignal;

			double BinarySignal = Dre*(1-eta*Drep + eta*eta*(Drep*Drep + 0.5*Dre*Drepp - 0.5*Ecc*SinBinarySignal*Dre*Drep/(1-Ecc*CosBinarySignal)));
	
			a[i] = orig[i] - BinarySignal + blin*orig[i];

                        a[i] = a[i]/period - phase - trunc(a[i] / period - phase);
                        a[i] = a[i] + 0.5  - trunc(a[i]+1);
                        a[i] = exp(-0.5*a[i]*a[i]/width);
		       
		   }

		   extern "C" __global__ void AddInterpGRBinary(double *a, double *orig, double *InterpCosBinary, double *InterpSinBinary, double *InterpTrueAnomaly, double BinaryPeriod, double BinaryPhase, double BinaryAmp, double BinaryOmega, double Ecc, double M2, double OMDot, double SINI, double Gamma, double PBDot, double SqEcc_th, double Ecc_r, double arr, double ar, double phase,  double period, double width, double blin, double pepoch){



                        const int i = blockDim.x*blockIdx.x + threadIdx.x;
                        
                        //double BPhase = (orig[i]/BinaryPeriod)*(1.0 - 0.5*PBDot*(orig[i]/BinaryPeriod)) + BinaryPhase;
			double BPhase = (orig[i]/BinaryPeriod + BinaryPhase)*(1.0 - 0.5*PBDot*(orig[i]/BinaryPeriod + BinaryPhase));
			int norbits = trunc(BPhase);
                        BPhase = BPhase - norbits;
                        BPhase = 10000*((BPhase + 1) - trunc((BPhase+1)));

                        int LowBin = floor(BPhase);
                        int HighBin = LowBin+1;
                        double CosBinarySignal = InterpCosBinary[LowBin]+(InterpCosBinary[HighBin] - InterpCosBinary[LowBin])*(BPhase-LowBin);
                        double SinBinarySignal = InterpSinBinary[LowBin]+(InterpSinBinary[HighBin] - InterpSinBinary[LowBin])*(BPhase-LowBin);
			double TrueAnomaly = InterpTrueAnomaly[LowBin]+(InterpTrueAnomaly[HighBin] - InterpTrueAnomaly[LowBin])*(BPhase-LowBin);


				//double sqr1me2 = sqrt(1-Ecc*Ecc);
			double cume = CosBinarySignal-Ecc;
			double onemecu = 1.0-Ecc*CosBinarySignal;

				//double Ecc_r = Ecc*(1 + Dr);
				//double Ecc_th = Ecc*(1 + DTheta);

				//double sae = sqr1me2*SinBinarySignal/onemecu;
				//double cae = cume/onemecu;

			double ae = TrueAnomaly;
				//double ae = atan2(sae, cae);
				//ae = ae + 2*M_PI - trunc((ae+2*M_PI)/(2*M_PI))*2*M_PI;
			ae = 2.0*M_PI*norbits + ae;

			double omega = BinaryOmega + OMDot*ae;
			double SinOmega = sin(omega);
			double CosOmega = cos(omega);

			double alpha = BinaryAmp*SinOmega;
			double beta =  BinaryAmp*SqEcc_th*CosOmega;

			double bg = beta+Gamma;
			double dre = alpha*(CosBinarySignal-Ecc_r) + bg*SinBinarySignal;
			double drep = -alpha*SinBinarySignal + bg*CosBinarySignal;
			double drepp = -alpha*CosBinarySignal - bg*SinBinarySignal;
			double anhat=(2*M_PI/BinaryPeriod)/onemecu;

			double brace = onemecu-SINI*(SinOmega*cume+SqEcc_th*CosOmega*SinBinarySignal);

			double dlogbr = log(brace);
			double ds = -2*M2*dlogbr;

			double BinarySignal = dre*(1-anhat*drep+(anhat*anhat)*(drep*drep + 0.5*dre*drepp - 0.5*Ecc*SinBinarySignal*dre*drep/onemecu)) + ds;

			a[i] = orig[i] - BinarySignal + blin*orig[i];

			a[i] = a[i]/period - phase - trunc(a[i] / period - phase);
			a[i] = a[i] + 0.5 - trunc(a[i]+1);
			a[i] = exp(-0.5*a[i]*a[i]/width);


		}

		    extern "C" __global__ void AddInterpCircBinary2(double *a, double *orig, double *InterpCosBinary, double *InterpSinBinary, double BinaryPeriod, double BinaryCosAmp, double BinarySinAmp, double blin, double pepoch)
		  {
			const int i = blockDim.x*blockIdx.x + threadIdx.x;
			
			double BPhase = orig[i]/BinaryPeriod;
			BPhase = BPhase - trunc(BPhase);
                        BPhase = 10000*((BPhase + 1) - trunc((BPhase+1)));

			int LowBin = floor(BPhase);
			int HighBin = LowBin+1;
			double BinaryCosSignal = BinaryCosAmp*InterpCosBinary[LowBin]+(InterpCosBinary[HighBin] - InterpCosBinary[LowBin])*(BPhase-LowBin);
			double BinarySinSignal = BinarySinAmp*InterpSinBinary[LowBin]+(InterpSinBinary[HighBin] - InterpSinBinary[LowBin])*(BPhase-LowBin);

			a[i] = orig[i] + BinaryCosSignal + BinarySinSignal - blin*(orig[i]-pepoch);
		       
		   }

		    extern "C" __global__ void AddCircBinary(double *a, double *orig, double BinaryAmp, double BinaryPeriod, double BinaryPhase, double phase, double blin, double pepoch)
		  {
			const int i = blockDim.x*blockIdx.x + threadIdx.x;

			a[i] = orig[i] + BinaryAmp*cos(2*M_PI*orig[i]/BinaryPeriod + BinaryPhase) - phase - blin*(orig[i]-pepoch);
		       
		   }


		    extern "C" __global__ void MakeSignal(double *a, double *orig, double period, double width, double phase)
		  {
			const int i = blockDim.x*blockIdx.x + threadIdx.x;

			a[i] = orig[i]/period - phase - trunc(orig[i] / period - phase);
			a[i] = (a[i]+1) - trunc(a[i]+1);
			a[i] = a[i]-0.5;
			a[i] = exp(-0.5*a[i]*a[i]/width);

		     
		   }

		    extern "C" __global__ void GetPhaseBins(double *a, double period)
		  {
			const int i = blockDim.x*blockIdx.x + threadIdx.x;

			a[i] = ((a[i]) - period * trunc((a[i]) / period)) ;
			a[i] = ((a[i]+ period) - period * trunc((a[i]+period) / period)) ;
			a[i] = a[i] - period/2;
			a[i] = a[i]/period;
			
		     
		   }

		    extern "C" __global__ void Scatter(double *real, double *imag, double TimeScale, double *samplefreqs)
		  {
			const int i = blockDim.x*blockIdx.x + threadIdx.x;

			double RProf = real[i];
			double IProf = imag[i];


			double RConv = 1.0/(samplefreqs[i]*samplefreqs[i]*TimeScale*TimeScale+1);
			double IConv = -samplefreqs[i]*TimeScale/(samplefreqs[i]*samplefreqs[i]*TimeScale*TimeScale+1); //NB Timescale = Tau/(pow(((chanfreq, 4)/pow(10.0, 9.0*4.0));

			real[i] = RProf*RConv - IProf*IConv;
			imag[i] = RProf*IConv + IProf*RConv;
		       
		   }

