#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <project_main.h>

#define iterations 3600
#define BlockSizeX 26
#define BlockSizeY 26
#define stepsPerKernel 4
#define BlockFinalSizeX (BlockSizeX - 2*stepsPerKernel)
#define BlockFinalSizeY (BlockSizeY - 2*stepsPerKernel)

#define tPlot 100000

#define lx 1000
#define ly 1000

#define PSx 50
#define PSy 40
#define PSamplitude 0.1f
#define PSperiod 3.0f

#define obst_x 333
#define obst_y 300
#define obst_r 150

#define leftWall 300
#define rightWall 500
#define btmWall 0
#define topWall 599
#define slitWidth 30

#define uMax 0.1
#define Re 1000
#define nu uMax * 2 * obst_r / Re
#define omega 1.0f / (3.0f * nu + 0.5f)

#define in 0
#define out (lx-1)

#define tx threadIdx.x
#define ty threadIdx.y
#define bx blockIdx.x
#define by blockIdx.y
#define bdx blockDim.x
#define bdy blockDim.y
#define gdx gridDim.x
#define gdy gridDim.y

#define dir(K) x + y*lx + K*lx*ly
#define dirS(K) tx + ty*bdx + K*bdx*bdy

#define iceil(n,d) ((n-1)/d)+1

#define cudaCheckErrors(msg) \
   do { \
      cudaError_t __err = cudaGetLastError();\
      if (__err != cudaSuccess) { \
         printf("Fatal error %s (%s at %s:%d)\n", msg, cudaGetErrorString(__err), __FILE__, __LINE__); \
         exit(1); \
      } \
   } while (0)

__device__ float t[9]; //= {4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36};
__constant__ int cx[9];
__constant__ int cy[9];
__constant__ int opp[9];

__global__ void tiledKernel (float *fIn, float *fOut, char *bbRegion, int iteration) {
	const int x = bx * (BlockSizeX - 2*stepsPerKernel ) + tx - stepsPerKernel;
	const int y = by*BlockFinalSizeY + ty - stepsPerKernel;
	bool onGrid;
	onGrid = (0 <= x && x < lx && 0 <= y && y < ly);

	volatile __shared__ float fInS[9 * BlockSizeX * BlockSizeY];
	volatile __shared__ float fOutS[9 * BlockSizeX * BlockSizeY];
	#pragma unroll
	for (int k = 0; k < 9; ++k) {
		fInS[dirS(k)] = (onGrid) ? fIn[dir(k)] : 0.0f;
		fOutS[dirS(k)] = (onGrid) ? fOut[dir(k)] : 0.0f;
	}

	float temp, rho, ux, uy, vert, left, right;

	bool activeThread = true;
	
	#pragma unroll
	for (int step = 0; step < stepsPerKernel;) {
		rho = 0.0f;
		ux = 0.0f;
		uy = 0.0f;
		vert = 0.0f;
		left = 0.0f;
		right = 0.0f;

		if (onGrid && activeThread) {
			// MACROSCOPIC VARIABLES
			#pragma unroll
			for (int k = 0; k < 9; ++k) {
				temp = (onGrid /*x < lx && y < ly*/) ? fInS[dirS(k)] : 0.0f;
				rho += temp;
				ux += cx[k] * temp;
				uy += cy[k] * temp;
				if (k == 0 || k == 2 || k == 4) {
					vert += temp;
				}
				if (k == 3 || k == 6 || k == 7) {
					left += temp;
				}
				if (k == 1 || k == 5 || k == 8) {
					right += temp;
				}
			}
			ux /= rho;
			uy /= rho;
			vert /= rho;
			left /= rho;
			right /= rho;
		}
	
	
		// MACROSCOPIC (DIRICHLET) BOUNDARY CONDITIONS
			// Inlet: Poiuseville profile
		__syncthreads();
		if (x == in && 0 < y && y < ly-1) {
			ux = 4.0f * uMax / ((ly-2.0f)*(ly-2.0f)) * ((y-0.5f)*(ly-2.0f) - (y-0.5f)*(y-0.5f));
			uy = 0.0f;
			rho = 1.0f / (1.0f - ux) * ( vert + 2*left ) ;
		}
			// Outlet: constant pressure
	/*	__syncthreads();
		if (x == out && 0 < y && y < ly-1) { 
			rho = 1.0f;
			ux = -1.0f + 1.0f / rho * ( vert + 2*right ) ;
			uy = 0.0f;
		}
*/


		// MICROSCOPIC BOUNDARY CONDITIONS: 
			// Inlet: Zou/He B.C.
		__syncthreads();
		if (x == in && 0 < y && y < ly-1) {
			fInS[dirS(1)] = fInS[dirS(3)] + 2.0f/3.0f * rho * ux; 
			fInS[dirS(5)] = fInS[dirS(7)] + 0.5f*(fInS[dirS(4)]-fInS[dirS(2)]) + 0.5f*rho*uy + 1.0f/6.0f*rho*ux;
			fInS[dirS(8)] = fInS[dirS(6)] + 0.5f*(fInS[dirS(2)]-fInS[dirS(4)]) - 0.5f*rho*uy + 1.0f/6.0f*rho*ux;
		}
	 
			// Outlet: Zou/He BC
/*		__syncthreads();
		if (x == out && 0 < y && y < ly-1) {
			fInS[dirS(3)] = fInS[dirS(1)] - 2.0f/3.0f * rho * ux;
			fInS[dirS(7)] = fInS[dirS(5)] + 0.5f*(fInS[dirS(2)]-fInS[dirS(4)]) - 0.5f*rho*uy - 1.0f/6.0f*rho*ux;
			fInS[dirS(6)] = fInS[dirS(8)] + 0.5f*(fInS[dirS(4)]-fInS[dirS(2)]) + 0.5f*rho*uy - 1.0f/6.0f*rho*ux;
		}
*/

		// Collision
		__syncthreads();
		if (onGrid && activeThread) {
			float cu;
			bool isPS = false; // (x == PSx && y == PSy);
			if (isPS) {
				rho = 2.0f + PSamplitude * sin(2*M_PI*iteration/PSperiod);
			}
			#pragma unroll
			for (int k = 0; k < 9; ++k) {
				cu = 3.0f * (cx[k]*ux + cy[k]*uy);
				temp = rho * t[k] * ( 1.0f + cu + 0.5*(cu*cu) - 1.5f*(ux*ux + uy*uy));		
				if (isPS) {
					fOutS[dirS(k)] = temp;
				} else {
					fOutS[dirS(k)] = fInS[dirS(k)] - omega * (fInS[dirS(k)] - temp);
				}
			}
		}

		// Obstacle (bounce-back)
		__syncthreads();
		if (onGrid && activeThread) {
			if (bbRegion[x+y*lx] == 1) {
				#pragma unroll
				for (int k = 0; k < 9; ++k) {
					fOutS[dirS(k)] = fInS[dirS(opp[k])]; 
				}
			}
		}



		// increase the border of inactive threads. this is to prevent cross-block talk and allow iteration within a kernel.
		++step;
		activeThread = (tx<step || bdx-step<=tx || ty<step || bdy-step<=ty) ? false : activeThread;
		
		// Streaming
		__syncthreads();
		if (0 <= x && x < lx && 0 <= y && y < ly && onGrid && activeThread) {
			int xSource, ySource;
			#pragma unroll
			for (int k = 0; k < 9; ++k) {
//THIS IS WRONG ;;; MODULO BDX MAKES NO SENSE
				xSource = tx - cx[k];
				ySource = ty - cy[k];

//				xSource += bdx;
//				ySource += bdy;

//				xSource %= bdx;
//				ySource %= bdy;
		
/*		perhaps you should check to see if it wraps around, and if so, just ignore the load.
		this could prevent uncoalesced access and it shouldn't matter since boundary conditions 
		should be already handled by inlet/outlet/obstacles (unless you have a periodic simulation).
*/		
				fInS[dirS(k)] = fOutS[xSource + ySource*bdx + k*bdx*bdy];
			}
/*			if (x == 0) {
				ux = 4.0f * uMax / ((ly-2.0f)*(ly-2.0f)) * ((y-0.5f)*(ly-2.0f) - (y-0.5f)*(y-0.5f));
				uy = 0.0f;
				rho = 1.0f / (1.0f - ux) * ( vert + 2*left ) ;
				fInS[dirS(1)] = fInS[dirS(3)] + 2.0f/3.0f * rho * ux; 
				fInS[dirS(5)] = fInS[dirS(7)] + 0.5f*(fInS[dirS(4)] - fInS[dirS(2)]) + 0.5f*rho*uy + 1.0f/6.0f*rho*ux;
			        fInS[dirS(8)] = fInS[dirS(6)] + 0.5f*(fInS[dirS(2)] - fInS[dirS(4)]) - 0.5f*rho*uy + 1.0f/6.0f*rho*ux;
		

		fInS[dirS(1)] = fInS[dirS(3)]  ;
                                fInS[dirS(5)] = fInS[dirS(7)];
                                fInS[dirS(8)] = fInS[dirS(6)];
			}
*/			if (x == lx-1 && 0 < y && y <= ly-1) {
				rho = 1.0f;
				ux = -1.0f + 1.0f / rho * ( fInS[dirS(0)] + fInS[dirS(2)] + fInS[dirS(4)] + 2*(fInS[dirS(1)] + fInS[dirS(5)] + fInS[dirS(8)]) ) ;
				uy = 0.0f;

				fInS[dirS(3)] = fInS[dirS(1)] - 2.0f/3.0f * rho * ux;
				fInS[dirS(7)] = fInS[dirS(5)] + 0.5f*(fInS[dirS(2)]-fInS[dirS(4)]) - 0.5f*rho*uy - 1.0f/6.0f*rho*ux;
				fInS[dirS(6)] = fInS[dirS(8)] + 0.5f*(fInS[dirS(4)]-fInS[dirS(2)]) + 0.5f*rho*uy - 1.0f/6.0f*rho*ux;


/*				fInS[dirS(3)] = fInS[dirS(1)];
                                fInS[dirS(7)] = fInS[dirS(5)];
                                fInS[dirS(6)] = fInS[dirS(8)];
*/			}
		}
	}
	
	__syncthreads();
	if (activeThread && onGrid) {
		#pragma unroll
		for (int k = 0; k < 9; ++k) {
			fIn[dir(k)] = fInS[dirS(k)];
		}
//		fIn[dir(5)] = 2.0f;
	} else {
		if (onGrid) { 
			for (int k = 0; k < 9; ++k) {
//				fIn[dir(k)] = 0.0f;
			}
//			fIn[dir(5)] = 3.0f;
		}
	}
}



__global__ void cylinderKernelCollide (float *fIn, float *fOut, char *bbRegion, int iteration) {
	// Grid location
	const int x = bx*bdx + tx;
	const int y = by*bdy + ty;

	// MACROSCOPIC VARIABLES
	float temp;
	float rho = 0.0f;
	float ux = 0.0f;
	float uy = 0.0f;
	float vert = 0.0f; // these three are used for inlets/outlets
	float left = 0.0f;
	float right = 0.0f; 
	#pragma unroll
	for (int k = 0; k < 9; ++k) {
		temp = (x < lx && y < ly) ? fIn[dir(k)]  : 0.0f;
		rho += temp;
		ux += cx[k] * temp;
		uy += cy[k] * temp;
		if (k == 0 || k == 2 || k == 4) {
			vert += temp;
		}
		if (k == 3 || k == 6 || k == 7) {
			left += temp;
		}
		if (k == 1 || k == 5 || k == 8) {
			right += temp;
		}
	}
	ux /= rho;
	uy /= rho;
	vert /= rho;
	left /= rho;
	right /= rho;

	

	// MACROSCOPIC (DIRICHLET) BOUNDARY CONDITIONS
	    // Inlet: Poiuseville profile
	__syncthreads();
	if (x == in && 0 < y && y < ly-1) {
		ux = 4.0f * uMax / ((ly-2.0f)*(ly-2.0f)) * ((y-0.5f)*(ly-2.0f) - (y-0.5f)*(y-0.5f));
		uy = 0.0f;
		rho = 1.0f / (1.0f - ux) * ( vert + 2*left ) ;
	}
	    // Outlet: constant pressure
	__syncthreads();
	if (x == out && 0 < y && y < ly-1) { 
		rho = 1.0f;
		ux = -1.0f + 1.0f / rho * ( vert + 2*right ) ;
		uy = 0.0f;
	}



	// MICROSCOPIC BOUNDARY CONDITIONS: 
	    // Inlet: Zou/He B.C.
	__syncthreads();
	if (x == in && 0 < y && y < ly-1) {
		fIn[x + y*lx + 1*lx*ly] = fIn[x+y*lx+3*lx*ly] + 2.0f/3.0f * rho * ux; 
		fIn[x+y*lx+5*lx*ly] = fIn[x+y*lx+7*lx*ly] + 0.5f*(fIn[x+y*lx+4*lx*ly] - fIn[x+y*lx+2*lx*ly]) + 0.5f*rho*uy + 1.0f/6.0f*rho*ux;
                fIn[x+y*lx+8*lx*ly] = fIn[x+y*lx+6*lx*ly] + 0.5f*(fIn[x+y*lx+2*lx*ly] - fIn[x+y*lx+4*lx*ly]) - 0.5f*rho*uy + 1.0f/6.0f*rho*ux;
	}
 
	    // Outlet: Zou/He BC
	__syncthreads();
	if (x == out && 0 < y && y < ly-1) {
		fIn[dir(3)] = fIn[dir(1)] - 2.0f/3.0f * rho * ux;
		fIn[dir(7)] = fIn[dir(5)] + 0.5f*(fIn[dir(2)]-fIn[dir(4)]) - 0.5f*rho*uy - 1.0f/6.0f*rho*ux;
		fIn[dir(6)] = fIn[dir(8)] + 0.5f*(fIn[dir(4)]-fIn[dir(2)]) + 0.5f*rho*uy - 1.0f/6.0f*rho*ux;
	}


	// Collision
	__syncthreads();
	float cu;
	bool isPS =/* false;*/ (x == PSx && y == PSy);
	if (isPS) {
		rho = 1.0f + PSamplitude * sin(2*M_PI*iteration/PSperiod);
	}
	#pragma unroll
	for (int k = 0; k < 9; ++k) {
		cu = 3.0f * (cx[k]*ux + cy[k]*uy);
		temp = rho * t[k] * ( 1.0f + cu + 0.5*(cu*cu) - 1.5f*(ux*ux + uy*uy));		
		if (isPS) {
			fOut[dir(k)] = temp;
		} else {
			fOut[dir(k)] = fIn[dir(k)] - omega * (fIn[dir(k)] - temp);
		}
	}


	// Obstacle (bounce-back)
	__syncthreads();
	if (bbRegion[x+y*lx] == (char)1) {
		#pragma unroll
		for (int k = 0; k < 9; ++k) {
			fOut[dir(k)] = fIn[dir(opp[k])]; 
		}
	}
}



__global__ void cylinderKernelStream (float *fIn, float *fOut) {
        const int x = bx*bdx + tx;
        const int y = by*bdy + ty;

	// Streaming
	__syncthreads();
	int xSource, ySource;
	#pragma unroll
	for (int k = 0; k < 9; ++k) {
		xSource = x - cx[k];
		ySource = y - cy[k];

		xSource += lx;
		ySource += ly;

		xSource %= lx;
		ySource %= ly;

		fIn[dir(k)] = fOut[xSource + ySource*lx + k*lx*ly];
	}
}

void hostCode() {
	float temp;

	// Set up constants
	float tH[9];
	tH[0] = 4./9.;
	tH[1] = 1./9.;
	tH[2] = 1./9.;
	tH[3] = 1./9.;
	tH[4] = 1./9.;
	tH[5] = 1./36.;
	tH[6] = 1./36.;
	tH[7] = 1./36.;
	tH[8] = 1./36.;	
	cudaMemcpyToSymbol(t, tH, 9*sizeof(float));
	cudaCheckErrors("copying tH to symbol");

	int cxH[9];  
	int cyH[9];
	cxH[0] =  0; cyH[0] =  0;
        cxH[1] =  1; cyH[1] =  0;
        cxH[2] =  0; cyH[2] =  1;
        cxH[3] = -1; cyH[3] =  0;
        cxH[4] =  0; cyH[4] = -1;
        cxH[5] =  1; cyH[5] =  1;
        cxH[6] = -1; cyH[6] =  1;
        cxH[7] = -1; cyH[7] = -1;
        cxH[8] =  1; cyH[8] = -1;
	cudaMemcpyToSymbol(cx, cxH, 9*sizeof(int));
	cudaCheckErrors("copying cxH to symbol");
	cudaMemcpyToSymbol(cy, cyH, 9*sizeof(int));
	cudaCheckErrors("copying cyH to symbol");

	int oppH[9]; // opp = [ 1,   4,  5,  2,  3,    8,   9,   6,   7];
	oppH[0] = 0;
        oppH[1] = 3;
        oppH[2] = 4;
        oppH[3] = 1;
        oppH[4] = 2;
        oppH[5] = 7;
        oppH[6] = 8;
        oppH[7] = 5;
        oppH[8] = 6;
	cudaMemcpyToSymbol(opp, oppH, 9*sizeof(unsigned int));	
	cudaCheckErrors("copying oppH to symbol");

	bool bbRegionH[lx*ly];
	char *bbRegionD = (char *) malloc(9*lx*ly*sizeof(char));
	for (int i = 0; i < lx; ++i) {
		for (int j = 0; j < ly; ++j) {
			if (j==0 || j==ly-1 /*|| (i==obst_x &&*/ /*pow(i-obst_x,2)+pow(j-obst_y,2) <= pow(obst_r,2))*//* || i==0 || i==lx-1*/) {
				bbRegionH[i+j*lx] = (char)1;
			} else {
				bbRegionH[i+j*lx] = (char)0;
			}
/*			if (j == 40) {
				if (i%50 > 5) {
	//				bbRegionH[i+j*lx] = (char)1;
				}
			}
*/
			if (i==leftWall || i == rightWall || (leftWall<=i && i<=rightWall && (j<=btmWall || j>=topWall))) {
//				bbRegionH[i+j*lx] = (char)1;
			}
			if (leftWall<=i && i<=rightWall && j>700) {
//				bbRegionH[i+j*lx] = (char)1;
			}
			if (i==leftWall && (btmWall<j && j<btmWall+slitWidth)) {
//				bbRegionH[i+j*lx] = (char)0;
			}
			if (i==rightWall && (topWall-slitWidth<j && j<topWall)) {
//				bbRegionH[i+j*lx] = (char)0;
			}

		}
	}
	srand(1);
	int X, Y, R, R2;
	for (int n = 0; n < 1500; ++n) {
		X = (rand()%(lx-200) + 100)/1;
		Y = (rand()%ly)/1;
		R = (rand()%20); //3 + 5*rand())/1;
		R2 = R*R;
		for (int i = 0; i < lx; ++i) {
			for (int j = 0; j < ly; ++j) {
				if (pow(i-X,2)+pow(j-Y,2) <= R2) {
					bbRegionH[i+j*lx] = (char)1;
				}
			}
		}
	}

	cudaMalloc(&bbRegionD, lx*ly*sizeof(char));
	cudaCheckErrors("cudaMalloc of bbRegionD");
	cudaMemcpy(bbRegionD, bbRegionH, lx*ly*sizeof(char), cudaMemcpyHostToDevice);
	cudaCheckErrors("copying bbRegionH to device");
//	cudaMemcpyToSymbol(bbRegion, bbRegionH, lx*ly*sizeof(bool));
//	cudaCheckErrors("copying bbRegion to symbol");
	
	
	// Initialize Poiseuille equilibrium across the lattice
	float ux;
	float uy = 0.0f;	
	float rho = 1.0f;
	float cu;
	float fInH[9*lx*ly];
	for (int j = 0; j < ly; ++j) {
		ux = 4 * uMax;
		ux /= ((ly-2)*(ly-2));
		ux *= ((j-0.5f)*(ly-2) - (j-0.5f)*(j-0.5f));

		for (int i = 0; i < lx; ++i) {
			for (int k = 0; k < 9; ++k) {
				cu = 3 * (cxH[k]*ux + cyH[k]*uy);
				fInH[i + j*lx + k*lx*ly] = rho * tH[k] * ( 1 + cu + 0.5*(cu*cu) - 1.5*(ux*ux + uy*uy));
			//	fInH[i + j*lx + k*lx*ly] = tH[k];
			}
		}
	}

	// transfer lattice to device
	float *fInD = (float *) malloc(9*lx*ly*sizeof(float));
	cudaMalloc(&fInD, 9*lx*ly*sizeof(float));
	cudaCheckErrors("cudaMalloc of fInD");
	cudaMemcpy(fInD, fInH, 9*lx*ly*sizeof(float), cudaMemcpyHostToDevice);
	cudaCheckErrors("copying fInH to device");

	float *fOutD = (float *) malloc(9*lx*ly*sizeof(float));
	cudaMalloc(&fOutD, 9*lx*ly*sizeof(float));
	cudaCheckErrors("allocating fOutD on host");

	// set up grid
	dim3 blocksPerGrid(iceil(lx,BlockFinalSizeX), iceil(ly,BlockFinalSizeY));
//	dim3 blocksPerGrid(iceil(lx,BlockSizeX), iceil(ly,BlockSizeY));
	dim3 threadsPerBlock(BlockSizeX, BlockSizeY);

	// LAUNCH THE KERNEL
	printf("Launching kernel.\n");
	cudaDeviceSynchronize();
	cudaCheckErrors("synchronizing before kernel launch");	

	// For our images.
	char rgb[lx*ly*3];
	float u[lx*ly];
	temp = 0.0f;
	HsvColor hsv;
	hsv.s = 255;
	hsv.v = 255;
	RgbColor tricolor;

	double timer = 0.;
	startTimer(&timer);
	for (int iteration = 0; iteration < iterations/stepsPerKernel; ++iteration) {
		tiledKernel <<< blocksPerGrid,threadsPerBlock >>> (fInD, fOutD, bbRegionD, iteration);

//		cylinderKernelCollide <<< blocksPerGrid,threadsPerBlock >>> (fInD, fOutD, bbRegionD, iteration);
//		cudaCheckErrors("launching collide kernel");
//		cylinderKernelStream <<< blocksPerGrid,threadsPerBlock >>> (fInD, fOutD);
//		cudaCheckErrors("launching stream kernel");
		
		// Draw a pretty picture!
		if ((iteration % tPlot) == tPlot-1) {
			printf("\n\ndrawing %d (%d)\n", iteration, iteration*stepsPerKernel);
			cudaMemcpy(fInH, fInD, 9*lx*ly*sizeof(float), cudaMemcpyDeviceToHost);
			cudaCheckErrors("copying fInD to host");
		
			for (int i = 0; i < lx; ++i) {
				for (int j = 0; j < ly; ++j) {
					ux = 0.0f;
					uy = 0.0f;
					if (j==ly/2) {
						if (i==0) printf("\nfor LHS: \n");
						if (i==lx-1) printf("for RHS: \n");
					}
					for (int k = 0; k < 9; ++k) {
						if (j==ly/2 && (i==0 || i==lx-1)) 
							printf("--%d : %f\n", k, fInH[i+j*lx+k*lx*ly]);
						ux += cxH[k] * fInH[i + j*lx + k*lx*ly];
						uy += cyH[k] * fInH[i + j*lx + k*lx*ly];
					}
					u[i + j*lx] = pow(ux*ux + uy*uy, 0.5);
					if (temp < u[i + j*lx]) {
						temp = u[i + j*lx];
					}
				}
			}
			temp /= 200.0f;
			for (int n = 0; n < lx*ly; ++n) {
				u[n] /= temp;
			}

			for (int n = 0; n < lx*ly; ++n) {
				hsv.h = 200 - (int) u[n];
				tricolor = HsvToRgb(hsv);
				if (bbRegionH[n]) {
					rgb[3*n] = 0;
					rgb[3*n + 1] = 0;
					rgb[3*n + 2] = 0;
				} else {
					rgb[3*n] = tricolor.r; 
					rgb[3*n + 1] = tricolor.g;
					rgb[3*n + 2] = tricolor.b;
				}
			}

			write_bmp(iteration*stepsPerKernel, lx, ly, rgb);
		}
	}
	printf("Kernel launch complete.\n");
	double time = stopNreadTimer(&timer);
	printf("It took %8.2f s to run %8.2f Mlu, which is %8.2f Mlu/s (baseline times %8.2f).\n", time, (double)lx*(double)ly*(double)iterations/pow(10,6), (double)lx*(double)ly*(double)iterations/(pow(10,6)*time), ((double)lx*(double)ly*(double)iterations/time)/3150000.);

}
