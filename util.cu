#include <stdio.h>
// ---------------------------------------------------------------
// General CUDA GPU utility functions that are executed on the host
// ---------------------------------------------------------------

// Routine that Selects between multiple GPU devices
// if GPU device number invalid return -1 
int DeviceSelect(int device_id) {
   int num_devices,device=-1;
   cudaGetDeviceCount(&num_devices);
   if (num_devices>0) {
      if (device_id >=0 && device_id < num_devices) {
         device=device_id;
         cudaSetDevice(device);
      }
      else {
         device=-1;
         printf("Error: Cuda Device %d does not exist\n",device_id);
      }
   }
   return device;
}

// routine that outputs GPU info for the selected device
// [Note: no error checking device must be valid!]
void DeviceInfo(int device_id) {
   cudaDeviceProp properties;
   if (device_id>=0) {
      cudaGetDeviceProperties(&properties, device_id);
      printf("Selected CUDA Device (%d)= %s Characteristics\n",device_id,
         properties.name);
      printf("    Total Global Memory = %u\n",properties.totalGlobalMem);
      printf("    Total Constant Memory = %u\n",properties.totalConstMem);
      printf("    Shared Memory Per Block = %u\n",properties.sharedMemPerBlock);
      printf("    Registers Per Block = %d\n",properties.regsPerBlock);
      printf("    Warp Size = %d\n",properties.warpSize);
      printf("    Number of SM = %d\n",properties.multiProcessorCount);
      printf("    Maximum Number of Threads Per Block = %d\n",
         properties.maxThreadsPerBlock);
      printf("    Maximum Number of Threads Per SM = %d\n",
         properties.maxThreadsPerMultiProcessor);
      printf("    Maximum Block Dimensions = (%u,%u,%u)\n",
         properties.maxThreadsDim[0],properties.maxThreadsDim[1],
         properties.maxThreadsDim[2]);
      printf("    Maximum Grid Size = (%u,%u,%u)\n",properties.maxGridSize[0],
         properties.maxGridSize[1],properties.maxGridSize[2]);
      printf("    Compute Mode %d\n",properties.computeMode);
      printf("    Number of concurrent Kernels = %d\n",
         properties.concurrentKernels);
      printf("    Base Processor Clock Rate = %d\n",properties.clockRate);
      printf("    Memory Clock Rate = %d\n",properties.memoryClockRate);
      printf("    L2 Cache Size = %d\n",properties.l2CacheSize);
      printf("\n");
   }
}

