// Simple Fused-Multiply Add (FMA) kernel
// Just a test application to verify things are working!

#include <CmnUtils/CudaRuntime/tcCudaBuffer.hpp>
#include <CmnUtils/CudaRuntime/tcCudaStream.hpp>
#include <CmnUtils/SetupAndVerify/VerifyResults.hpp>
#include <CmnUtils/SetupAndVerify/Random.hpp>

#include <iostream>

using namespace CmnUtils;

// GPU
// D = A .* B + C
__global__ void ElementFmaKernel(
   const float * apfA, const float * apfB, const float * apfC, float * apfD, int anNumElems)
{
   int lnThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
   if(lnThreadIdx < anNumElems)
   {
      apfD[lnThreadIdx] = apfA[lnThreadIdx] * apfB[lnThreadIdx] + apfC[lnThreadIdx];
   }
}

// CPU
// D = A .* B + C
void ElementFmaCpu(
   const float * apfA, const float * apfB, const float * apfC, float * apfD, int anNumElems)
{
   for(int lnI = 0; lnI < anNumElems; ++lnI)
   {
      apfD[lnI] = apfA[lnI] * apfB[lnI] + apfC[lnI];
   }
}

// Main
int main()
{
   // Number of elements and the error tolerance
   static constexpr int pnNumElems = 1e6;
   static constexpr float prTol = 1e-6;
   static constexpr int pnNumThreads = 256;
   static constexpr int pnNumBlocks = pnNumElems / pnNumThreads + (pnNumElems % pnNumThreads > 0);

   // Create stream for all operations
   CudaRuntime::tcCudaStream lcStrm;

   // Create CUDA buffers out of A, B, and C random values
   CudaRuntime::tcCudaBuffer<float> lcA = 
      SetupAndVerify::RandomCudaRealUniform<float>(1, 100, pnNumElems, lcStrm.Stream());
   CudaRuntime::tcCudaBuffer<float> lcB = 
      SetupAndVerify::RandomCudaRealUniform<float>(1, 100, pnNumElems, lcStrm.Stream());
   CudaRuntime::tcCudaBuffer<float> lcC = 
      SetupAndVerify::RandomCudaRealUniform<float>(1, 100, pnNumElems, lcStrm.Stream());

   // Create output device buffer and CPU output vector
   CudaRuntime::tcCudaBuffer<float> lcResultDevice(pnNumElems, true);
   std::vector<float> lcResultCpu(pnNumElems);

   // Compute CPU result
   ElementFmaCpu(lcA.PinnedBuffer(), lcB.PinnedBuffer(), lcC.PinnedBuffer(), lcResultCpu.data(), pnNumElems);

   // Compute GPU results
   ElementFmaKernel<<<pnNumThreads, pnNumBlocks, 0, lcStrm.Stream()>>>
      (lcA.DeviceBuffer(), lcB.DeviceBuffer(), lcC.DeviceBuffer(), lcResultDevice.DeviceBuffer(),
      pnNumElems);
   
   // Copy results and sync
   lcResultDevice.CopyDeviceToPinned(lcStrm.Stream());
   lcStrm.Sync();

   // Compare
   SetupAndVerify::VerifyArrays<float>(
      lcResultDevice.PinnedBuffer(), lcResultCpu.data(), 
      pnNumElems, prTol);

   std::cout << "Arrays are equal!" << std::endl;
}
