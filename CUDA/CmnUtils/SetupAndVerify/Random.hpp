#ifndef Random_hpp
#define Random_hpp

#include <CmnUtils/CudaRuntime/tcCudaBuffer.hpp>

#include <cuda_runtime.h>
#include <random>
#include <optional>
#include <chrono>

namespace CmnUtils
{
namespace SetupAndVerify
{

/// Creates a uniformly random array (vector)
template<typename trPrecision>
std::vector<trPrecision> RandomVectorRealUniform(
   trPrecision arStart, trPrecision arEnd, 
   int anNumElems, std::optional<long long> anOptSeed=std::nullopt)
{
   long long lnSeed = (anOptSeed.has_value()) ? 
      anOptSeed.value() : 
      std::chrono::high_resolution_clock::now().time_since_epoch().count();

   // Mersenne Twister
   std::mt19937 lcTwister(lnSeed);
   std::uniform_real_distribution<trPrecision> lcRealDist(arStart, arEnd);

   std::vector<trPrecision> lcRandomNums;
   lcRandomNums.reserve(anNumElems);

   for(int lnElem = 0; lnElem < anNumElems; ++lnElem)
   {
      lcRandomNums.push_back(lcRealDist(lcTwister));
   }

   return lcRandomNums;
}

/// Creates a uniformly random GPU buffer and starts the copy from pinned to device using the
/// provided CUDA stream
template<typename trPrecision>
CudaRuntime::tcCudaBuffer<trPrecision> RandomCudaRealUniform(
   trPrecision arStart, trPrecision arEnd, 
   int anNumElems, cudaStream_t ahStrm,
   std::optional<long long> anOptSeed=std::nullopt)
{
   std::vector<trPrecision> lcRandomNums = RandomVectorRealUniform(
      arStart, arEnd, anNumElems, anOptSeed);
   
   CudaRuntime::tcCudaBuffer<trPrecision> lcCudaBuf(anNumElems, true);
   lcCudaBuf.CopyArrayToPinned(lcRandomNums.data(), lcRandomNums.size());
   lcCudaBuf.CopyPinnedToDevice(ahStrm);

   return lcCudaBuf;
}

}
}

#endif
