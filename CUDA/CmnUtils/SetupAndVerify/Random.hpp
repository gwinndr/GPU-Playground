#ifndef Random_hpp
#define Random_hpp

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

}
}

#endif
