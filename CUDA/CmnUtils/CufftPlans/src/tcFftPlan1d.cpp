#include <tcFftPlan1d.hpp>
#include <CmnUtils/CudaRuntime/CheckError.hpp>
#include <CmnUtils/Utils/Hash.hpp>

namespace CmnUtils
{

namespace CufftPlans
{

// *************************************************************************************************
tcFftPlan1d::tcFftPlan1d(cufftType ahType, int anX, int anBatch, bool abAllocateWorkspace) :
   tcFftPlanBase(teFftPlanType::eeCufftPlan1D, ahType, abAllocateWorkspace),
   mnX(anX),
   mnBatch(anBatch)
{
   // Create 1-D plan
   CudaRuntime::CheckError(
      cufftPlan1d(&Handle(), mnX, Type(), anBatch),
      __FILE__, __LINE__
   );
}

// *************************************************************************************************
int tcFftPlan1d::X(void) const
{
   return mnX;
}

// *************************************************************************************************
int tcFftPlan1d::Batch(void) const
{
   return mnBatch;
}

// *************************************************************************************************
std::size_t tcFftPlan1d::Hash(void) const
{
   std::size_t lnHash = tcFftPlanBase::Hash();

   Utils::hash_combine(lnHash, mnX);
   Utils::hash_combine(lnHash, mnBatch);

   return lnHash;
}

}
}
