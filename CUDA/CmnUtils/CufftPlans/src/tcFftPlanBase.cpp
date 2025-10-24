#include <tcFftPlanBase.hpp>
#include <CmnUtils/CudaRuntime/CheckError.hpp>
#include <CmnUtils/Utils/Hash.hpp>
#include <bitset>

namespace CmnUtils
{

namespace CufftPlans
{

// *************************************************************************************************
tcFftPlanBase::tcFftPlanBase(teFftPlanType aePlanType, cufftType ahType, bool abAllocateWorkspace)
   :
   mePlanType(aePlanType),
   mhType(ahType)
{
   CudaRuntime::CheckError(
      cufftCreate(&mhHandle),
      __FILE__, __LINE__
   );

   // Set auto allocation
   CudaRuntime::CheckError(
      cufftSetAutoAllocation(mhHandle, abAllocateWorkspace),
      __FILE__, __LINE__
   );
}

// *************************************************************************************************
tcFftPlanBase::~tcFftPlanBase()
{
   FreeResources();
}

/// Moves data (constructor)
/// NOTE: Usage of moved object past this point is undefined
tcFftPlanBase::tcFftPlanBase(tcFftPlanBase&& arrcOther)
{
   MoveObject(arrcOther);
}

/// Moves data (equal operator), will delete existing data as well (if it exists)
/// NOTE: Usage of moved object past this point is undefined
tcFftPlanBase& tcFftPlanBase::operator=(tcFftPlanBase&& arrcOther)
{
   if (this != &arrcOther)
   {
      FreeResources();
      MoveObject(arrcOther);
   }
   return *this;
}

// *************************************************************************************************
cufftHandle & tcFftPlanBase::Handle(void)
{
   return mhHandle;
}

// *************************************************************************************************
const cufftHandle & tcFftPlanBase::Handle(void) const
{
   return mhHandle;
}

// *************************************************************************************************
cufftType tcFftPlanBase::Type(void) const
{
   return mhType;
}

// *************************************************************************************************
std::size_t tcFftPlanBase::Hash(void) const
{
   static const std::hash<cufftType> pcHasher;

   std::size_t lnHash = pcHasher(mhType);
   Utils::hash_combine(lnHash, mePlanType);

   return lnHash;
}

// *************************************************************************************************
void tcFftPlanBase::FreeResources(void)
{
   // Only delete if we actually own the handle
   if(mbOwnership)
   {
      CudaRuntime::CheckError(
         cufftDestroy(mhHandle),
         __FILE__, __LINE__
      );
   }
}

// *************************************************************************************************
void tcFftPlanBase::MoveObject(tcFftPlanBase& arcOther)
{
   mhHandle = arcOther.mhHandle;
   mbOwnership = true;
   arcOther.mbOwnership = false;
}

}
}
