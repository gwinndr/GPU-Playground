#ifndef tcFftPlanBase_hpp
#define tcFftPlanBase_hpp

#include <teFftPlanType.hpp>
#include <cufft.h>
#include <cstdlib>

namespace CmnUtils
{

namespace CufftPlans
{

/// Generic base class for FFT plans
/// Designed for use with the FFT Plan manager
class tcFftPlanBase
{
public:
   /// Destroys the FFT plan
   virtual ~tcFftPlanBase();

   /// Moves data (constructor)
   /// NOTE: Usage of moved object past this point is undefined
   tcFftPlanBase(tcFftPlanBase&& arrcOther);

   /// Moves data (equal operator), will delete existing data as well (if it exists)
   /// NOTE: Usage of moved object past this point is undefined
   virtual tcFftPlanBase& operator=(tcFftPlanBase&& arrcOther);

   /// Gets the Handle associated with this plan
   cufftHandle & Handle(void);

   /// Gets the Handle associated with this plan (const)
   const cufftHandle & Handle(void) const;

   /// Gets the cuFFT type associated with this plan
   cufftType Type(void) const;

   /// Hash code for this plan.
   /// NOTE: Deriving classes must override this method and combine
   virtual std::size_t Hash(void) const;

   // delete copy
   tcFftPlanBase(const tcFftPlanBase&) = delete;
   tcFftPlanBase& operator=(const tcFftPlanBase&) = delete;

protected:
   /// Creates the handle and sets the auto workspace allocation based on given boolean
   /// Must be built by inheriting class.
   /// NOTE: INHERITING CLASS MUST CREATE THE ACTUAL PLAN!
   tcFftPlanBase(teFftPlanType aePlanType, cufftType ahType, bool abAllocateWorkspace=true);

private:
   /// Boolean denoting if this class has ownership (i.e. not moved)
   bool mbOwnership = true;

   /// Underlying type of plan (cufftPlan1D, cufftPlanMany, etc.)
   teFftPlanType mePlanType;

   /// Handle associated with this plan
   cufftHandle mhHandle;

   /// Type of the FFT (C2C, R2C, etc.)
   cufftType mhType;

   /// Frees resources if owned
   void FreeResources(void);

   /// Performs the actual move of the handle to another FFT plan base
   void MoveObject(tcFftPlanBase& arcOther);
};

}

}

#endif
