#ifndef tcFftPlan1d_hpp
#define tcFftPlan1d_hpp

#include <CmnUtils/CufftPlans/inc/tcFftPlanBase.hpp>

namespace CmnUtils
{

namespace CufftPlans
{

/// Class encapsulating a 1-dimensional FFT plan
class tcFftPlan1d : public tcFftPlanBase
{
public:
   /// Can construct with or without a workspace (FFT plan manager maintains one single workspace)
   tcFftPlan1d(cufftType ahType, int anX, int anBatch, bool abAllocateWorkspace=true);

   /// Gets the X-dimension
   int X(void) const;

   /// Gets the batch count
   int Batch(void) const;

   /// Hash code for this plan
   std::size_t Hash(void) const override;

private:

   /// X-dimension for the FFT plan
   int mnX;

   /// Batch count for the FFT plan
   int mnBatch;
};

}
}

#endif
