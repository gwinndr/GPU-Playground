#ifndef teFftPlanType_hpp
#define teFftPlanType_hpp

namespace CmnUtils
{

namespace CufftPlans
{

/// Enum defining different types of of FFT plans supported by cuFFT
enum teFftPlanType : int
{
   eeCufftPlan1D,
   eeCufftPlan2D,
   eeCufftPlan3D,
   eeCufftPlanMany
};

}
}

#endif
