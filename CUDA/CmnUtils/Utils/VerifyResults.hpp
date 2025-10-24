#ifndef VerifyResults_hpp
#define VerifyResults_hpp

#include <stdexcept>
#include <sstream>

namespace CmnUtils
{
namespace SetupAndVerify
{

// Verifies arrays match within tolerance
template<class tcType>
void VerifyArrays(
   const tcType * apcResult1, const tcType * apcResult2, int anNumElems, tcType acTol)
{
   for(int lnI = 0; lnI < anNumElems; ++lnI)
   {
      tcType lcDiff = apcResult1[lnI] - apcResult2[lnI];
      if(lcDiff >= acTol)
      {
         std::stringstream lcErr;
         lcErr << 
            "BAD MATCH AT ELEM=" << lnI << "! Lhs=" << apcResult1[lnI] << "  "
            "Rhs=" << apcResult2[lnI] << "Diff=" << lcDiff << "  Tol=" << acTol;
            
         throw std::runtime_error(lcErr.str());
      }
   }
}

}
}

#endif
