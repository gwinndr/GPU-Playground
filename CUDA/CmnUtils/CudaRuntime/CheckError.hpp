#ifndef CheckError_hpp
#define CheckError_hpp

#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>

namespace CmnUtils
{

namespace CudaRuntime
{

/// Throws exception for anything other than success
void CheckError(cudaError_t ahErr, const char * apnFile, int anLine)
{
   if(ahErr != cudaSuccess)
   {
      std::stringstream lcErrStrm;
      lcErrStrm << "Error on " << apnFile << ":" << anLine << ". Call did not return cudaSuccess. "
         "ErrorCode=" << ahErr;

      throw std::runtime_error(lcErrStrm.str());
   }
}

}
}

#endif
