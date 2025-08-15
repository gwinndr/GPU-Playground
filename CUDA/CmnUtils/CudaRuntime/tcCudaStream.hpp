#ifndef tcCudaStream_hpp
#define tcCudaStream_hpp

#include <CmnUtils/CudaRuntime/CheckError.hpp>

#include <cuda_runtime.h>

namespace CmnUtils
{

namespace CudaRuntime
{

/// Class to manage CUDA streams in an RAII fashion
class tcCudaStream
{
public:
   /// Creates the CUDA stream
   tcCudaStream(unsigned anFlags=cudaStreamNonBlocking);

   /// Destroys the CUDA stream
   virtual ~tcCudaStream();

   /// Gets the allocated CUDA stream
   cudaStream_t Stream(void);

   /// Blocks until the stream completes all work
   void Sync(void);

   // delete copy/move (rule of 5)
   tcCudaStream(const tcCudaStream&) = delete;
   tcCudaStream& operator=(const tcCudaStream&) = delete;
   tcCudaStream(tcCudaStream&&) = delete;
   tcCudaStream& operator=(tcCudaStream&&) = delete;

private:
   cudaStream_t mhStrm;
};

// *************************************************************************************************
tcCudaStream::tcCudaStream(unsigned anFlags)
{
   CheckError(
      cudaStreamCreateWithFlags(&mhStrm, anFlags),
      __FILE__, __LINE__
   );
}

// *************************************************************************************************
tcCudaStream::~tcCudaStream()
{
   CheckError(
      cudaStreamDestroy(mhStrm),
      __FILE__, __LINE__
   );
}

// *************************************************************************************************
cudaStream_t tcCudaStream::Stream(void)
{
   return mhStrm;
}

// *************************************************************************************************
void tcCudaStream::Sync(void)
{
   CheckError(
      cudaStreamSynchronize(mhStrm),
      __FILE__, __LINE__
   );
}

}
}

#endif
