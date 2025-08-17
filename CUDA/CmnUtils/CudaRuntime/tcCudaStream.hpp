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

   /// Moves data (constructor)
   /// NOTE: Usage of moved object past this point is undefined
   tcCudaStream(tcCudaStream&& arrcOther);

   /// Moves data (equal operator), will delete existing data as well (if it exists)
   /// NOTE: Usage of moved object past this point is undefined
   tcCudaStream& operator=(tcCudaStream&& arrcOther);

   /// Gets the allocated CUDA stream
   cudaStream_t Stream(void);

   /// Blocks until the stream completes all work
   void Sync(void);

   // delete copy
   tcCudaStream(const tcCudaStream&) = delete;
   tcCudaStream& operator=(const tcCudaStream&) = delete;

private:
   /// Helper method to do actual moving of data when the move operator is used
   void MoveObject(tcCudaStream& arcOther);

   /// Helper method to free resources and reset the class
   void FreeResources(void);

   /// CUDA stream
   cudaStream_t mhStrm = nullptr;
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
   
}

// *************************************************************************************************
tcCudaStream::tcCudaStream(tcCudaStream&& arrcOther)
{
   MoveObject(arrcOther);
}

// *************************************************************************************************
tcCudaStream& tcCudaStream::operator=(tcCudaStream&& arrcOther)
{
   if (this != &arrcOther)
   {
      FreeResources();
      MoveObject(arrcOther);
   }
   return *this;
}

// *************************************************************************************************
void tcCudaStream::MoveObject(tcCudaStream& arcOther)
{
   mhStrm = arcOther.mhStrm;
   arcOther.mhStrm = nullptr;
}
// *************************************************************************************************
void tcCudaStream::FreeResources(void)
{
   if(mhStrm != nullptr)
   {
      CheckError(
         cudaStreamDestroy(mhStrm),
         __FILE__, __LINE__
      );
      mhStrm = nullptr;
   }
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
