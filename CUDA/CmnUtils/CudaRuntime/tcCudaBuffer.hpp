#ifndef tcCudaBuffer_hpp
#define tcCudaBuffer_hpp

#include <CmnUtils/CudaRuntime/CheckError.hpp>

#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstring>
#include <sstream>

namespace CmnUtils
{

namespace CudaRuntime
{

/// Class to manage CUDA buffers (with pinned memory) in a RAII manner
template<class tcType>
class tcCudaBuffer
{
public:
   /// Allocates the device buffer (and pinned memory if specified)
   tcCudaBuffer(int anNumberOfItems, bool abAllocPinned=false);

   /// Destroys allocated buffers
   virtual ~tcCudaBuffer();

   /// Checks if there's a pinned memory buffer
   bool HasPinned(void);

   /// Gets the device pointer
   tcType * DeviceBuffer(void);

   /// Gets the device pointer (const version)
   const tcType * DeviceBuffer(void) const;

   /// Gets the pinned pointer
   tcType * PinnedBuffer(void);

   /// Gets the pinned pointer (const version)
   const tcType * PinnedBuffer(void) const;

   /// Resets device buffer (memsets to 0) using the given stream
   void ResetDeviceBuffer(cudaStream_t ahStrm);

   /// Resets pinned buffer (memsets to 0)
   void ResetPinnedBuffer(void);

   /// Copies given array into the pinned buffer
   void CopyArrayToPinned(const tcType * apcArr, std::size_t anNumElems);

   /// Copies data from Pinned->Device using the given stream
   void CopyPinnedToDevice(cudaStream_t ahStrm);

   /// Copies data from Device->Pinned using the given stream
   void CopyDeviceToPinned(cudaStream_t ahStrm);

   /// Copies data from this class's device buffer into the given class's device buffer
   void CopyDeviceToDevice(tcCudaBuffer & arcTargetBuffer, cudaStream_t ahStrm);

   /// Gets the size in number of items
   int NumItems(void);

   /// Gets the size in number of bytes
   int SizeBytes(void);

   // delete copy/move (rule of 5)
   // TODO: Implement the move operator (useful for copy-ellision, etc.)
   tcCudaBuffer(const tcCudaBuffer&) = delete;
   tcCudaBuffer& operator=(const tcCudaBuffer&) = delete;
   tcCudaBuffer(tcCudaBuffer&&) = delete;
   tcCudaBuffer& operator=(tcCudaBuffer&&) = delete;

private:
   /// Helper method to throw exception if pinned memory is not allocated
   void ThrowIfNoPinned(void);

   /// Helper method to throw exception if given items exceed our number of items
   void ThrowIfItemsExceeded(int anGivenItems);

   /// GPU Buffer
   tcType * mpcDeviceBuf = nullptr;

   /// Host Pinned Buffer (can be nullptr if no pinned memory)
   tcType * mpcPinnedBuf = nullptr;

   /// Number of items
   int mnNumItems = 0;
};

// *************************************************************************************************
template<class tcType>
tcCudaBuffer<tcType>::tcCudaBuffer(int anNumberOfItems, bool abAllocPinned)
   :
   mnNumItems(anNumberOfItems)
{
   // Require a positive number of items
   if(mnNumItems <= 0)
   {
      throw std::runtime_error("Number of items must be positive! NumItems=" + mnNumItems);
   }

   // Allocate GPU memory
   CheckError(
      cudaMalloc(&mpcDeviceBuf, SizeBytes()),
      __FILE__, __LINE__
   );

   // Allocate Pinned memory if specified
   if(abAllocPinned)
   {
      CheckError(
         cudaMallocHost(&mpcPinnedBuf, SizeBytes()),
         __FILE__, __LINE__
      );
   }
}

// *************************************************************************************************
template<class tcType>
tcCudaBuffer<tcType>::~tcCudaBuffer()
{
   // Deallocate GPU memory
   CheckError(
      cudaFree(mpcDeviceBuf),
      __FILE__, __LINE__
   );

   // Deallocate Pinned memory if allocated
   if(mpcPinnedBuf != nullptr)
   {
      CheckError(
         cudaFreeHost(mpcPinnedBuf),
         __FILE__, __LINE__
      );
   }
}

// *************************************************************************************************
template<class tcType>
bool tcCudaBuffer<tcType>::HasPinned(void)
{
   return mpcPinnedBuf != nullptr;
}

// *************************************************************************************************
template<class tcType>
tcType * tcCudaBuffer<tcType>::DeviceBuffer(void)
{
   return mpcDeviceBuf;
}

// *************************************************************************************************
template<class tcType>
const tcType * tcCudaBuffer<tcType>::DeviceBuffer(void) const
{
   return mpcDeviceBuf;
}

// *************************************************************************************************
template<class tcType>
tcType * tcCudaBuffer<tcType>::PinnedBuffer(void)
{
   ThrowIfNoPinned();
   return mpcPinnedBuf;
}

// *************************************************************************************************
template<class tcType>
const tcType * tcCudaBuffer<tcType>::PinnedBuffer(void) const
{
   ThrowIfNoPinned();
   return mpcPinnedBuf;
}

// *************************************************************************************************
template<class tcType>
void tcCudaBuffer<tcType>::ResetDeviceBuffer(cudaStream_t ahStrm)
{
   CheckError(
      cudaMemsetAsync(
         DeviceBuffer(), 0u, 
         SizeBytes(), ahStrm),
      __FILE__, __LINE__
   );
}

// *************************************************************************************************
template<class tcType>
void tcCudaBuffer<tcType>::ResetPinnedBuffer(void)
{
   std::memset(PinnedBuffer(), 0u, SizeBytes());
}

// *************************************************************************************************
template<class tcType>
void tcCudaBuffer<tcType>::CopyArrayToPinned(const tcType * apcArr, std::size_t anNumElems)
{
   ThrowIfItemsExceeded(anNumElems);

   std::copy(apcArr, apcArr + anNumElems, PinnedBuffer());
}

// *************************************************************************************************
template<class tcType>
void tcCudaBuffer<tcType>::CopyPinnedToDevice(cudaStream_t ahStrm)
{
   CheckError(
      cudaMemcpyAsync(
         DeviceBuffer(), PinnedBuffer(), 
         SizeBytes(),
         cudaMemcpyHostToDevice, ahStrm
      ),
      __FILE__, __LINE__
   );
}

// *************************************************************************************************
template<class tcType>
void tcCudaBuffer<tcType>::CopyDeviceToPinned(cudaStream_t ahStrm)
{
   CheckError(
      cudaMemcpyAsync(
         PinnedBuffer(), DeviceBuffer(), 
         SizeBytes(),
         cudaMemcpyDeviceToHost, ahStrm
      ),
      __FILE__, __LINE__
   );
}

// *************************************************************************************************
template<class tcType>
void tcCudaBuffer<tcType>::CopyDeviceToDevice(tcCudaBuffer & arcTargetBuffer, cudaStream_t ahStrm)
{
   arcTargetBuffer.ThrowIfItemsExceeded(mnNumItems);

   CheckError(
      cudaMemcpyAsync(
         arcTargetBuffer.DeviceBuffer(), DeviceBuffer(), 
         SizeBytes(),
         cudaMemcpyDeviceToDevice, ahStrm
      ),
      __FILE__, __LINE__
   );
}

// *************************************************************************************************
template<class tcType>
int tcCudaBuffer<tcType>::NumItems(void)
{
   return mnNumItems;
}

// *************************************************************************************************
template<class tcType>
int tcCudaBuffer<tcType>::SizeBytes(void)
{
   return NumItems() * sizeof(tcType);
}

// *************************************************************************************************
template<class tcType>
void tcCudaBuffer<tcType>::ThrowIfNoPinned(void)
{
   if(!HasPinned())
   {
      throw std::runtime_error("No pinned memory has been allocated for this buffer!");
   }
}

// *************************************************************************************************
template<class tcType>
void tcCudaBuffer<tcType>::ThrowIfItemsExceeded(int anGivenItems)
{
   if(anGivenItems > mnNumItems)
   {
      std::stringstream lcErr;
      lcErr << 
         "Items given exceeds the size of our buffer! GivenItems=" << anGivenItems << 
         " BufferItems=" << mnNumItems;
      throw std::runtime_error(lcErr.str());
   }
}

}
}

#endif
