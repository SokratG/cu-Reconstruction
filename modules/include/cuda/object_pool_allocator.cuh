#ifndef CUPHOTO_LIB_OBJECT_POOL_ALLOCATOR_CUH
#define CUPHOTO_LIB_OBJECT_POOL_ALLOCATOR_CUH

#include "types.cuh"
#include "CudaUtils/cudaUtility.cuh"

namespace cuphoto {

class CudaManaged {
public:
    using void_ptr = void*;
public:
    __host__ void_ptr operator new(std::size_t len) {
        void_ptr ptr = nullptr;
        CUDA(cudaMallocManaged(&ptr, len, cudaMemAttachGlobal));
        CUDA(cudaDeviceSynchronize());
        return ptr;
    }
    
    __host__ void operator delete(void_ptr ptr) {
        CUDA(cudaDeviceSynchronize());
        CUDA(cudaFree(ptr));
    }
};

template<class T>
struct CudaObjectPoolAllocator : CudaManaged {
public:
    using Ptr = CudaObjectPoolAllocator*;
    using Ref = CudaObjectPoolAllocator&;
    using PRef = CudaObjectPoolAllocator::Ptr&;
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    using size_type = size_t;
    using difference_ptr = std::ptrdiff_t;

    template<typename U>
    struct rebind {
        using other = CudaObjectPoolAllocator<U>;
    };
public:
    ~CudaObjectPoolAllocator() {
        free();
    }

    __host__ void reserve(const std::size_t pool_size) {
        if (pool_size == 0) {
            LogError(LOG_CUDA "CudaObjectPoolAllocator::CudaObjectPoolAllocator() -- can't allocate query CUDA memory %lu\n", pool_size);
            throw std::bad_alloc();
        }
        free();
        mem_pool_size = pool_size;
        current_idx = 0;
        CUDA(cudaMallocManaged(&current_ptr, mem_pool_size * sizeof(T), cudaMemAttachGlobal));
    }

    __host__ void free() {
        if (current_ptr != nullptr)
            CUDA(cudaFree(current_ptr));
    }

    __host__ __device__ pointer allocate() {
        if (current_idx > mem_pool_size) {
            // LogError(LOG_CUDA "OctreeObjectPoolAllocator::allocate() -- can't allocate OctreeNode. \
            //                    The memory limit is exceeded. %lu\n", mem_pool_size);
            return nullptr;
        }

        if (current_ptr == nullptr)
            return nullptr;
        
        pointer top_ptr = &current_ptr[current_idx];
        current_idx += 1;
        return top_ptr;
    }

    __host__ __device__ void reset() {
        current_idx = 0;
    }

    __host__ __device__ ui64 pool_size() const {
        return mem_pool_size;
    }

    __host__ __device__ ui64 allocated_size() const {
        return current_idx;
    }

protected:
    pointer current_ptr = nullptr;
    std::size_t mem_pool_size;
    ui64 current_idx;
};

}


#endif