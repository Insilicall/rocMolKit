// Isolated repro for the v0.2.0-alpha non-deterministic segfault.
// No RDKit, no boost-python — just hipMallocAsync / hipFreeAsync
// patterns mirroring AsyncDeviceVector.
//
// If this crashes intermittently → bug is in AsyncDeviceVector / runtime.
// If never crashes → bug is in interaction with RDKit / boost-python.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

#define HIP_CHECK(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "HIP error %d at %s:%d\n", e, __FILE__, __LINE__); std::abort(); } } while (0)

template <typename T>
class MiniAsyncVec {
public:
    MiniAsyncVec(size_t n, hipStream_t s) : size_(n), stream_(s) {
        if (n > 0) HIP_CHECK(hipMallocAsync(&data_, n * sizeof(T), s));
    }
    ~MiniAsyncVec() {
        if (data_ != nullptr) HIP_CHECK(hipFreeAsync(data_, stream_));
    }
    MiniAsyncVec(const MiniAsyncVec&) = delete;
    MiniAsyncVec(MiniAsyncVec&& o) noexcept
        : size_(o.size_), data_(o.data_), stream_(o.stream_) { o.data_ = nullptr; }
    void zero() { if (size_) HIP_CHECK(hipMemsetAsync(data_, 0, size_*sizeof(T), stream_)); }
    void resize(size_t n) {
        if (n == size_) return;
        if (n == 0) { if (data_) { HIP_CHECK(hipFreeAsync(data_, stream_)); data_ = nullptr; } size_ = 0; return; }
        T* nd; HIP_CHECK(hipMallocAsync(&nd, n*sizeof(T), stream_));
        if (size_ && data_) {
            HIP_CHECK(hipMemcpyAsync(nd, data_, std::min(size_, n)*sizeof(T), hipMemcpyDeviceToDevice, stream_));
            HIP_CHECK(hipFreeAsync(data_, stream_));
        }
        data_ = nd; size_ = n;
    }
private:
    size_t size_ = 0;
    T* data_ = nullptr;
    hipStream_t stream_ = nullptr;
};

int main(int argc, char** argv) {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    const int n_iters = (argc > 1) ? std::atoi(argv[1]) : 100;
    for (int i = 0; i < n_iters; ++i) {
        MiniAsyncVec<double> a(100, stream);
        MiniAsyncVec<int> b(50, stream);
        MiniAsyncVec<short> c(20, stream);
        MiniAsyncVec<unsigned char> d(10, stream);
        a.zero(); b.zero(); c.zero(); d.zero();
        a.resize(200); b.resize(0); b.resize(75);
        // also a vec with default (null) stream — common pattern in nvMolKit
        MiniAsyncVec<double> def(50, nullptr);
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamDestroy(stream));
    std::printf("OK after %d iterations\n", n_iters);
    return 0;
}
