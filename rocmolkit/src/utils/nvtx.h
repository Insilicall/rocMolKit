// rocMolKit — NVTX shim (NO-OP).
// Original nvMolKit uses NVIDIA Tools Extension for profiling annotations.
// AMD equivalent is rocTX (roctracer/roctx.h). For now we ship a no-op shim
// to keep API surface compatible without pulling roctracer at link time.
//
// To enable rocTX profiling later: link roctx64 and replace these no-ops with
//   roctxRangePushA(name) / roctxRangePop() / roctxNameOsThread(...).

#ifndef NVMOLKIT_UTILS_NVTX_H
#define NVMOLKIT_UTILS_NVTX_H

#include <string>
#include <cstdint>

#include <hip/hip_runtime.h>

namespace nvMolKit {

namespace NvtxColor {
constexpr uint32_t kGrey   = 0xFF808080;
constexpr uint32_t kRed    = 0xFFFF0000;
constexpr uint32_t kGreen  = 0xFF00FF00;
constexpr uint32_t kBlue   = 0xFF0000FF;
constexpr uint32_t kYellow = 0xFFFFFF00;
constexpr uint32_t kCyan   = 0xFF00FFFF;
constexpr uint32_t kOrange = 0xFFFFA500;
}  // namespace NvtxColor

class ScopedNvtxRange {
 public:
  explicit ScopedNvtxRange(const std::string& /*name*/, uint32_t /*color*/ = NvtxColor::kGrey) noexcept {}
  explicit ScopedNvtxRange(const char* /*name*/, uint32_t /*color*/ = NvtxColor::kGrey) noexcept {}
  ScopedNvtxRange(const ScopedNvtxRange&)            = delete;
  ScopedNvtxRange& operator=(const ScopedNvtxRange&) = delete;
  ScopedNvtxRange(ScopedNvtxRange&&)                 = delete;
  ScopedNvtxRange& operator=(ScopedNvtxRange&&)      = delete;
  void             pop() noexcept {}
  ~ScopedNvtxRange() noexcept = default;
};

}  // namespace nvMolKit

// Free-function API used by upstream code (e.g. utils/device.cpp).
inline void nvtxNameCudaStreamA(hipStream_t /*stream*/, const char* /*name*/) noexcept {}

#endif  // NVMOLKIT_UTILS_NVTX_H
