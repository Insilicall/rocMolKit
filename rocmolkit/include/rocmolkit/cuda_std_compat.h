// rocMolKit — shim for NVIDIA libcudacxx (<cuda/std/*>) headers.
// libcudacxx is NVIDIA's CUDA-aware port of the C++ standard library.
// On AMD/ROCm we just use the host C++20 std::, which works inside HIP kernels
// since clang's HIP compiler shares the same standard library.

#pragma once

#include <span>
#include <tuple>
#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <optional>
#include <variant>
#include <numeric>
#include <algorithm>

// Make code that writes `cuda::std::span<T>` resolve to `std::span<T>`.
namespace cuda {
namespace std {
    using ::std::span;
    using ::std::tuple;
    using ::std::array;
    using ::std::byte;
    using ::std::nullptr_t;
    using ::std::size_t;
    using ::std::ptrdiff_t;

    using ::std::pair;
    using ::std::make_pair;
    using ::std::make_tuple;
    using ::std::tuple_size;
    using ::std::tuple_element;
    using ::std::tuple_element_t;
    using ::std::get;
    using ::std::tie;

    using ::std::optional;
    using ::std::variant;
    using ::std::nullopt;
    using ::std::nullopt_t;

    using ::std::is_same;
    using ::std::is_same_v;
    using ::std::enable_if;
    using ::std::enable_if_t;
    using ::std::remove_const;
    using ::std::remove_const_t;
    using ::std::remove_reference;
    using ::std::remove_reference_t;

    using ::std::move;
    using ::std::forward;

    // Math
    using ::std::abs;
    using ::std::min;
    using ::std::max;
    using ::std::clamp;
    using ::std::sqrt;
    using ::std::pow;
    using ::std::numeric_limits;

    inline constexpr auto dynamic_extent = ::std::dynamic_extent;
}  // namespace std
}  // namespace cuda
