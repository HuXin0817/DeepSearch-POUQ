#pragma once

#ifdef __linux__
#include <sys/mman.h>
#endif

#include <cstdlib>
#include <cstring>

namespace deepsearch {

template <typename T>
struct align_alloc {
  T* ptr = nullptr;
  using value_type = T;

  T* allocate(int n) {
    if (n <= 1 << 14) {
      int sz = (n * sizeof(T) + 63) >> 6 << 6;
      return ptr = (T*)std::aligned_alloc(64, sz);
    }
    int sz = (n * sizeof(T) + (1 << 21) - 1) >> 21 << 21;
    ptr = (T*)std::aligned_alloc(1 << 21, sz);
#ifdef __linux__
    madvise(ptr, sz, MADV_HUGEPAGE);
#endif
    return ptr;
  }

  void deallocate(T*, int) { free(ptr); }

  template <typename U>
  struct rebind {
    typedef align_alloc<U> other;
  };

  bool operator!=(const align_alloc& rhs) { return ptr != rhs.ptr; }
};

inline void* alloc2M(size_t nbytes) {
  size_t len = (nbytes + (1 << 21) - 1) >> 21 << 21;
  auto p = std::aligned_alloc(1 << 21, len);
  std::memset(p, 0, len);
  return p;
}

inline void* alloc64B(size_t nbytes) {
  size_t len = (nbytes + (1 << 6) - 1) >> 6 << 6;
  auto p = std::aligned_alloc(1 << 6, len);
  std::memset(p, 0, len);
  return p;
}

inline constexpr int64_t do_align(int64_t x, int64_t align) {
  return (x + align - 1) / align * align;
}

}  // namespace deepsearch
