#pragma once

#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "exceptions.h"

namespace deepsearch {
namespace core {

// 对齐内存分配器
template <typename T>
class AlignedAllocator {
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U>;
  };

  AlignedAllocator(size_t alignment = 64) : alignment_(alignment) {}

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U>& other)
      : alignment_(other.alignment_) {}

  pointer allocate(size_type n) {
    size_t size = n * sizeof(T);
    size_t aligned_size = (size + alignment_ - 1) & ~(alignment_ - 1);

    void* ptr = std::aligned_alloc(alignment_, aligned_size);
    if (!ptr) {
      THROW_MEMORY_ERROR("aligned_alloc failed");
    }

    return static_cast<pointer>(ptr);
  }

  void deallocate(pointer ptr, size_type) { std::free(ptr); }

  bool operator==(const AlignedAllocator& other) const {
    return alignment_ == other.alignment_;
  }

  bool operator!=(const AlignedAllocator& other) const {
    return !(*this == other);
  }

 private:
  size_t alignment_;

  template <typename U>
  friend class AlignedAllocator;
};

// RAII对齐缓冲区
template <typename T>
class AlignedBuffer {
 private:
  T* data_;
  size_t size_;
  size_t alignment_;

 public:
  AlignedBuffer(size_t count, size_t alignment = 64)
      : size_(count), alignment_(alignment) {
    size_t total_size = count * sizeof(T);
    size_t aligned_size = (total_size + alignment - 1) & ~(alignment - 1);

    data_ = static_cast<T*>(std::aligned_alloc(alignment, aligned_size));
    if (!data_) {
      THROW_MEMORY_ERROR("AlignedBuffer allocation failed");
    }

    // 初始化为零
    std::memset(data_, 0, aligned_size);
  }

  ~AlignedBuffer() {
    if (data_) {
      std::free(data_);
    }
  }

  // 禁止拷贝
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  // 允许移动
  AlignedBuffer(AlignedBuffer&& other) noexcept
      : data_(other.data_), size_(other.size_), alignment_(other.alignment_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
    if (this != &other) {
      if (data_) {
        std::free(data_);
      }
      data_ = other.data_;
      size_ = other.size_;
      alignment_ = other.alignment_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  T* data() { return data_; }
  const T* data() const { return data_; }
  size_t size() const { return size_; }
  size_t alignment() const { return alignment_; }

  T& operator[](size_t index) { return data_[index]; }
  const T& operator[](size_t index) const { return data_[index]; }

  T* begin() { return data_; }
  const T* begin() const { return data_; }
  T* end() { return data_ + size_; }
  const T* end() const { return data_ + size_; }
};

// 内存池（简化版本）
template <typename T>
class MemoryPool {
 private:
  struct Block {
    alignas(T) char data[sizeof(T)];
    Block* next;
  };

  Block* free_list_;
  std::vector<std::unique_ptr<Block[]>> chunks_;
  size_t chunk_size_;

 public:
  explicit MemoryPool(size_t chunk_size = 1024)
      : free_list_(nullptr), chunk_size_(chunk_size) {
    allocate_chunk();
  }

  ~MemoryPool() = default;

  T* allocate() {
    if (!free_list_) {
      allocate_chunk();
    }

    Block* block = free_list_;
    free_list_ = free_list_->next;
    return reinterpret_cast<T*>(block);
  }

  void deallocate(T* ptr) {
    Block* block = reinterpret_cast<Block*>(ptr);
    block->next = free_list_;
    free_list_ = block;
  }

 private:
  void allocate_chunk() {
    auto chunk = std::make_unique<Block[]>(chunk_size_);
    Block* chunk_ptr = chunk.get();

    // 将新块链接到自由列表
    for (size_t i = 0; i < chunk_size_ - 1; ++i) {
      chunk_ptr[i].next = &chunk_ptr[i + 1];
    }
    chunk_ptr[chunk_size_ - 1].next = free_list_;
    free_list_ = chunk_ptr;

    chunks_.push_back(std::move(chunk));
  }
};

}  // namespace core
}  // namespace deepsearch
