#pragma once

#include <exception>
#include <string>

namespace deepsearch {
namespace core {

// 基础异常类
class DeepSearchException : public std::exception {
 protected:
  std::string message_;
  std::string file_;
  int line_;

 public:
  DeepSearchException(const std::string& msg, const std::string& file = "",
                      int line = 0)
      : message_(msg), file_(file), line_(line) {}

  const char* what() const noexcept override { return message_.c_str(); }

  const std::string& message() const { return message_; }
  const std::string& file() const { return file_; }
  int line() const { return line_; }
};

// 参数异常
class InvalidParameterException : public DeepSearchException {
 public:
  explicit InvalidParameterException(const std::string& param_name,
                                     const std::string& file = "", int line = 0)
      : DeepSearchException("Invalid parameter: " + param_name, file, line) {}
};

// 文件IO异常
class FileIOException : public DeepSearchException {
 public:
  explicit FileIOException(const std::string& filename,
                           const std::string& file = "", int line = 0)
      : DeepSearchException("File I/O error: " + filename, file, line) {}
};

// 内存异常
class MemoryException : public DeepSearchException {
 public:
  explicit MemoryException(const std::string& operation,
                           const std::string& file = "", int line = 0)
      : DeepSearchException("Memory error in: " + operation, file, line) {}
};

// 索引异常
class IndexException : public DeepSearchException {
 public:
  explicit IndexException(const std::string& msg, const std::string& file = "",
                          int line = 0)
      : DeepSearchException("Index error: " + msg, file, line) {}
};

// 便利宏
#define THROW_INVALID_PARAM(param) \
  throw InvalidParameterException(param, __FILE__, __LINE__)

#define THROW_FILE_IO_ERROR(filename) \
  throw FileIOException(filename, __FILE__, __LINE__)

#define THROW_MEMORY_ERROR(operation) \
  throw MemoryException(operation, __FILE__, __LINE__)

#define THROW_INDEX_ERROR(msg) throw IndexException(msg, __FILE__, __LINE__)

}  // namespace core
}  // namespace deepsearch
