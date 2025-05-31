#pragma once

#include <any>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace deepsearch {
namespace core {

// 基础配置类
class BaseConfig {
 public:
  virtual ~BaseConfig() = default;
  virtual std::string to_string() const = 0;
  virtual void from_string(const std::string& str) = 0;
};

// HNSW配置
struct HNSWConfig : public BaseConfig {
  size_t M = 16;
  size_t ef_construction = 200;
  size_t max_elements = 1000000;
  bool allow_replace_deleted = false;
  size_t random_seed = 100;

  std::string to_string() const override;
  void from_string(const std::string& str) override;
};

// 搜索配置
struct SearchConfig : public BaseConfig {
  size_t ef = 50;
  size_t num_threads = 1;
  bool use_prefetch = true;
  size_t batch_size = 1000;

  std::string to_string() const override;
  void from_string(const std::string& str) override;
};

// 量化配置
struct QuantizationConfig : public BaseConfig {
  size_t nbits = 8;
  size_t subvector_size = 8;
  size_t num_centroids = 256;

  std::string to_string() const override;
  void from_string(const std::string& str) override;
};

// 配置管理器
class ConfigManager {
 public:
  static ConfigManager& instance();

  // 通用配置注册和获取
  template <typename ConfigType>
  void register_config(const std::string& name,
                       std::shared_ptr<ConfigType> config) {
    configs_[name] = config;
  }

  template <typename ConfigType>
  std::shared_ptr<ConfigType> get_config(const std::string& name) const {
    auto it = configs_.find(name);
    if (it != configs_.end()) {
      return std::static_pointer_cast<ConfigType>(it->second);
    }
    return nullptr;
  }

  // 文件I/O
  void load_from_file(const std::string& path);
  void save_to_file(const std::string& path) const;

  // 便捷的配置设置方法
  void set_hnsw_config(const HNSWConfig& config);
  void set_search_config(const SearchConfig& config);
  void set_quantization_config(const QuantizationConfig& config);

  // 便捷的配置获取方法
  HNSWConfig get_hnsw_config() const;
  SearchConfig get_search_config() const;
  QuantizationConfig get_quantization_config() const;

  // 配置管理
  void reset_to_defaults();
  bool has_config(const std::string& name) const;
  std::vector<std::string> get_config_names() const;

 private:
  ConfigManager() = default;
  std::unordered_map<std::string, std::shared_ptr<BaseConfig>> configs_;
};

}  // namespace core
}  // namespace deepsearch
