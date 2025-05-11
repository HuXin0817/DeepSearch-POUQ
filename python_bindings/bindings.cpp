#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <thread>

// 条件包含OpenMP头文件
#ifdef _OPENMP
#include <omp.h>
#endif

#include "hnsw/builder.h"
#include "hnsw/hnsw.h"
#include "searcher.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace {

// 统一的异常处理宏
#define THROW_IF_NOT(cond, msg) \
  if (!(cond)) throw std::invalid_argument(msg)

// 增强的数组形状验证函数
inline void validate_input_array(const py::buffer_info& buffer) {
  THROW_IF_NOT(buffer.ndim == 1 || buffer.ndim == 2,
               "Input must be a 1D or 2D array. Got " +
                   std::to_string(buffer.ndim) + "D array");
}

// 统一的数组数据提取函数
std::tuple<size_t, size_t, float*> get_array_data(py::object obj) {
  py::array_t<float, py::array::c_style | py::array::forcecast> arr(obj);
  auto buf = arr.request();
  validate_input_array(buf);

  size_t rows = buf.ndim == 2 ? buf.shape[0] : 1;
  size_t dim = buf.ndim == 2 ? buf.shape[1] : buf.shape[0];
  return {rows, dim, static_cast<float*>(buf.ptr)};
}

}  // namespace

// 条件编译OpenMP函数
#ifdef _OPENMP
void set_num_threads(int num_threads) { omp_set_num_threads(num_threads); }
#else
void set_num_threads(int) {
  // 无操作
}
#endif

// 使用PIMPL模式隐藏实现细节
class Graph {
 private:
  std::unique_ptr<deepsearch::Graph<int>> graph_;

 public:
  Graph() : graph_(std::make_unique<deepsearch::Graph<int>>()) {}

  explicit Graph(const std::string& filename)
      : graph_(std::make_unique<deepsearch::Graph<int>>()) {
    graph_->load(filename);
  }

  void save(const std::string& filename) const { graph_->save(filename); }

  const deepsearch::Graph<int>& impl() const { return *graph_; }
};

class Index {
  std::unique_ptr<deepsearch::Builder> index_;

 public:
  Index(const std::string& index_type, int dim, const std::string& metric,
        int R = 32, int L = 200) {
    THROW_IF_NOT(dim > 0, "Dimension must be positive");
    THROW_IF_NOT(R > 0, "R parameter must be positive");
    THROW_IF_NOT(L >= 0, "L parameter must be non-negative");

    if (index_type == "HNSW") {
      index_ = std::make_unique<deepsearch::Hnsw>(dim, metric, R, L);
    } else {
      throw std::invalid_argument("Unsupported index type: " + index_type);
    }
  }

  Graph build(py::object input) {
    auto [n, dim, data] = get_array_data(input);
    THROW_IF_NOT(dim == index_->Dim(), "Input dimension mismatch. Expected " +
                                           std::to_string(index_->Dim()) +
                                           ", got " + std::to_string(dim));
    index_->Build(data, n);
    return Graph();
  }
};

class Searcher {
  std::unique_ptr<deepsearch::SearcherBase> searcher_;
  size_t data_dim_;

 public:
  Searcher(const Graph& graph, py::object data, const std::string& metric,
           int level)
      : searcher_(deepsearch::create_searcher(graph.impl(), metric, level)) {
    auto [n, dim, ptr] = get_array_data(data);
    data_dim_ = dim;
    searcher_->SetData(ptr, n, dim);
  }

  py::array_t<int> search(py::object query, int k) {
    py::array_t<float, py::array::c_style | py::array::forcecast> items(query);
    int *ids;
    ids = new int[k];
    searcher_->Search(items.data(0), k, ids);
    py::capsule free_when_done(ids, [](void *f) { delete[] f; });
    return py::array_t<int>({k}, {sizeof(int)}, ids, free_when_done);

//    auto [nq, dim, qdata] = get_array_data(query);
//    THROW_IF_NOT(dim == data_dim_, "Query dimension mismatch. Expected " +
//                                       std::to_string(data_dim_));
//
//    auto* ids = new int[nq * k];  // 使用原生数组避免unique_ptr的释放问题
//    py::capsule free_when_done(ids,
//                               [](void* f) { delete[] static_cast<int*>(f); });
//
//    searcher_->Search(qdata, k, ids);
//
//    return py::array_t<int>(k, ids);
  }

  py::array_t<int> batch_search(py::object query, int k, int num_threads = 0) {
    auto query_data = get_array_data(query);
    auto nq = std::get<0>(query_data);      // 或根据实际返回类型调整
    auto dim = std::get<1>(query_data);
    auto qdata = std::get<2>(query_data);   // 假设 qdata 是 float* 类型

    THROW_IF_NOT(dim == data_dim_, "Query dimension mismatch. Expected " +
                                       std::to_string(data_dim_));

    auto* ids = new int[nq * k];  // 使用原生数组避免unique_ptr的释放问题
    py::capsule free_when_done(ids,
                               [](void* f) { delete[] static_cast<int*>(f); });

    // OpenMP并行区域
#ifdef _OPENMP
    {
      py::gil_scoped_release release;
      const auto original_threads = omp_get_max_threads();
      if (num_threads > 0) omp_set_num_threads(num_threads);

      try {
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < nq; ++i) {
          searcher_->Search(qdata + i * dim, k, ids + i * k);
        }
      } catch (...) {
        omp_set_num_threads(original_threads);
        throw;
      }
      omp_set_num_threads(original_threads);
    }
#else
    // 顺序执行
    for (size_t i = 0; i < nq; ++i) {
      searcher_->Search(qdata + i * dim, k, ids + i * k);
    }
#endif

    // 返回二维数组
    return py::array_t<int>(
        {static_cast<ssize_t>(nq), static_cast<ssize_t>(k)},  // Shape
        {static_cast<ssize_t>(k * sizeof(int)),               // Strides (row)
         static_cast<ssize_t>(sizeof(int))},                  // Strides (col)
        ids,                                                  // 数据指针
        free_when_done                                        // 内存管理
    );
  }

  void set_ef(int ef) {
    THROW_IF_NOT(ef > 0, "ef must be positive");
    searcher_->SetEf(ef);
  }

  void optimize(int num_threads = 0) {
#ifdef _OPENMP
    const auto original_threads = omp_get_max_threads();
    if (num_threads > 0) omp_set_num_threads(num_threads);
#endif

    try {
      searcher_->Optimize(num_threads);
    } catch (...) {
#ifdef _OPENMP
      omp_set_num_threads(original_threads);
#endif
      throw;
    }

#ifdef _OPENMP
    omp_set_num_threads(original_threads);
#endif
  }
};

PYBIND11_MODULE(deepsearch, m) {
  m.doc() = "DeepSearch Python bindings";

#ifdef _OPENMP
  m.def("set_num_threads", &set_num_threads, py::arg("num_threads"),
        "Set global OpenMP thread count");
#else
  m.def(
      "set_num_threads",
      [](int) {
        py::print("Warning: OpenMP support not available, threading disabled");
      },
      py::arg("num_threads"),
      "Thread setting unavailable (compiled without OpenMP support)");
#endif

  py::class_<Graph>(m, "Graph")
      .def(py::init<>())
      .def(py::init<const std::string&>(), py::arg("filename"))
      .def("save", &Graph::save, py::arg("filename"), "Save graph to file")
      .def(
          "load", [](Graph& g, const std::string& f) { g = Graph(f); },
          py::arg("filename"), "Load graph from file");

  py::class_<Index>(m, "Index")
      .def(py::init<const std::string&, int, const std::string&, int, int>(),
           py::arg("index_type"), py::arg("dim"), py::arg("metric"),
           py::arg("R") = 32, py::arg("L") = 200,
           "Initialize index\n"
           "Args:\n"
           "  index_type: NSG or HNSW\n"
           "  dim: data dimension\n"
           "  metric: distance metric\n"
           "  R: graph degree\n"
           "  L: construction complexity")
      .def("build", &Index::build, py::arg("data"), "Build index from data");

  py::class_<Searcher>(m, "Searcher")
      .def(py::init<const Graph&, py::object, const std::string&, int>(),
           py::arg("graph"), py::arg("data"), py::arg("metric"),
           py::arg("level"),
           "Initialize searcher\n"
           "Args:\n"
           "  graph: prebuilt graph\n"
           "  data: original data vectors\n"
           "  metric: distance metric\n"
           "  level: search level")
      .def("set_ef", &Searcher::set_ef, py::arg("ef"),
           "Set search ef parameter")
      .def("search", &Searcher::search, py::arg("query"), py::arg("k"),
           "Single query search")
      .def("batch_search", &Searcher::batch_search, py::arg("query"),
           py::arg("k"), py::arg("num_threads") = 0,
           "Batch search with threading")
      .def("optimize", &Searcher::optimize, py::arg("num_threads") = 0,
           "Optimize search structure");
}
