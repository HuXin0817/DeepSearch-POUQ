#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "hnsw/builder.h"
#include "hnsw/hnsw.h"
#include "searcher.h"

namespace py = pybind11;

// -----------------------------------------------------------------------------
// 1. Generic buffer adapter: supports 1D or 2D arrays of any scalar type T
// -----------------------------------------------------------------------------
template <typename T>
struct NdArray {
  T* ptr;
  ssize_t rows;
  ssize_t cols;
};

template <typename T>
NdArray<T> to_buffer(py::object obj) {
  auto arr = py::array_t < T, py::array::c_style | py::array::forcecast > (obj);
  auto buf = arr.request();

  if (buf.ndim < 1 || buf.ndim > 2)
    throw py::buffer_error("Expected 1D or 2D array, got " +
                           std::to_string(buf.ndim) + "D");

  ssize_t rows = (buf.ndim == 2 ? buf.shape[0] : 1);
  ssize_t cols = (buf.ndim == 2 ? buf.shape[1] : buf.shape[0]);
  return {static_cast<T*>(buf.ptr), rows, cols};
}

// -----------------------------------------------------------------------------
// 2. Parallel-for utility: uses OpenMP if available, otherwise falls back to
// serial
// -----------------------------------------------------------------------------
template <typename Func>
void parallel_for(size_t n, int num_threads, Func f) {
#ifdef _OPENMP
  int original = omp_get_max_threads();
  if (num_threads > 0) omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < n; ++i) {
    f(i);
  }
  omp_set_num_threads(original);
#else
  for (size_t i = 0; i < n; ++i) {
    f(i);
  }
#endif
}

// -----------------------------------------------------------------------------
// 3. Graph wrapper
// -----------------------------------------------------------------------------
struct Graph {
  deepsearch::Graph<int> graph;

  Graph() = default;
  explicit Graph(const deepsearch::Graph<int>& graph) : graph(graph) {}
  explicit Graph(const std::string& filename) { graph.load(filename); }

  void save(const std::string& filename) { graph.save(filename); }
  void load(const std::string& filename) { graph.load(filename); }
};

// -----------------------------------------------------------------------------
// 4. Index wrapper
// -----------------------------------------------------------------------------
struct Index {
  std::unique_ptr<deepsearch::Builder> idx;

  Index(const std::string& type, int dim, const std::string& metric, int R = 32,
        int L = 200) {
    if (dim <= 0) throw py::value_error("`dim` must be positive");
    if (R <= 0) throw py::value_error("`R` must be positive");
    if (L < 0) throw py::value_error("`L` must be non-negative");

    if (type == "HNSW") {
      idx = std::make_unique<deepsearch::Hnsw>(dim, metric, R, L);
    } else {
      throw py::value_error("Unknown index type: " + type);
    }
  }

  Graph build(py::object data) {
    auto buf = to_buffer<float>(data);
    if (buf.cols != idx->Dim())
      throw py::value_error("Dimension mismatch: expected " +
                            std::to_string(idx->Dim()) + ", got " +
                            std::to_string(buf.cols));
    idx->Build(buf.ptr, buf.rows);
    return Graph(idx->GetGraph());
  }
};

// -----------------------------------------------------------------------------
// 5. Searcher wrapper
// -----------------------------------------------------------------------------
struct Searcher {
  std::unique_ptr<deepsearch::SearcherBase> sr;
  ssize_t dim_;

  Searcher(const Graph& graph, py::object data, const std::string& metric,
           int level) {
    auto buf = to_buffer<float>(data);
    dim_ = buf.cols;
    sr = deepsearch::create_searcher(graph.graph, metric, level);
    sr->SetData(buf.ptr, buf.rows, buf.cols);
  }

  py::array_t<int> search(py::object query, int k) {
    auto buf = to_buffer<float>(query);
    if (buf.rows != 1 || buf.cols != dim_)
      throw py::value_error("Query must be shape (1, " + std::to_string(dim_) +
                            ")");

    int* ids = new int[k];
    sr->Search(buf.ptr, k, ids);

    py::capsule free_when_done(ids,
                               [](void* f) { delete[] static_cast<int*>(f); });

    return py::array_t<int>({k},            // shape
                            {sizeof(int)},  // strides
                            ids,            // pointer
                            free_when_done  // capsule
    );
  }

  py::array_t<int> batch_search(py::object query, int k, int num_threads = 0) {
    auto buf = to_buffer<float>(query);
    if (buf.cols != dim_)
      throw py::value_error("Batch query dimension mismatch");

    size_t nq = buf.rows;
    int* ids = new int[nq * k];

    parallel_for(nq, num_threads, [&](size_t i) {
      sr->Search(buf.ptr + i * dim_, k, ids + i * k);
    });

    py::capsule free_when_done(ids,
                               [](void* f) { delete[] static_cast<int*>(f); });

    // 返回二维数组，Python 侧析构时自动调用 capsule
    return py::array_t<int>({(ssize_t)nq, (ssize_t)k},    // shape
                            {(ssize_t)(k * sizeof(int)),  // row stride
                             (ssize_t)(sizeof(int))},     // col stride
                            ids,                          // data ptr
                            free_when_done                // capsule
    );
  }

  void set_ef(int ef) {
    if (ef <= 0) throw py::value_error("`ef` must be positive");
    sr->SetEf(ef);
  }

  void optimize(int num_threads = 0) {
    // Use parallel_for with a single iteration to adjust threads
    parallel_for(1, num_threads, [&](size_t) { sr->Optimize(num_threads); });
  }
};

// -----------------------------------------------------------------------------
// 6. Module definition
// -----------------------------------------------------------------------------
PYBIND11_MODULE(deepsearch, m) {
  m.doc() = "DeepSearch Python bindings";

#ifdef _OPENMP
  m.def(
      "set_num_threads", [](int n) { omp_set_num_threads(n); },
      "Set global OpenMP thread count", py::arg("num_threads"));
#else
  m.def(
      "set_num_threads",
      [](int) { py::print("OpenMP not available; call ignored"); },
      "Dummy set_num_threads when OpenMP is disabled", py::arg("num_threads"));
#endif

  py::class_<Graph>(m, "Graph")
      .def(py::init<>())
      .def(py::init<const std::string&>(), py::arg("filename"))
      .def("save", &Graph::save, py::arg("filename"))
      .def("load", &Graph::load, py::arg("filename"));

  py::class_<Index>(m, "Index")
      .def(py::init<const std::string&, int, const std::string&, int, int>(),
           py::arg("type"), py::arg("dim"), py::arg("metric"),
           py::arg("R") = 32, py::arg("L") = 200)
      .def("build", &Index::build, py::arg("data"),
           "Build the index from a float array");

  py::class_<Searcher>(m, "Searcher")
      .def(py::init<const Graph&, py::object, const std::string&, int>(),
           py::arg("graph"), py::arg("data"), py::arg("metric"),
           py::arg("level"))
      .def("search", &Searcher::search, py::arg("query"), py::arg("k"),
           "Search a single vector")
      .def("batch_search", &Searcher::batch_search, py::arg("query"),
           py::arg("k"), py::arg("num_threads") = 0,
           "Search multiple vectors in parallel")
      .def("set_ef", &Searcher::set_ef, py::arg("ef"),
           "Set the `ef` parameter for search")
      .def("optimize", &Searcher::optimize, py::arg("num_threads") = 0,
           "Optimize the searcher's prefetch settings");
}
