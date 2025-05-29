#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "hnsw/hnsw.h"
#include "searcher.h"

template <typename T>
void load_fvecs(const char* filename, T*& p, int64_t& n, int64_t& dim) {
  std::ifstream fs(filename, std::ios::binary);
  if (!fs.is_open()) {
    throw std::runtime_error("Cannot open file: " +
                             std::filesystem::path(filename).string());
  }

  // 1. 读入 single-vector 的维度（4 字节 little-endian）
  int32_t dim32;
  fs.read(reinterpret_cast<char*>(&dim32), sizeof(dim32));
  dim = dim32;

  // 2. 计算文件总字节数，进而计算向量个数 n
  fs.seekg(0, std::ios::end);
  std::streamoff total_bytes = fs.tellg();
  // 每条记录占 4 字节头 + dim * sizeof(T)
  const std::size_t rec_size = sizeof(dim32) + dim * sizeof(T);
  if (total_bytes % rec_size != 0) {
    throw std::runtime_error("Corrupted .fvecs file: unexpected file size");
  }
  n = total_bytes / rec_size;

  std::cout << "Read path: " << filename << ", n = " << n << ", dim = " << dim
            << std::endl;

  // 3. 对齐分配：posix_memalign 要求 alignment 为 2 的幂，且 pointer 可 free()
  void* raw_ptr = nullptr;
  int err = posix_memalign(&raw_ptr, 64, n * dim * sizeof(T));
  if (err != 0 || raw_ptr == nullptr) {
    throw std::bad_alloc();
  }
  p = reinterpret_cast<T*>(raw_ptr);

  // 4. 回到文件开头，逐条读取数据
  fs.seekg(sizeof(dim32), std::ios::beg);
  for (int64_t i = 0; i < n; ++i) {
    // 4.1 跳过 4 字节的头部（已经在第一次循环前做过一次 seek）
    //     这里不再每次都调用 seekg，只是在循环首执行一次
    fs.read(reinterpret_cast<char*>(p + i * dim), dim * sizeof(T));
  }

  fs.close();
}

int main(int argc, char** argv) {
  if (argc < 8) {
    printf(
        "Usage: ./main base_path query_path gt_path graph_path level "
        "topk search_ef num_threads\n");
    exit(-1);
  }
  std::string base_path = argv[1];
  std::string query_path = argv[2];
  std::string gt_path = argv[3];
  std::string graph_path = argv[4];
  int level = std::stoi(argv[5]);
  int topk = std::stoi(argv[6]);
  int search_ef = std::stoi(argv[7]);
  int num_threads = 1;
  int iters = 10;
  if (argc >= 9) {
    num_threads = std::stoi(argv[8]);
  }
  if (argc >= 10) {
    iters = std::stoi(argv[9]);
  }
  float *base, *query;
  int* gt;
  int64_t N, dim, nq, gt_k;
  load_fvecs(base_path.c_str(), base, N, dim);
  load_fvecs(query_path.c_str(), query, nq, dim);
  load_fvecs(gt_path.c_str(), gt, nq, gt_k);
  if (!std::filesystem::exists(graph_path)) {
    deepsearch::Hnsw hnsw(dim, "L2", 32, 200);
    hnsw.Build(base, N);
    hnsw.final_graph.save(graph_path);
  }
  deepsearch::Graph<int> graph;
  graph.load(graph_path);
  auto searcher = deepsearch::create_searcher(graph, "L2", level);
  searcher->SetData(base, N, dim);
  searcher->Optimize(num_threads);
  searcher->SetEf(search_ef);
  double recall;
  double best_qps = 0.0;
  for (int iter = 1; iter <= iters; ++iter) {
    printf("iter : [%d/%d]\n", iter, iters);
    std::vector<int> pred(nq * topk);
    auto st = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (int i = 0; i < nq; ++i) {
      searcher->Search(query + i * dim, topk, pred.data() + i * topk);
    }
    auto ed = std::chrono::high_resolution_clock::now();
    auto ela = std::chrono::duration<double>(ed - st).count();
    double qps = nq / ela;
    best_qps = std::max(qps, best_qps);
    int cnt = 0;
    for (int i = 0; i < nq; ++i) {
      std::unordered_set<int> st(gt + i * gt_k, gt + i * gt_k + topk);
      for (int j = 0; j < topk; ++j) {
        if (st.count(pred[i * topk + j])) {
          cnt++;
        }
      }
    }
    recall = (double)cnt / nq / topk;
    printf("\tRecall@%d = %.4lf, QPS = %.2lf\n", topk, recall, qps);
  }
  printf("Best QPS = %.2lf\n", best_qps);
  free(base);
  free(query);
  free(gt);
}
