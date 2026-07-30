#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cctype>
#include <limits>
#include <mutex>
#include <thread>
#include <atomic>
#include <memory>
#include <queue>

#include <sys/stat.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/tcp.h>

#include <hnswlib/hnswlib.h>
#include "data_paths.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

static bool file_exists(const std::string& path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

static double elapsed_sec(std::chrono::high_resolution_clock::time_point t0) {
    const auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count() / 1000.0;
}

static void print_progress(
    const std::string& stage,
    size_t done,
    size_t total,
    std::chrono::high_resolution_clock::time_point t0
) {
    const double percent = total > 0
        ? 100.0 * static_cast<double>(done) / static_cast<double>(total)
        : 100.0;

    std::cout << "[" << stage << "] "
              << done << " / " << total
              << " (" << std::fixed << std::setprecision(2) << percent << "%)"
              << ", elapsed=" << std::fixed << std::setprecision(1)
              << elapsed_sec(t0) << "s"
              << std::endl;
}

static std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return s;
}

enum class DatasetKind {
    Trip,
    MsMarco,
    Sift,
    Laion,
};

static DatasetKind parse_dataset_kind(const std::string& name) {
    const std::string s = to_lower_copy(name);
    if (s == "trip" || s == "tripclick" || s == "trip_click") return DatasetKind::Trip;
    if (s == "msmarco" || s == "ms_marco" || s == "marco" || s == "mscro") return DatasetKind::MsMarco;
    if (s == "sift") return DatasetKind::Sift;
    if (s == "laion" || s == "laion1m") return DatasetKind::Laion;
    throw std::runtime_error("Unsupported --dataset value: " + name + " (use trip, msmarco, sift, or laion)");
}

static std::string dataset_kind_name(DatasetKind dataset) {
    switch (dataset) {
        case DatasetKind::Trip: return "trip";
        case DatasetKind::MsMarco: return "msmarco";
        case DatasetKind::Sift: return "sift";
        case DatasetKind::Laion: return "laion";
    }
    return "unknown";
}

enum class MetricKind {
    L2,
    InnerProduct,
};

static MetricKind metric_for_dataset(DatasetKind dataset) {

    if (dataset == DatasetKind::Sift) return MetricKind::L2;
    return MetricKind::InnerProduct;
}

static std::string metric_name(MetricKind metric) {
    return metric == MetricKind::L2 ? "L2" : "INNER_PRODUCT";
}

struct Paths {
    std::string data_root;
    std::string dataset_name;
    std::string base_fvecs;
    std::string query_fvecs;
};

static Paths make_paths(DatasetKind dataset) {
    Paths p;
    p.data_root = mess::find_data_root().string();
    p.dataset_name = dataset_kind_name(dataset);

    switch (dataset) {
        case DatasetKind::Trip:
            p.base_fvecs = mess::data_path("dataset/trip_distilbert/passages.fvecs");
            p.query_fvecs = mess::data_path("dataset/trip_distilbert/queries.fvecs");
            break;
        case DatasetKind::MsMarco:
            p.base_fvecs = mess::data_path("dataset/msmarco_bert/passages.fvecs");
            p.query_fvecs = mess::data_path("dataset/msmarco_bert/queries.fvecs");
            break;
        case DatasetKind::Sift:
            p.base_fvecs = mess::data_path("dataset/sift/base.fvecs");
            p.query_fvecs = mess::data_path("dataset/sift/query.fvecs");
            break;
        case DatasetKind::Laion:
            p.base_fvecs = mess::data_path("dataset/laion1m/100k/laion_base.fvecs");
            p.query_fvecs = mess::data_path("dataset/laion1m/laion_query.fvecs");
            break;
    }
    return p;
}

static void require_file(const std::string& path, const std::string& name) {
    if (!file_exists(path)) throw std::runtime_error("Cannot find " + name + ": " + path);
}

struct RuntimeOptions {
    DatasetKind dataset = DatasetKind::Sift;

    int M = 64;
    int ef_construction = 80;
    int ef = 20;
    int top_k = 10;
    int threads = 16;

    std::string bind_ip = "0.0.0.0";
    int port = 9080;

    std::string base_fvecs_override;
    bool log_each_request = false;
};

static RuntimeOptions parse_args(int argc, char** argv) {
    RuntimeOptions opt;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) {
            opt.dataset = parse_dataset_kind(argv[++i]);
        } else if (arg == "--M" && i + 1 < argc) {
            opt.M = std::stoi(argv[++i]);
        } else if ((arg == "--ef_construction" || arg == "--efConstruction") && i + 1 < argc) {
            opt.ef_construction = std::stoi(argv[++i]);
        } else if ((arg == "--ef" || arg == "--efSearch" || arg == "--ef_search") && i + 1 < argc) {
            opt.ef = std::stoi(argv[++i]);
        } else if ((arg == "--k" || arg == "--top_k") && i + 1 < argc) {
            opt.top_k = std::stoi(argv[++i]);
        } else if ((arg == "--threads" || arg == "--omp_threads") && i + 1 < argc) {
            opt.threads = std::stoi(argv[++i]);
        } else if (arg == "--bind_ip" && i + 1 < argc) {
            opt.bind_ip = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            opt.port = std::stoi(argv[++i]);
        } else if (arg == "--base_fvecs" && i + 1 < argc) {
            opt.base_fvecs_override = argv[++i];
        } else if (arg == "--log_each_request") {
            opt.log_each_request = true;
        }
    }

    if (opt.M <= 0) opt.M = 16;
    if (opt.ef_construction <= 0) opt.ef_construction = 500;
    if (opt.top_k <= 0) opt.top_k = 10;
    if (opt.ef <= 0) opt.ef = opt.top_k;
    opt.ef = std::max(opt.ef, opt.top_k);
    if (opt.threads <= 0) opt.threads = 1;
    return opt;
}

static void load_fvecs(
    const std::string& filename,
    std::vector<float>& data,
    size_t& num,
    int& dim,
    const std::string& name
) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Cannot open fvecs file: " + filename);

    in.read(reinterpret_cast<char*>(&dim), 4);
    if (!in) throw std::runtime_error("Failed to read fvecs dim: " + filename);

    in.seekg(0, std::ios::end);
    const size_t file_size = static_cast<size_t>(in.tellg());
    const size_t item_size = 4 + static_cast<size_t>(dim) * sizeof(float);
    if (item_size == 0 || file_size % item_size != 0) {
        throw std::runtime_error("Invalid fvecs file size: " + filename);
    }

    num = file_size / item_size;
    data.resize(num * static_cast<size_t>(dim));
    in.seekg(0, std::ios::beg);

    std::cout << "[Load] " << name << ": " << filename
              << ", num=" << num << ", dim=" << dim << std::endl;

    const auto t0 = std::chrono::high_resolution_clock::now();
    const size_t report_every = std::max<size_t>(1, num / 20);

    for (size_t i = 0; i < num; ++i) {
        int cur_dim = 0;
        in.read(reinterpret_cast<char*>(&cur_dim), 4);
        if (!in || cur_dim != dim) {
            std::ostringstream oss;
            oss << "Corrupted fvecs or dim mismatch: " << filename
                << ", row=" << i << ", expected=" << dim << ", got=" << cur_dim;
            throw std::runtime_error(oss.str());
        }
        in.read(reinterpret_cast<char*>(data.data() + i * static_cast<size_t>(dim)),
                static_cast<std::streamsize>(dim * sizeof(float)));
        if (!in) throw std::runtime_error("Failed while reading fvecs data: " + filename);

        if ((i + 1) % report_every == 0 || i + 1 == num) {
            print_progress("Load " + name, i + 1, num, t0);
        }
    }

    std::cout << "[Load] Finished " << name << ": " << num << " vectors, dim=" << dim << std::endl;
}

static constexpr uint32_t HNSWLIB_PROTOCOL_MAGIC = 0x484E5357;
static constexpr uint16_t HNSWLIB_PROTOCOL_VERSION = 1;

#pragma pack(push, 1)
struct QueryBatchRequestHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t flags;
    int32_t batch_size;
    int32_t dim;
    int32_t top_k;
    int32_t ef;
    uint64_t query_id_start;
    int32_t body_bytes;
};

struct QueryBatchResponseHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t flags;
    int32_t batch_size;
    int32_t top_k;
    int32_t ef_used;
    int32_t body_bytes;
};

struct QueryResultHeader {
    uint64_t query_id;
    uint64_t server_compute_us;
    int32_t result_count;
    int32_t body_bytes;
};

struct ResultItem {
    int64_t label;
    float distance;
};
#pragma pack(pop)

static bool recv_all(int fd, void* buf, size_t len) {
    char* p = reinterpret_cast<char*>(buf);
    size_t recvd = 0;
    while (recvd < len) {
        const ssize_t r = ::recv(fd, p + recvd, len - recvd, 0);
        if (r <= 0) return false;
        recvd += static_cast<size_t>(r);
    }
    return true;
}

static bool send_all(int fd, const void* buf, size_t len) {
    const char* p = reinterpret_cast<const char*>(buf);
    size_t sent = 0;
    while (sent < len) {
        const ssize_t s = ::send(fd, p + sent, len - sent, 0);
        if (s <= 0) return false;
        sent += static_cast<size_t>(s);
    }
    return true;
}

template <typename T>
static void append_pod(std::vector<uint8_t>& buf, const T& value) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&value);
    buf.insert(buf.end(), p, p + sizeof(T));
}

static std::unique_ptr<hnswlib::SpaceInterface<float>> make_space(MetricKind metric, int dim) {
    if (metric == MetricKind::L2) {
        return std::unique_ptr<hnswlib::SpaceInterface<float>>(new hnswlib::L2Space(dim));
    }
    return std::unique_ptr<hnswlib::SpaceInterface<float>>(new hnswlib::InnerProductSpace(dim));
}

static void build_hnsw_index_parallel(
    hnswlib::HierarchicalNSW<float>& index,
    const std::vector<float>& base_data,
    size_t nb,
    int dim,
    int threads
) {
    std::atomic<size_t> next_id{0};
    std::atomic<size_t> done{0};
    std::mutex print_mutex;

    const auto t0 = std::chrono::high_resolution_clock::now();
    const size_t report_every = std::max<size_t>(1, nb / 20);

    auto worker = [&]() {
        while (true) {
            const size_t i = next_id.fetch_add(1);
            if (i >= nb) break;
            index.addPoint(
                static_cast<const void*>(base_data.data() + i * static_cast<size_t>(dim)),
                static_cast<hnswlib::labeltype>(i)
            );

            const size_t d = done.fetch_add(1) + 1;
            if (d % report_every == 0 || d == nb) {
                std::lock_guard<std::mutex> lock(print_mutex);
                print_progress("Build HNSWLIB", d, nb, t0);
            }
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(threads));
    for (int t = 0; t < threads; ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
}

class HnswlibServer {
public:
    HnswlibServer(
        std::unique_ptr<hnswlib::SpaceInterface<float>> space,
        std::unique_ptr<hnswlib::HierarchicalNSW<float>> index,
        int dim,
        int default_top_k,
        int default_ef,
        int threads,
        bool log_each_request
    ) : space_(std::move(space)),
        index_(std::move(index)),
        dim_(dim),
        default_top_k_(default_top_k),
        default_ef_(std::max(default_ef, default_top_k)),
        threads_(std::max(1, threads)),
        log_each_request_(log_each_request) {}

    void handle_connection(int connfd) {
        int flag = 1;
        ::setsockopt(connfd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
        int sndbuf = 4 * 1024 * 1024;
        int rcvbuf = 4 * 1024 * 1024;
        ::setsockopt(connfd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));
        ::setsockopt(connfd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

        try {
            while (true) {
                QueryBatchRequestHeader req{};
                if (!recv_all(connfd, &req, sizeof(req))) break;

                if (req.magic != HNSWLIB_PROTOCOL_MAGIC || req.version != HNSWLIB_PROTOCOL_VERSION) {
                    throw std::runtime_error("protocol mismatch in request");
                }
                if (req.batch_size <= 0 || req.dim != dim_) {
                    throw std::runtime_error("bad request: invalid batch_size or dimension mismatch");
                }
                if (req.top_k <= 0) req.top_k = default_top_k_;
                int effective_ef = req.ef > 0 ? req.ef : default_ef_;
                effective_ef = std::max(effective_ef, req.top_k);

                const size_t expected_body =
                    static_cast<size_t>(req.batch_size) *
                    static_cast<size_t>(req.dim) * sizeof(float);
                if (req.body_bytes != static_cast<int32_t>(expected_body)) {
                    throw std::runtime_error("bad request: body_bytes mismatch");
                }

                std::vector<float> queries(static_cast<size_t>(req.batch_size) * static_cast<size_t>(req.dim));
                if (!recv_all(connfd, queries.data(), expected_body)) break;

                std::vector<int64_t> labels(static_cast<size_t>(req.batch_size) * static_cast<size_t>(req.top_k), -1);
                std::vector<float> distances(static_cast<size_t>(req.batch_size) * static_cast<size_t>(req.top_k), 0.0f);

                const auto t0 = std::chrono::high_resolution_clock::now();
                {

                    std::lock_guard<std::mutex> lock(search_mutex_);
                    index_->setEf(static_cast<size_t>(effective_ef));

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_) schedule(dynamic)
#endif
                    for (int b = 0; b < req.batch_size; ++b) {
                        auto result = index_->searchKnn(
                            static_cast<const void*>(queries.data() + static_cast<size_t>(b) * static_cast<size_t>(req.dim)),
                            static_cast<size_t>(req.top_k)
                        );

                        std::vector<std::pair<float, hnswlib::labeltype>> items;
                        items.reserve(static_cast<size_t>(req.top_k));
                        while (!result.empty()) {
                            items.push_back(result.top());
                            result.pop();
                        }
                        std::sort(items.begin(), items.end(),
                                  [](const auto& a, const auto& b) { return a.first < b.first; });

                        const size_t base = static_cast<size_t>(b) * static_cast<size_t>(req.top_k);
                        for (size_t j = 0; j < items.size() && j < static_cast<size_t>(req.top_k); ++j) {
                            distances[base + j] = items[j].first;
                            labels[base + j] = static_cast<int64_t>(items[j].second);
                        }
                    }
                }
                const auto t1 = std::chrono::high_resolution_clock::now();

                const uint64_t batch_compute_us = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
                );
                const uint64_t per_query_compute_us =
                    batch_compute_us / static_cast<uint64_t>(std::max(1, req.batch_size));

                if (log_each_request_) {
                    std::cout << "[Server] batch_size=" << req.batch_size
                              << ", top_k=" << req.top_k
                              << ", requested_ef=" << req.ef
                              << ", used_ef=" << effective_ef
                              << ", batch_compute_ms=" << std::fixed << std::setprecision(3)
                              << static_cast<double>(batch_compute_us) / 1000.0
                              << std::endl;
                }

                std::vector<uint8_t> resp_body;
                resp_body.reserve(
                    static_cast<size_t>(req.batch_size) *
                    (sizeof(QueryResultHeader) + static_cast<size_t>(req.top_k) * sizeof(ResultItem))
                );

                for (int b = 0; b < req.batch_size; ++b) {
                    QueryResultHeader qh{};
                    qh.query_id = req.query_id_start + static_cast<uint64_t>(b);
                    qh.server_compute_us = per_query_compute_us;
                    qh.result_count = req.top_k;
                    qh.body_bytes = static_cast<int32_t>(static_cast<size_t>(req.top_k) * sizeof(ResultItem));
                    append_pod(resp_body, qh);

                    const size_t base = static_cast<size_t>(b) * static_cast<size_t>(req.top_k);
                    for (int j = 0; j < req.top_k; ++j) {
                        ResultItem item{};
                        item.label = labels[base + static_cast<size_t>(j)];
                        item.distance = distances[base + static_cast<size_t>(j)];
                        append_pod(resp_body, item);
                    }
                }

                QueryBatchResponseHeader resp{};
                resp.magic = HNSWLIB_PROTOCOL_MAGIC;
                resp.version = HNSWLIB_PROTOCOL_VERSION;
                resp.flags = req.flags;
                resp.batch_size = req.batch_size;
                resp.top_k = req.top_k;
                resp.ef_used = effective_ef;
                resp.body_bytes = static_cast<int32_t>(resp_body.size());

                if (!send_all(connfd, &resp, sizeof(resp))) break;
                if (!resp_body.empty() && !send_all(connfd, resp_body.data(), resp_body.size())) break;
            }
        } catch (const std::exception& e) {
            std::cerr << "[Server] connection error: " << e.what() << std::endl;
        }

        ::close(connfd);
    }

private:
    std::unique_ptr<hnswlib::SpaceInterface<float>> space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
    int dim_ = 0;
    int default_top_k_ = 10;
    int default_ef_ = 50;
    int threads_ = 16;
    bool log_each_request_ = false;
    std::mutex search_mutex_;
};

int main(int argc, char** argv) {
    try {
        RuntimeOptions opt = parse_args(argc, argv);

#ifdef _OPENMP
        if (opt.threads > 0) omp_set_num_threads(opt.threads);
#endif

        Paths paths = make_paths(opt.dataset);
        if (!opt.base_fvecs_override.empty()) paths.base_fvecs = opt.base_fvecs_override;
        require_file(paths.base_fvecs, "base fvecs");

        const MetricKind metric = metric_for_dataset(opt.dataset);

        std::cout << "=== hnswlib Runtime Server v1 ===" << std::endl;
        std::cout << "Dataset:               " << paths.dataset_name << std::endl;
        std::cout << "Resolved data dir:     " << paths.data_root << std::endl;
        std::cout << "Base fvecs:            " << paths.base_fvecs << std::endl;
        std::cout << "Metric:                " << metric_name(metric) << std::endl;
        std::cout << "M:                     " << opt.M << std::endl;
        std::cout << "ef_construction:       " << opt.ef_construction << std::endl;
        std::cout << "Default ef:            " << opt.ef << std::endl;
        std::cout << "Top-K default:         " << opt.top_k << std::endl;
        std::cout << "Threads:               " << opt.threads << std::endl;
        std::cout << "Listen:                " << opt.bind_ip << ":" << opt.port << std::endl;
        std::cout << "Protocol version:      " << HNSWLIB_PROTOCOL_VERSION << std::endl;
        std::cout << "Log each request:      " << (opt.log_each_request ? "true" : "false") << std::endl;

        std::vector<float> base_data;
        size_t nb = 0;
        int dim = 0;
        load_fvecs(paths.base_fvecs, base_data, nb, dim, "base");

        std::cout << "\n[Build] create hnswlib index" << std::endl;
        const auto build_t0 = std::chrono::high_resolution_clock::now();

        auto space = make_space(metric, dim);
        auto index = std::unique_ptr<hnswlib::HierarchicalNSW<float>>(
            new hnswlib::HierarchicalNSW<float>(space.get(), static_cast<size_t>(nb), opt.M, opt.ef_construction)
        );

        build_hnsw_index_parallel(*index, base_data, nb, dim, opt.threads);
        index->setEf(static_cast<size_t>(opt.ef));

        const double build_sec = elapsed_sec(build_t0);
        std::cout << "[Build] finished, build_time=" << std::fixed << std::setprecision(3)
                  << build_sec << "s" << std::endl;

        std::vector<float>().swap(base_data);

        HnswlibServer server(
            std::move(space),
            std::move(index),
            dim,
            opt.top_k,
            opt.ef,
            opt.threads,
            opt.log_each_request
        );

        int listenfd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listenfd < 0) throw std::runtime_error("socket() failed");

        int yes = 1;
        ::setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
        ::setsockopt(listenfd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(opt.port));
        if (opt.bind_ip == "0.0.0.0") {
            addr.sin_addr.s_addr = INADDR_ANY;
        } else if (::inet_pton(AF_INET, opt.bind_ip.c_str(), &addr.sin_addr) <= 0) {
            throw std::runtime_error("inet_pton() failed for bind_ip");
        }

        if (::bind(listenfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            throw std::runtime_error("bind() failed");
        }
        if (::listen(listenfd, 16) < 0) throw std::runtime_error("listen() failed");

        std::cout << "\n[Server] listening on " << opt.bind_ip << ":" << opt.port
                  << " (raw-query hnswlib search protocol)" << std::endl;

        while (true) {
            sockaddr_in cli{};
            socklen_t len = sizeof(cli);
            int connfd = ::accept(listenfd, reinterpret_cast<sockaddr*>(&cli), &len);
            if (connfd < 0) continue;
            std::thread(&HnswlibServer::handle_connection, &server, connfd).detach();
        }

        ::close(listenfd);
    } catch (const std::exception& e) {
        std::cerr << "HNSWLIB SERVER ERROR: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
