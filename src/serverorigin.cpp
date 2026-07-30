#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <queue>
#include <random>
#include <numeric>
#include <memory>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <mutex>
#include <cstring>
#include <cctype>
#include <cstdlib>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <netinet/tcp.h>

#include <omp.h>

#include "util/lsh_encoder.hpp"
#include "util/hnsw_index.hpp"
#include "util/hash_codeword.hpp"
#include "util/securesearchcontext.hpp"
#include "util/data_paths.hpp"

enum class FeatureFileType {
    FVECS,
    BIN_UINT8,
    BIN_FLOAT32,
};

enum class DatasetKind {
    Sift,
    Laion,
    Trip,
    MsMarco,
    Sift100M,
};

bool get_fvecs_metadata(const std::string& path, int& out_num, int& out_dim) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;
    in.read(reinterpret_cast<char*>(&out_dim), 4);
    if (!in) return false;
    in.seekg(0, std::ios::end);
    out_num = static_cast<int>(in.tellg() / (4 + out_dim * 4));
    return true;
}

bool get_bin_metadata(const std::string& path, int& out_num, int& out_dim) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;

    int32_t num_i32 = 0, dim_i32 = 0;
    in.read(reinterpret_cast<char*>(&num_i32), 4);
    in.read(reinterpret_cast<char*>(&dim_i32), 4);
    if (!in) return false;

    out_num = static_cast<int>(num_i32);
    out_dim = static_cast<int>(dim_i32);
    return true;
}

void load_fvecs(const std::string& filename, std::vector<float>& data, size_t& num, int& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open fvecs file: " + filename);
    }

    in.read(reinterpret_cast<char*>(&dim), 4);
    if (!in) throw std::runtime_error("Failed to read fvecs dim: " + filename);

    in.seekg(0, std::ios::end);
    num = static_cast<size_t>(in.tellg() / (4 + dim * 4));
    data.resize(num * static_cast<size_t>(dim));
    in.seekg(0, std::ios::beg);

    for (size_t i = 0; i < num; ++i) {
        int d = 0;
        in.read(reinterpret_cast<char*>(&d), 4);
        if (!in || d != dim) {
            throw std::runtime_error("Corrupted fvecs or dim mismatch: " + filename);
        }
        in.read(reinterpret_cast<char*>(data.data() + i * static_cast<size_t>(dim)), dim * 4);
        if (!in) throw std::runtime_error("Failed while reading fvecs data: " + filename);
    }
}

void load_bin_uint8_to_float(const std::string& filename, std::vector<float>& data, size_t& num, int& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open uint8 bin file: " + filename);
    }

    int32_t num_i32 = 0, dim_i32 = 0;
    in.read(reinterpret_cast<char*>(&num_i32), 4);
    in.read(reinterpret_cast<char*>(&dim_i32), 4);
    if (!in) throw std::runtime_error("Failed to read uint8 bin header: " + filename);

    num = static_cast<size_t>(num_i32);
    dim = static_cast<int>(dim_i32);
    data.resize(num * static_cast<size_t>(dim));

    const size_t chunk_size = 1000000;
    std::vector<uint8_t> buffer(chunk_size * static_cast<size_t>(dim));

    size_t vecs_read = 0;
    while (vecs_read < num) {
        const size_t to_read = std::min(chunk_size, num - vecs_read);
        in.read(reinterpret_cast<char*>(buffer.data()),
                static_cast<std::streamsize>(to_read * static_cast<size_t>(dim)));
        if (!in) throw std::runtime_error("Failed while reading uint8 bin payload: " + filename);

        #pragma omp parallel for default(none) shared(to_read, dim, vecs_read, data, buffer)
        for (int64_t i = 0; i < static_cast<int64_t>(to_read * static_cast<size_t>(dim)); ++i) {
            data[vecs_read * static_cast<size_t>(dim) + static_cast<size_t>(i)] =
                static_cast<float>(buffer[static_cast<size_t>(i)]);
        }
        vecs_read += to_read;
    }
}

void load_bin_float32(const std::string& filename, std::vector<float>& data, size_t& num, int& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open float32 bin file: " + filename);
    }

    int32_t num_i32 = 0, dim_i32 = 0;
    in.read(reinterpret_cast<char*>(&num_i32), 4);
    in.read(reinterpret_cast<char*>(&dim_i32), 4);
    if (!in) throw std::runtime_error("Failed to read float32 bin header: " + filename);

    num = static_cast<size_t>(num_i32);
    dim = static_cast<int>(dim_i32);
    data.resize(num * static_cast<size_t>(dim));

    in.read(reinterpret_cast<char*>(data.data()),
            static_cast<std::streamsize>(num * static_cast<size_t>(dim) * sizeof(float)));
    if (!in) throw std::runtime_error("Failed while reading float32 bin payload: " + filename);
}

void load_feature_file(const std::string& filename,
                       FeatureFileType file_type,
                       std::vector<float>& data,
                       size_t& num,
                       int& dim) {
    switch (file_type) {
        case FeatureFileType::FVECS:
            load_fvecs(filename, data, num, dim);
            break;
        case FeatureFileType::BIN_UINT8:
            load_bin_uint8_to_float(filename, data, num, dim);
            break;
        case FeatureFileType::BIN_FLOAT32:
            load_bin_float32(filename, data, num, dim);
            break;
        default:
            throw std::runtime_error("Unsupported feature file type.");
    }
}

class IBaseBatchReader {
public:
    virtual ~IBaseBatchReader() = default;
    virtual int dim() const = 0;
    virtual size_t total() const = 0;
    virtual size_t cursor() const = 0;
    virtual bool read_batch(size_t batch_size, std::vector<float>& out, size_t& out_num) = 0;
};

class FvecsBatchReader final : public IBaseBatchReader {
public:
    explicit FvecsBatchReader(const std::string& path)
        : in_(path, std::ios::binary), dim_(0), total_(0), cursor_(0) {
        if (!in_.is_open()) {
            throw std::runtime_error("Cannot open file: " + path);
        }

        in_.read(reinterpret_cast<char*>(&dim_), 4);
        if (!in_) {
            throw std::runtime_error("Failed to read dim from: " + path);
        }

        in_.seekg(0, std::ios::end);
        const std::streamoff bytes = in_.tellg();
        total_ = static_cast<size_t>(bytes / (4 + dim_ * 4));
        in_.seekg(0, std::ios::beg);
    }

    int dim() const override { return dim_; }
    size_t total() const override { return total_; }
    size_t cursor() const override { return cursor_; }

    bool read_batch(size_t batch_size, std::vector<float>& out, size_t& out_num) override {
        if (cursor_ >= total_) {
            out_num = 0;
            out.clear();
            return false;
        }

        out_num = std::min(batch_size, total_ - cursor_);
        out.resize(out_num * static_cast<size_t>(dim_));

        for (size_t i = 0; i < out_num; ++i) {
            int d = 0;
            in_.read(reinterpret_cast<char*>(&d), 4);
            if (!in_ || d != dim_) {
                throw std::runtime_error("Corrupted fvecs batch or dim mismatch.");
            }
            in_.read(reinterpret_cast<char*>(out.data() + i * static_cast<size_t>(dim_)), dim_ * 4);
            if (!in_) {
                throw std::runtime_error("Failed while reading fvecs batch.");
            }
        }

        cursor_ += out_num;
        return true;
    }

private:
    std::ifstream in_;
    int dim_;
    size_t total_;
    size_t cursor_;
};

class BinUint8BatchReader final : public IBaseBatchReader {
public:
    explicit BinUint8BatchReader(const std::string& path)
        : in_(path, std::ios::binary), dim_(0), total_(0), cursor_(0) {
        if (!in_.is_open()) {
            throw std::runtime_error("Cannot open file: " + path);
        }

        int32_t num_i32 = 0, dim_i32 = 0;
        in_.read(reinterpret_cast<char*>(&num_i32), 4);
        in_.read(reinterpret_cast<char*>(&dim_i32), 4);
        if (!in_) {
            throw std::runtime_error("Failed to read bin header from: " + path);
        }

        total_ = static_cast<size_t>(num_i32);
        dim_ = static_cast<int>(dim_i32);
    }

    int dim() const override { return dim_; }
    size_t total() const override { return total_; }
    size_t cursor() const override { return cursor_; }

    bool read_batch(size_t batch_size, std::vector<float>& out, size_t& out_num) override {
        if (cursor_ >= total_) {
            out_num = 0;
            out.clear();
            return false;
        }

        out_num = std::min(batch_size, total_ - cursor_);
        out.resize(out_num * static_cast<size_t>(dim_));

        std::vector<uint8_t> buffer(out_num * static_cast<size_t>(dim_));
        in_.read(reinterpret_cast<char*>(buffer.data()),
                 static_cast<std::streamsize>(out_num * static_cast<size_t>(dim_)));
        if (!in_) {
            throw std::runtime_error("Failed while reading uint8 bin batch.");
        }

        #pragma omp parallel for default(none) shared(out_num, dim_, out, buffer)
        for (int64_t i = 0; i < static_cast<int64_t>(out_num * static_cast<size_t>(dim_)); ++i) {
            out[static_cast<size_t>(i)] = static_cast<float>(buffer[static_cast<size_t>(i)]);
        }

        cursor_ += out_num;
        return true;
    }

private:
    std::ifstream in_;
    int dim_;
    size_t total_;
    size_t cursor_;
};

class BinFloat32BatchReader final : public IBaseBatchReader {
public:
    explicit BinFloat32BatchReader(const std::string& path)
        : in_(path, std::ios::binary), dim_(0), total_(0), cursor_(0) {
        if (!in_.is_open()) {
            throw std::runtime_error("Cannot open file: " + path);
        }

        int32_t num_i32 = 0, dim_i32 = 0;
        in_.read(reinterpret_cast<char*>(&num_i32), 4);
        in_.read(reinterpret_cast<char*>(&dim_i32), 4);
        if (!in_) {
            throw std::runtime_error("Failed to read bin header from: " + path);
        }

        total_ = static_cast<size_t>(num_i32);
        dim_ = static_cast<int>(dim_i32);
    }

    int dim() const override { return dim_; }
    size_t total() const override { return total_; }
    size_t cursor() const override { return cursor_; }

    bool read_batch(size_t batch_size, std::vector<float>& out, size_t& out_num) override {
        if (cursor_ >= total_) {
            out_num = 0;
            out.clear();
            return false;
        }

        out_num = std::min(batch_size, total_ - cursor_);
        out.resize(out_num * static_cast<size_t>(dim_));

        in_.read(reinterpret_cast<char*>(out.data()),
                 static_cast<std::streamsize>(out_num * static_cast<size_t>(dim_) * sizeof(float)));
        if (!in_) {
            throw std::runtime_error("Failed while reading float32 bin batch.");
        }

        cursor_ += out_num;
        return true;
    }

private:
    std::ifstream in_;
    int dim_;
    size_t total_;
    size_t cursor_;
};

std::unique_ptr<IBaseBatchReader> make_base_reader(const std::string& path, FeatureFileType file_type) {
    switch (file_type) {
        case FeatureFileType::FVECS:
            return std::make_unique<FvecsBatchReader>(path);
        case FeatureFileType::BIN_UINT8:
            return std::make_unique<BinUint8BatchReader>(path);
        case FeatureFileType::BIN_FLOAT32:
            return std::make_unique<BinFloat32BatchReader>(path);
        default:
            throw std::runtime_error("Unsupported base file type.");
    }
}

static inline uint64_t fnv1a64(const std::vector<uint8_t>& bytes) {
    uint64_t h = 1469598103934665603ULL;
    for (uint8_t b : bytes) {
        h ^= static_cast<uint64_t>(b);
        h *= 1099511628211ULL;
    }
    return h;
}

void analyze_collisions_sampled(const std::vector<HashCodeword>& codes,
                                int hash_bits,
                                size_t max_samples = 50000) {
    const size_t sample_n = std::min(max_samples, codes.size());
    if (sample_n == 0) return;

    std::unordered_map<uint64_t, int> bucket;
    bucket.reserve(sample_n * 2);

    for (size_t i = 0; i < sample_n; ++i) {
        bucket[fnv1a64(codes[i].getBytes())]++;
    }

    int max_collision = 0;
    for (const auto& kv : bucket) {
        max_collision = std::max(max_collision, kv.second);
    }

    const double usage_rate = 100.0 * static_cast<double>(bucket.size()) / static_cast<double>(sample_n);

    std::cout << "\n>>> Phase 0: Collision Analysis (sampled)" << std::endl;
    std::cout << "Collision Report (" << hash_bits << " bits, sampled " << sample_n << "):" << std::endl;
    std::cout << "  - Unique Fingerprints: " << bucket.size()
              << " (" << std::fixed << std::setprecision(2) << usage_rate << "%)" << std::endl;
    std::cout << "  - Max Collisions in sample: " << max_collision << std::endl;
}

static constexpr uint32_t PROTOCOL_MAGIC = 0x51425443;
static constexpr uint16_t PROTOCOL_VERSION = 1;

static constexpr uint16_t REQUEST_FLAG_INSERT = 1 << 14;
static constexpr uint16_t REQUEST_FLAG_DELETE = 1 << 15;

struct QueryBatchRequestHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t flags;
    int32_t batch_size;
    int32_t top_k;
    int32_t per_shard_pool;
    int32_t num_tables;
    int32_t sub_hash_bytes;
    int32_t body_bytes;
};

struct QueryBatchResponseHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t flags;
    int32_t batch_size;
    int32_t body_bytes;
};

struct UpdateResponseHeader {
    int32_t status;
    int64_t label;
    int32_t touched_shards;
    int32_t message_bytes;
};

struct QueryResponseHeader {
    uint64_t query_id;
    uint64_t server_compute_us;
    int32_t candidate_count;
    int32_t body_bytes;
};

struct CandidateHeader {
    int32_t label;
    int32_t cipher_size;
};

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

template <typename T>
static T read_pod(const std::vector<uint8_t>& buf, size_t& offset) {
    if (offset + sizeof(T) > buf.size()) {
        throw std::runtime_error("read_pod out of range");
    }
    T value{};
    std::memcpy(&value, buf.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

static HashCodeword make_codeword_from_ptr(const uint8_t* bytes, int sub_hash_bits) {
    const int expected_bytes = (sub_hash_bits + 7) / 8;
    std::vector<uint8_t> tmp(static_cast<size_t>(expected_bytes));
    std::memcpy(tmp.data(), bytes, static_cast<size_t>(expected_bytes));
    return HashCodeword(tmp);
}

static std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return s;
}

struct ServerDatasetConfig {
    DatasetKind dataset;
    std::string name;
    std::string dataset_root;
    std::string base_path;
    FeatureFileType base_file_type;
};

struct RuntimeOptions {
    DatasetKind dataset = DatasetKind::Sift;

    int hash_bits = 8192;
    int num_tables = 64;
    int graphs_per_vector = 16;

    int M = 16;
    int efConstruction = 300;
    int efSearch = 0;
    int top_k = 10;
    int candidate_pool = 4500;

    size_t build_batch_size = 20000;
    int omp_threads = 0;

    std::string bind_ip = "0.0.0.0";
    int port = 9090;

    float p_data = 0.08f;

    std::string base_path_override;
};

static DatasetKind parse_dataset_kind(const std::string& dataset_arg) {
    const std::string ds = to_lower_copy(dataset_arg);
    if (ds == "sift" || ds == "sift1m") return DatasetKind::Sift;
    if (ds == "laion" || ds == "laion1m") return DatasetKind::Laion;
    if (ds == "trip" || ds == "tripclick") return DatasetKind::Trip;
    if (ds == "msmarco" || ds == "msmarco_bert") return DatasetKind::MsMarco;
    if (ds == "sift100m" || ds == "sift100m_uint8") return DatasetKind::Sift100M;
    throw std::runtime_error(
        "Unsupported dataset: " + dataset_arg +
        " (use sift, laion, trip, msmarco, or sift100m)"
    );
}

static const char* dataset_kind_to_string(DatasetKind dataset) {
    switch (dataset) {
        case DatasetKind::Sift: return "sift";
        case DatasetKind::Laion: return "laion";
        case DatasetKind::Trip: return "trip";
        case DatasetKind::MsMarco: return "msmarco";
        case DatasetKind::Sift100M: return "sift100m";
        default: return "unknown";
    }
}

static ServerDatasetConfig make_server_dataset_config(DatasetKind dataset) {
    switch (dataset) {
        case DatasetKind::Sift:
            return {
                DatasetKind::Sift,
                "sift",
                mess::data_path("dataset/sift"),
                mess::data_path("dataset/sift/base.fvecs"),
                FeatureFileType::FVECS
            };
        case DatasetKind::Laion:
            return {
                DatasetKind::Laion,
                "laion",
                mess::data_path("dataset/laion1m"),
                mess::data_path("dataset/laion1m/100k/laion_base.fvecs"),
                FeatureFileType::FVECS
            };
        case DatasetKind::Trip:
            return {
                DatasetKind::Trip,
                "trip",
                mess::data_path("dataset/trip_distilbert"),
                mess::data_path("dataset/trip_distilbert/passages.fvecs"),
                FeatureFileType::FVECS
            };
        case DatasetKind::MsMarco:
            return {
                DatasetKind::MsMarco,
                "msmarco",
                mess::data_path("dataset/msmarco_bert"),
                mess::data_path("dataset/msmarco_bert/passages.fvecs"),
                FeatureFileType::FVECS
            };
        case DatasetKind::Sift100M:
            return {
                DatasetKind::Sift100M,
                "sift100m",
                mess::data_path("dataset/sift100m"),
                mess::data_path("dataset/sift100m/base.bin"),
                FeatureFileType::BIN_UINT8
            };
    }
    throw std::runtime_error("Unsupported dataset kind.");
}

static void print_server_usage(const char* prog) {
    std::cout << "Usage: " << prog
              << " [--dataset sift|laion|trip|msmarco|sift100m]"
              << " [--M N] [--efConstruction N] [--efSearch N]"
              << " [--candidate_pool N] [--hash_bits N] [--num_tables N]"
              << " [--graphs_per_vector N] [--p_data X]"
              << " [--bind_ip IP] [--port N] [--base_path PATH]"
              << " [--omp_threads N]" << std::endl;
}

static RuntimeOptions parse_runtime_options(int argc, char** argv) {
    RuntimeOptions opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) {
            opt.dataset = parse_dataset_kind(argv[++i]);
        } else if (arg == "--M" && i + 1 < argc) {
            opt.M = std::stoi(argv[++i]);
        } else if (arg == "--efConstruction" && i + 1 < argc) {
            opt.efConstruction = std::stoi(argv[++i]);
        } else if (arg == "--efSearch" && i + 1 < argc) {
            opt.efSearch = std::stoi(argv[++i]);
        } else if (arg == "--top_k" && i + 1 < argc) {
            opt.top_k = std::stoi(argv[++i]);
        } else if (arg == "--candidate_pool" && i + 1 < argc) {
            opt.candidate_pool = std::stoi(argv[++i]);
        } else if (arg == "--hash_bits" && i + 1 < argc) {
            opt.hash_bits = std::stoi(argv[++i]);
        } else if (arg == "--num_tables" && i + 1 < argc) {
            opt.num_tables = std::stoi(argv[++i]);
        } else if (arg == "--graphs_per_vector" && i + 1 < argc) {
            opt.graphs_per_vector = std::stoi(argv[++i]);
        } else if (arg == "--build_batch_size" && i + 1 < argc) {
            opt.build_batch_size = static_cast<size_t>(std::stoll(argv[++i]));
        } else if (arg == "--omp_threads" && i + 1 < argc) {
            opt.omp_threads = std::stoi(argv[++i]);
        } else if (arg == "--bind_ip" && i + 1 < argc) {
            opt.bind_ip = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            opt.port = std::stoi(argv[++i]);
        } else if (arg == "--p_data" && i + 1 < argc) {
            opt.p_data = std::stof(argv[++i]);
        } else if (arg == "--base_path" && i + 1 < argc) {
            opt.base_path_override = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_server_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
    }
    return opt;
}

class ServerEngine {
public:
    explicit ServerEngine(const RuntimeOptions& opt)
        : runtime(opt),
          data_cfg(make_server_dataset_config(opt.dataset)) {
        if (!runtime.base_path_override.empty()) {
            data_cfg.base_path = runtime.base_path_override;
        }
        if (runtime.omp_threads > 0) {
            omp_set_num_threads(runtime.omp_threads);
        }

        dataset_kind = data_cfg.dataset;
        dataset_root = data_cfg.dataset_root;
        base_path = data_cfg.base_path;
        base_file_type = data_cfg.base_file_type;

        hash_bits = runtime.hash_bits;
        num_tables = runtime.num_tables;
        graphs_per_vector = runtime.graphs_per_vector;
        M = runtime.M;
        ef_construction = runtime.efConstruction;
        k = runtime.top_k;
        candidate_pool = runtime.candidate_pool;
        ef_search = runtime.efSearch;
        build_batch_size = runtime.build_batch_size;
        p_data = runtime.p_data;
    }

    RuntimeOptions runtime;
    ServerDatasetConfig data_cfg;

    DatasetKind dataset_kind;
    std::string dataset_root;
    std::string base_path;
    FeatureFileType base_file_type;

    int num_vectors = 0;
    int input_dim = 0;

    int hash_bits = 0;
    int num_tables = 0;
    int graphs_per_vector = 0;
    int M = 0;
    int ef_construction = 0;
    int k = 0;
    int candidate_pool = 0;
    int per_shard_pool = 0;
    int ef_search = 0;
    int sub_hash_bits = 0;
    int sub_hash_bytes = 0;
    size_t build_batch_size = 0;
    float p_data = 0.0f;

    std::unique_ptr<LSHEncoder> encoder;
    std::unique_ptr<SecureSearchContext> graph_secure_ctx;
    std::unique_ptr<SecureSearchContext> payload_secure_ctx;

    std::vector<std::unique_ptr<HNSWIndex>> indices;
    std::vector<std::vector<uint8_t>> encrypted_payloads;

    std::mutex index_mutex;
    std::vector<uint8_t> deleted_labels;
    size_t next_dynamic_label = 0;

    void init_and_build() {
        bool meta_ok = false;
        switch (base_file_type) {
            case FeatureFileType::FVECS:
                meta_ok = get_fvecs_metadata(base_path, num_vectors, input_dim);
                break;
            case FeatureFileType::BIN_UINT8:
            case FeatureFileType::BIN_FLOAT32:
                meta_ok = get_bin_metadata(base_path, num_vectors, input_dim);
                break;
        }
        if (!meta_ok) throw std::runtime_error("Failed to read base metadata from " + base_path);

        std::cout << "=== Server: Vector Search System ===" << std::endl;
        std::cout << "Dataset name: " << dataset_kind_to_string(dataset_kind) << std::endl;
        std::cout << "Dataset root: " << dataset_root << std::endl;
        std::cout << "Base path: " << base_path << std::endl;
        std::cout << "Dataset: " << num_vectors << " vectors, Dim=" << input_dim << std::endl;
        std::cout << "HNSW: M=" << M << ", efConstruction=" << ef_construction
                  << ", efSearch=" << ef_search << std::endl;
        std::cout << "LSH: hash_bits=" << hash_bits << ", num_tables=" << num_tables
                  << ", graphs_per_vector=" << graphs_per_vector
                  << ", p_data=" << p_data << std::endl;

        graphs_per_vector = std::min(graphs_per_vector, num_tables);
        if (hash_bits % (num_tables * 8) != 0) {
            throw std::runtime_error("Total hash bits must be evenly divisible by (num_tables * 8)!");
        }
        sub_hash_bits = hash_bits / num_tables;
        sub_hash_bytes = sub_hash_bits / 8;
        per_shard_pool = std::max(1, candidate_pool / num_tables);
        ef_search = std::max(1, ef_search);

        encoder = std::make_unique<LSHEncoder>(input_dim, hash_bits, HashMethod::ISOHASH, dataset_root);
        SearchContext search_cfg;
        search_cfg.k = k;
        search_cfg.ef_search = ef_search;
        search_cfg.verbose = true;

        SecureSearchContext::Config graph_cfg;
        graph_cfg.num_graphs = num_tables;
        graph_cfg.input_dim = input_dim;
        graph_cfg.search = search_cfg;
        graph_secure_ctx = std::make_unique<SecureSearchContext>(graph_cfg);

        SecureSearchContext::Config payload_cfg;
        payload_cfg.num_graphs = 1;
        payload_cfg.input_dim = input_dim;
        payload_cfg.search = search_cfg;
        payload_cfg.use_pre_shared_master_key = true;
        payload_cfg.master_key = {
            0x10, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87,
            0x98, 0xa9, 0xba, 0xcb, 0xdc, 0xed, 0xfe, 0x0f,
            0x55, 0x44, 0x33, 0x22, 0x11, 0xaa, 0xbb, 0xcc,
            0xdd, 0xee, 0xff, 0x12, 0x34, 0x56, 0x78, 0x9a
        };
        payload_secure_ctx = std::make_unique<SecureSearchContext>(payload_cfg);

        const size_t est_elements = std::max(
            static_cast<size_t>(1),
            static_cast<size_t>((static_cast<size_t>(num_vectors) * static_cast<size_t>(graphs_per_vector)) /
                                static_cast<size_t>(num_tables) * 1.2)
        );
        indices.reserve(num_tables);
        for (int s = 0; s < num_tables; ++s) {
            indices.emplace_back(std::make_unique<HNSWIndex>(sub_hash_bits, est_elements, M, ef_construction));
            indices[s]->set_ef(ef_search);
        }
        encrypted_payloads.resize(static_cast<size_t>(num_vectors));
        deleted_labels.assign(static_cast<size_t>(num_vectors), 0);
        next_dynamic_label = static_cast<size_t>(num_vectors);

        std::cout << "\n>>> Phase 1: Stream Build Index + Encrypt Payloads" << std::endl;
        const auto t1 = std::chrono::high_resolution_clock::now();

        auto base_reader = make_base_reader(base_path, base_file_type);
        if (base_reader->dim() != input_dim) throw std::runtime_error("Base dim mismatch.");

        std::mt19937 rng(42);
        std::vector<int> shard_candidates(num_tables);
        std::iota(shard_candidates.begin(), shard_candidates.end(), 0);
        bool did_collision_analysis = false;
        size_t global_base_id = 0;

        while (global_base_id < static_cast<size_t>(num_vectors)) {
            std::vector<float> batch_data;
            size_t batch_num = 0;
            if (!base_reader->read_batch(build_batch_size, batch_data, batch_num)) break;

            std::vector<HashCodeword> batch_codes = encoder->batch_encode(batch_data.data(), batch_num);
            if (!did_collision_analysis) {
                analyze_collisions_sampled(batch_codes, hash_bits, 30000);
                did_collision_analysis = true;
            }
            for (size_t i = 0; i < batch_num; ++i) batch_codes[i].inject_noise(p_data);

            std::vector<std::vector<HashCodeword>> shard_codes(num_tables);
            std::vector<std::vector<size_t>> shard_labels(num_tables);
            const size_t avg_per_shard = std::max<size_t>(1, (batch_num * static_cast<size_t>(graphs_per_vector)) /
                static_cast<size_t>(num_tables) + 16);
            for (int s = 0; s < num_tables; ++s) {
                shard_codes[s].reserve(avg_per_shard);
                shard_labels[s].reserve(avg_per_shard);
            }
            for (size_t i = 0; i < batch_num; ++i) {
                const size_t label = global_base_id + i;
                const float* raw_vec = batch_data.data() + i * static_cast<size_t>(input_dim);
                encrypted_payloads[label] = payload_secure_ctx->encrypt_vector(0, raw_vec, input_dim);
                std::shuffle(shard_candidates.begin(), shard_candidates.end(), rng);
                for (int j = 0; j < graphs_per_vector; ++j) {
                    const int s = shard_candidates[j];
                    HashCodeword sub_code = batch_codes[i].split(s, num_tables);
                    shard_codes[s].push_back(std::move(sub_code));
                    shard_labels[s].push_back(label);
                }
            }
            #pragma omp parallel for schedule(dynamic)
            for (int s = 0; s < num_tables; ++s) {
                if (!shard_codes[s].empty()) indices[s]->add_batch(shard_codes[s], shard_labels[s]);
            }
            global_base_id += batch_num;
            std::cout << "  -> Built " << global_base_id << " / " << num_vectors << std::endl;
        }

        const auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "Build finished in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                  << " ms." << std::endl;
    }

    bool is_active_label(size_t label) const {
        return label < encrypted_payloads.size()
            && !encrypted_payloads[label].empty()
            && label < deleted_labels.size()
            && deleted_labels[label] == 0;
    }

    std::vector<int> route_shards_for_label(size_t label) const {
        std::vector<int> shards(num_tables);
        std::iota(shards.begin(), shards.end(), 0);

        uint32_t seed = static_cast<uint32_t>(
            (label * 11400714819323198485ull) ^ 0x9e3779b97f4a7c15ull
        );
        std::mt19937 rng(seed);
        std::shuffle(shards.begin(), shards.end(), rng);

        const int t = std::min(graphs_per_vector, num_tables);
        shards.resize(static_cast<size_t>(t));
        return shards;
    }

    size_t insert_item_locked(int64_t requested_label, const float* raw_vec, int dim) {
        if (dim != input_dim) {
            throw std::runtime_error("insert dimension mismatch");
        }

        size_t label;
        if (requested_label >= 0) {
            label = static_cast<size_t>(requested_label);
        } else {
            label = next_dynamic_label++;
        }

        if (label >= encrypted_payloads.size()) {
            encrypted_payloads.resize(label + 1);
            deleted_labels.resize(label + 1, 1);
        }

        if (!encrypted_payloads[label].empty() && deleted_labels[label] == 0) {
            throw std::runtime_error("insert label already exists and is active");
        }

        std::vector<HashCodeword> full_codes = encoder->batch_encode(raw_vec, 1);
        if (full_codes.empty()) {
            throw std::runtime_error("insert failed to encode vector");
        }
        full_codes[0].inject_noise(p_data);

        encrypted_payloads[label] = payload_secure_ctx->encrypt_vector(0, raw_vec, input_dim);
        deleted_labels[label] = 0;

        std::vector<int> shards = route_shards_for_label(label);
        for (int s : shards) {
            HashCodeword sub_code = full_codes[0].split(s, num_tables);
            std::vector<HashCodeword> one_code;
            one_code.push_back(std::move(sub_code));

            std::vector<size_t> one_label;
            one_label.push_back(label);

            indices[s]->add_batch(one_code, one_label);
        }

        if (label >= next_dynamic_label) {
            next_dynamic_label = label + 1;
        }
        return label;
    }

    bool logical_delete_locked(size_t label) {
        if (label >= encrypted_payloads.size()) {
            return false;
        }
        if (encrypted_payloads[label].empty()) {
            return false;
        }
        if (label >= deleted_labels.size()) {
            deleted_labels.resize(label + 1, 1);
        }
        if (deleted_labels[label] != 0) {
            return false;
        }

        deleted_labels[label] = 1;
        return true;
    }

    void send_update_response(int connfd,
                              uint16_t flags,
                              int32_t status,
                              int64_t label,
                              int32_t touched_shards,
                              const std::string& message) {
        std::vector<uint8_t> body;

        UpdateResponseHeader uresp{};
        uresp.status = status;
        uresp.label = label;
        uresp.touched_shards = touched_shards;
        uresp.message_bytes = static_cast<int32_t>(message.size());

        append_pod(body, uresp);
        if (!message.empty()) {
            body.insert(body.end(), message.begin(), message.end());
        }

        QueryBatchResponseHeader resp{};
        resp.magic = PROTOCOL_MAGIC;
        resp.version = PROTOCOL_VERSION;
        resp.flags = flags;
        resp.batch_size = 1;
        resp.body_bytes = static_cast<int32_t>(body.size());

        send_all(connfd, &resp, sizeof(resp));
        if (!body.empty()) {
            send_all(connfd, body.data(), body.size());
        }
    }

    void handle_insert_request(int connfd,
                               const QueryBatchRequestHeader& req,
                               const std::vector<uint8_t>& req_body) {
        try {
            size_t off = 0;
            int64_t requested_label = read_pod<int64_t>(req_body, off);
            int32_t dim = read_pod<int32_t>(req_body, off);
            if (dim <= 0) {
                throw std::runtime_error("invalid insert dimension");
            }
            const size_t vec_bytes = static_cast<size_t>(dim) * sizeof(float);
            if (off + vec_bytes > req_body.size()) {
                throw std::runtime_error("insert body too small for vector");
            }

            const float* raw_vec = reinterpret_cast<const float*>(req_body.data() + off);

            size_t label;
            {
                std::lock_guard<std::mutex> lock(index_mutex);
                label = insert_item_locked(requested_label, raw_vec, dim);
            }

            send_update_response(
                connfd,
                req.flags,
                0,
                static_cast<int64_t>(label),
                graphs_per_vector,
                "insert ok"
            );
        } catch (const std::exception& e) {
            send_update_response(connfd, req.flags, -1, -1, 0, e.what());
        }
    }

    void handle_delete_request(int connfd,
                               const QueryBatchRequestHeader& req,
                               const std::vector<uint8_t>& req_body) {
        try {
            size_t off = 0;
            int64_t label_i64 = read_pod<int64_t>(req_body, off);
            if (label_i64 < 0) {
                throw std::runtime_error("delete label must be non-negative");
            }

            bool ok;
            {
                std::lock_guard<std::mutex> lock(index_mutex);
                ok = logical_delete_locked(static_cast<size_t>(label_i64));
            }

            send_update_response(
                connfd,
                req.flags,
                ok ? 0 : -2,
                label_i64,
                ok ? graphs_per_vector : 0,
                ok ? "delete ok" : "label not found or already deleted"
            );
        } catch (const std::exception& e) {
            send_update_response(connfd, req.flags, -1, -1, 0, e.what());
        }
    }

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
            if (req.magic != PROTOCOL_MAGIC || req.version != PROTOCOL_VERSION) {
                throw std::runtime_error("protocol mismatch in request");
            }

            std::vector<uint8_t> req_body(static_cast<size_t>(req.body_bytes));
            if (req.body_bytes > 0) {
                if (!recv_all(connfd, req_body.data(), req_body.size())) break;
            }

            if ((req.flags & REQUEST_FLAG_INSERT) != 0) {
                handle_insert_request(connfd, req, req_body);
                continue;
            }
            if ((req.flags & REQUEST_FLAG_DELETE) != 0) {
                handle_delete_request(connfd, req, req_body);
                continue;
            }

            std::lock_guard<std::mutex> query_lock(index_mutex);

            if ((req.flags == 0) || (req.batch_size <= 1)) {
                std::vector<uint8_t> batch_body;
                size_t req_off = 0;

                for (int b = 0; b < req.batch_size; ++b) {
                    const auto t0 = std::chrono::high_resolution_clock::now();
                    std::vector<std::vector<int>> shard_hits(req.num_tables);

                    #pragma omp parallel for schedule(dynamic)
                    for (int s = 0; s < req.num_tables; ++s) {
                        const uint8_t* shard_ptr =
                            req_body.data() +
                            req_off +
                            static_cast<size_t>(s) * static_cast<size_t>(req.sub_hash_bytes);

                        HashCodeword q_sub_code = make_codeword_from_ptr(shard_ptr, sub_hash_bits);
                        auto local_candidates = indices[s]->search(q_sub_code, req.per_shard_pool);

                        shard_hits[s].reserve(local_candidates.size());
                        for (const auto& cand : local_candidates) {
                            if (cand.first < 0) {
                                continue;
                            }
                            const size_t label = static_cast<size_t>(cand.first);
                            if (!is_active_label(label)) {
                                continue;
                            }
                            shard_hits[s].push_back(cand.first);
                        }
                    }

                    std::vector<int> merged_candidates;
                    size_t total_hits = 0;
                    for (const auto& v : shard_hits) total_hits += v.size();
                    merged_candidates.reserve(total_hits);
                    for (auto& v : shard_hits) {
                        merged_candidates.insert(merged_candidates.end(), v.begin(), v.end());
                    }

                    const auto t1 = std::chrono::high_resolution_clock::now();

                    std::vector<size_t> cipher_sizes(merged_candidates.size(), 0);
                    size_t q_body_bytes = 0;
                    for (size_t idx = 0; idx < merged_candidates.size(); ++idx) {
                        cipher_sizes[idx] =
                            encrypted_payloads[static_cast<size_t>(merged_candidates[idx])].size();
                        q_body_bytes += sizeof(CandidateHeader) + cipher_sizes[idx];
                    }

                    QueryResponseHeader qresp{};
                    qresp.query_id = static_cast<uint64_t>(b);
                    qresp.server_compute_us =
                        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                    qresp.candidate_count = static_cast<int32_t>(merged_candidates.size());
                    qresp.body_bytes = static_cast<int32_t>(q_body_bytes);

                    append_pod(batch_body, qresp);

                    for (size_t idx = 0; idx < merged_candidates.size(); ++idx) {
                        const int label = merged_candidates[idx];
                        const auto& cipher = encrypted_payloads[static_cast<size_t>(label)];

                        CandidateHeader ch{};
                        ch.label = label;
                        ch.cipher_size = static_cast<int32_t>(cipher.size());

                        append_pod(batch_body, ch);
                        batch_body.insert(batch_body.end(), cipher.begin(), cipher.end());
                    }

                    req_off +=
                        static_cast<size_t>(req.num_tables) *
                        static_cast<size_t>(req.sub_hash_bytes);
                }

                QueryBatchResponseHeader batch_resp{};
                batch_resp.magic = PROTOCOL_MAGIC;
                batch_resp.version = PROTOCOL_VERSION;
                batch_resp.flags = req.flags;
                batch_resp.batch_size = req.batch_size;
                batch_resp.body_bytes = static_cast<int32_t>(batch_body.size());

                if (!send_all(connfd, &batch_resp, sizeof(batch_resp))) break;
                if (!batch_body.empty()) {
                    if (!send_all(connfd, batch_body.data(), batch_body.size())) break;
                }

                continue;
            }

            std::vector<std::vector<uint8_t>> query_bodies(static_cast<size_t>(req.batch_size));
            std::vector<size_t> query_body_sizes(static_cast<size_t>(req.batch_size), 0);

            #pragma omp parallel for schedule(static) num_threads(req.batch_size)
            for (int b = 0; b < req.batch_size; ++b) {
                const size_t req_off =
                    static_cast<size_t>(b) *
                    static_cast<size_t>(req.num_tables) *
                    static_cast<size_t>(req.sub_hash_bytes);

                const auto t0 = std::chrono::high_resolution_clock::now();
                std::vector<std::vector<int>> shard_hits(req.num_tables);

                for (int s = 0; s < req.num_tables; ++s) {
                    const uint8_t* shard_ptr =
                        req_body.data() +
                        req_off +
                        static_cast<size_t>(s) * static_cast<size_t>(req.sub_hash_bytes);

                    HashCodeword q_sub_code = make_codeword_from_ptr(shard_ptr, sub_hash_bits);
                    auto local_candidates = indices[s]->search(q_sub_code, req.per_shard_pool);

                    shard_hits[s].reserve(local_candidates.size());
                        for (const auto& cand : local_candidates) {
                            if (cand.first < 0) {
                                continue;
                            }
                            const size_t label = static_cast<size_t>(cand.first);
                            if (!is_active_label(label)) {
                                continue;
                            }
                            shard_hits[s].push_back(cand.first);
                        }
                }

                std::vector<int> merged_candidates;
                size_t total_hits = 0;
                for (const auto& v : shard_hits) total_hits += v.size();
                merged_candidates.reserve(total_hits);
                for (auto& v : shard_hits) {
                    merged_candidates.insert(merged_candidates.end(), v.begin(), v.end());
                }

                const auto t1 = std::chrono::high_resolution_clock::now();

                std::vector<size_t> cipher_sizes(merged_candidates.size(), 0);
                size_t q_body_bytes = 0;
                for (size_t idx = 0; idx < merged_candidates.size(); ++idx) {
                    cipher_sizes[idx] =
                        encrypted_payloads[static_cast<size_t>(merged_candidates[idx])].size();
                    q_body_bytes += sizeof(CandidateHeader) + cipher_sizes[idx];
                }

                QueryResponseHeader qresp{};
                qresp.query_id = static_cast<uint64_t>(b);
                qresp.server_compute_us =
                    std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                qresp.candidate_count = static_cast<int32_t>(merged_candidates.size());
                qresp.body_bytes = static_cast<int32_t>(q_body_bytes);

                std::vector<uint8_t>& local_body = query_bodies[static_cast<size_t>(b)];
                local_body.reserve(sizeof(QueryResponseHeader) + q_body_bytes);

                append_pod(local_body, qresp);

                for (size_t idx = 0; idx < merged_candidates.size(); ++idx) {
                    const int label = merged_candidates[idx];
                    const auto& cipher = encrypted_payloads[static_cast<size_t>(label)];

                    CandidateHeader ch{};
                    ch.label = label;
                    ch.cipher_size = static_cast<int32_t>(cipher.size());

                    append_pod(local_body, ch);
                    local_body.insert(local_body.end(), cipher.begin(), cipher.end());
                }

                query_body_sizes[static_cast<size_t>(b)] = local_body.size();
            }

            std::vector<uint8_t> batch_body;
            size_t total_batch_body_bytes = 0;
            for (size_t b = 0; b < static_cast<size_t>(req.batch_size); ++b) {
                total_batch_body_bytes += query_body_sizes[b];
            }
            batch_body.reserve(total_batch_body_bytes);

            for (size_t b = 0; b < static_cast<size_t>(req.batch_size); ++b) {
                batch_body.insert(
                    batch_body.end(),
                    query_bodies[b].begin(),
                    query_bodies[b].end()
                );
            }

            QueryBatchResponseHeader batch_resp{};
            batch_resp.magic = PROTOCOL_MAGIC;
            batch_resp.version = PROTOCOL_VERSION;
            batch_resp.flags = req.flags;
            batch_resp.batch_size = req.batch_size;
            batch_resp.body_bytes = static_cast<int32_t>(batch_body.size());

            if (!send_all(connfd, &batch_resp, sizeof(batch_resp))) break;
            if (!batch_body.empty()) {
                if (!send_all(connfd, batch_body.data(), batch_body.size())) break;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[Server] connection error: " << e.what() << std::endl;
    }
    ::close(connfd);
}
};

int main(int argc, char** argv) {
    try {
        RuntimeOptions runtime = parse_runtime_options(argc, argv);
        ServerEngine server(runtime);
        server.init_and_build();

        int listenfd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listenfd < 0) throw std::runtime_error("socket() failed");
        int opt = 1;
        ::setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        ::setsockopt(listenfd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(runtime.port));
        if (runtime.bind_ip == "0.0.0.0") {
            addr.sin_addr.s_addr = INADDR_ANY;
        } else if (::inet_pton(AF_INET, runtime.bind_ip.c_str(), &addr.sin_addr) <= 0) {
            throw std::runtime_error("invalid --bind_ip: " + runtime.bind_ip);
        }

        if (::bind(listenfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            throw std::runtime_error("bind() failed");
        }
        if (::listen(listenfd, 16) < 0) {
            throw std::runtime_error("listen() failed");
        }
        std::cout << "Server listening on " << runtime.bind_ip << ":" << runtime.port
                  << " ... (batch-capable)" << std::endl;

        while (true) {
            sockaddr_in cli{};
            socklen_t len = sizeof(cli);
            int connfd = ::accept(listenfd, reinterpret_cast<sockaddr*>(&cli), &len);
            if (connfd < 0) continue;
            std::thread(&ServerEngine::handle_connection, &server, connfd).detach();
        }
        ::close(listenfd);
    } catch (const std::exception& e) {
        std::cerr << "SERVER ERROR: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
