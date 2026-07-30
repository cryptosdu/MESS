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
#include <cstring>
#include <cctype>
#include <sstream>
#include <cstdlib>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <netinet/tcp.h>

#include <omp.h>

#include "util/lsh_encoder.hpp"
#include "util/hash_codeword.hpp"
#include "util/securesearchcontext.hpp"
#include "util/data_paths.hpp"

enum class FeatureFileType {
    FVECS,
    BIN_UINT8,
    BIN_FLOAT32,
};

enum class GroundTruthFileType {
    IVECS,
    BIN_INT32,
};

enum class EvalMetric {
    Recall,
    MRR,
};

enum class DatasetKind {
    Sift,
    Laion,
    Trip,
    MsMarco,
    Sift100M,
};

struct MsMarcoCompassMeta {
    std::vector<int> offset_to_pid;
    std::vector<int> offset_to_qid;
    std::unordered_map<int, std::unordered_set<int>> qrels;
};

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

void load_ground_truth_ivecs(const std::string& filename, std::vector<int>& data, int& num, int& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open ivecs ground-truth file: " + filename);
    }

    in.read(reinterpret_cast<char*>(&dim), 4);
    if (!in) throw std::runtime_error("Failed to read ivecs dim: " + filename);

    in.seekg(0, std::ios::end);
    num = static_cast<int>(in.tellg() / (4 + dim * 4));
    data.resize(static_cast<size_t>(num) * static_cast<size_t>(dim));
    in.seekg(0, std::ios::beg);

    for (int i = 0; i < num; ++i) {
        int d = 0;
        in.read(reinterpret_cast<char*>(&d), 4);
        if (!in || d != dim) {
            throw std::runtime_error("Corrupted ivecs or dim mismatch: " + filename);
        }
        in.read(reinterpret_cast<char*>(data.data() + static_cast<size_t>(i) * static_cast<size_t>(dim)),
                dim * static_cast<int>(sizeof(int32_t)));
        if (!in) throw std::runtime_error("Failed while reading ivecs data: " + filename);
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

    std::vector<uint8_t> buffer(num * static_cast<size_t>(dim));
    in.read(reinterpret_cast<char*>(buffer.data()),
            static_cast<std::streamsize>(num * static_cast<size_t>(dim)));
    if (!in) throw std::runtime_error("Failed while reading uint8 bin payload: " + filename);

    for (size_t i = 0; i < num * static_cast<size_t>(dim); ++i) {
        data[i] = static_cast<float>(buffer[i]);
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

void load_bin_int32(const std::string& filename, std::vector<int>& data, int& num, int& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open int32 bin ground-truth file: " + filename);
    }

    int32_t num_i32 = 0, dim_i32 = 0;
    in.read(reinterpret_cast<char*>(&num_i32), 4);
    in.read(reinterpret_cast<char*>(&dim_i32), 4);
    if (!in) throw std::runtime_error("Failed to read int32 bin header: " + filename);

    num = static_cast<int>(num_i32);
    dim = static_cast<int>(dim_i32);
    data.resize(static_cast<size_t>(num) * static_cast<size_t>(dim));

    in.read(reinterpret_cast<char*>(data.data()),
            static_cast<std::streamsize>(static_cast<size_t>(num) * static_cast<size_t>(dim) * sizeof(int32_t)));
    if (!in) throw std::runtime_error("Failed while reading int32 bin payload: " + filename);
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

void load_ground_truth_file(const std::string& filename,
                            GroundTruthFileType file_type,
                            std::vector<int>& data,
                            int& num,
                            int& dim) {
    switch (file_type) {
        case GroundTruthFileType::IVECS:
            load_ground_truth_ivecs(filename, data, num, dim);
            break;
        case GroundTruthFileType::BIN_INT32:
            load_bin_int32(filename, data, num, dim);
            break;
        default:
            throw std::runtime_error("Unsupported ground-truth file type.");
    }
}

inline float exact_l2_sq(const float* a, const float* b, int dim) {
    float dist = 0.0f;
    for (int i = 0; i < dim; ++i) {
        const float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

inline float exact_inner_product(const float* a, const float* b, int dim) {
    float score = 0.0f;
    for (int i = 0; i < dim; ++i) {
        score += a[i] * b[i];
    }
    return score;
}

static bool dataset_uses_inner_product(DatasetKind dataset) {
    return dataset == DatasetKind::Trip ||
           dataset == DatasetKind::MsMarco ||
           dataset == DatasetKind::Laion;
}

static int parse_first_int_before_tab(const std::string& line) {
    const size_t pos = line.find('\t');
    const std::string token = (pos == std::string::npos) ? line : line.substr(0, pos);
    return std::stoi(token);
}

static std::string strip_record_newline(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
        s.pop_back();
    }
    return s;
}

static int parse_first_int_field_from_tsv_record(const std::string& record) {
    const size_t pos = record.find('\t');
    std::string token = (pos == std::string::npos) ? record : record.substr(0, pos);

    size_t l = 0;
    while (l < token.size() && std::isspace(static_cast<unsigned char>(token[l]))) {
        ++l;
    }

    size_t r = token.size();
    while (r > l && std::isspace(static_cast<unsigned char>(token[r - 1]))) {
        --r;
    }

    if (l >= r) {
        throw std::runtime_error("Empty first field in TSV record.");
    }

    return std::stoi(token.substr(l, r - l));
}

static void load_collection_offset_to_pid(const std::string& filename, std::vector<int>& offset_to_pid) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open collection.tsv: " + filename);
    }

    offset_to_pid.clear();

    std::string record;
    record.reserve(4096);

    bool in_quotes = false;
    bool at_field_start = true;

    auto flush_record = [&]() {
        std::string rec = strip_record_newline(record);
        record.clear();

        if (rec.empty()) {
            at_field_start = true;
            return;
        }

        offset_to_pid.push_back(parse_first_int_field_from_tsv_record(rec));
        at_field_start = true;
    };

    char c = 0;
    while (in.get(c)) {
        if (c == '"') {
            if (in_quotes) {
                if (in.peek() == '"') {

                    record.push_back(c);
                    in.get(c);
                    record.push_back(c);
                    at_field_start = false;
                    continue;
                }

                in_quotes = false;
                record.push_back(c);
                at_field_start = false;
                continue;
            }

            if (at_field_start) {

                in_quotes = true;
                record.push_back(c);
                at_field_start = false;
                continue;
            }
        }

        if (c == '\n' && !in_quotes) {
            flush_record();
            continue;
        }

        record.push_back(c);

        if (!in_quotes) {
            if (c == '\t') {
                at_field_start = true;
            } else if (c != '\r') {
                at_field_start = false;
            }
        }
    }

    if (!record.empty()) {
        flush_record();
    }
}

static void load_query_offset_to_qid(const std::string& filename, std::vector<int>& offset_to_qid) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open queries.dev.small.tsv: " + filename);
    }

    offset_to_qid.clear();

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        offset_to_qid.push_back(parse_first_int_before_tab(line));
    }
}

static void load_qrels_tsv(
    const std::string& filename,
    std::unordered_map<int, std::unordered_set<int>>& qrels
) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open qrels.dev.small.tsv: " + filename);
    }

    qrels.clear();

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        int qid = 0;
        int none_field = 0;
        int pid = 0;
        int relevance_score = 0;
        if (!(iss >> qid >> none_field >> pid >> relevance_score)) {
            throw std::runtime_error("Malformed qrels line: " + line);
        }

        auto& rel_set = qrels[qid];
        if (relevance_score > 0) {
            rel_set.insert(pid);
        }
    }
}

static MsMarcoCompassMeta load_msmarco_compass_meta(
    const std::string& collection_tsv,
    const std::string& queries_tsv,
    const std::string& qrels_tsv
) {
    MsMarcoCompassMeta meta;
    load_collection_offset_to_pid(collection_tsv, meta.offset_to_pid);
    load_query_offset_to_qid(queries_tsv, meta.offset_to_qid);
    load_qrels_tsv(qrels_tsv, meta.qrels);
    return meta;
}

static constexpr uint32_t PROTOCOL_MAGIC = 0x51425443;
static constexpr uint16_t PROTOCOL_VERSION = 1;

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

enum class GroundTruthMrrMode {
    FirstGroundTruth,
    Qrels,
};

struct RuntimeOptions {

    DatasetKind dataset = DatasetKind::Sift;

    bool use_query_batch = true;
    int query_batch_size = 1;
    size_t test_queries = 100;

    EvalMetric eval_metric = EvalMetric::Recall;
    bool metric_overridden = false;

    int hash_bits = 8192;
    int num_tables = 64;
    int top_k = 10;
    int candidate_pool = 2000;

    float p_base = 0.08f;
    float p_inst = 0.00f;

    int omp_threads = 0;

    std::string server_ip = "127.0.0.1";
    int port = 9080;

    std::string query_path_override;
    std::string gt_path_override;
    std::string collection_tsv_override;
    std::string queries_tsv_override;
    std::string qrels_tsv_override;
};

struct ClientDatasetConfig {
    DatasetKind dataset;
    std::string name;
    std::string dataset_root;
    std::string query_path;
    std::string gt_path;
    std::string collection_tsv;
    std::string queries_tsv;
    std::string qrels_tsv;

    FeatureFileType query_file_type;
    GroundTruthFileType gt_file_type;
    EvalMetric default_metric;
    GroundTruthMrrMode mrr_mode;
};

static std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return s;
}

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

static const char* eval_metric_to_string(EvalMetric metric) {
    switch (metric) {
        case EvalMetric::Recall: return "Recall";
        case EvalMetric::MRR: return "MRR";
        default: return "Unknown";
    }
}

static ClientDatasetConfig make_client_dataset_config(DatasetKind dataset) {
    switch (dataset) {
        case DatasetKind::Sift:
            return {
                DatasetKind::Sift,
                "sift",
                mess::data_path("dataset/sift"),
                mess::data_path("dataset/sift/query.fvecs"),
                mess::data_path("dataset/sift/gt.ivecs"),
                "",
                "",
                "",
                FeatureFileType::FVECS,
                GroundTruthFileType::IVECS,
                EvalMetric::Recall,
                GroundTruthMrrMode::FirstGroundTruth
            };
        case DatasetKind::Laion:
            return {
                DatasetKind::Laion,
                "laion",
                mess::data_path("dataset/laion1m"),
                mess::data_path("dataset/laion1m/laion_query.fvecs"),
                mess::data_path("dataset/laion1m/100k/gt.ivecs"),
                "",
                "",
                "",
                FeatureFileType::FVECS,
                GroundTruthFileType::IVECS,
                EvalMetric::Recall,
                GroundTruthMrrMode::FirstGroundTruth
            };
        case DatasetKind::Trip:
            return {
                DatasetKind::Trip,
                "trip",
                mess::data_path("dataset/trip_distilbert"),
                mess::data_path("dataset/trip_distilbert/queries.fvecs"),
                mess::data_path("dataset/trip_distilbert/gt_10.ivecs"),
                mess::data_path("dataset/trip_distilbert/benchmark_tsv/documents/docs.tsv"),
                mess::data_path("dataset/trip_distilbert/benchmark_tsv/topics/topics.head.val.tsv"),
                mess::data_path("dataset/trip_distilbert/benchmark_tsv/qrels/qrels.dctr.head.val.tsv"),
                FeatureFileType::FVECS,
                GroundTruthFileType::IVECS,
                EvalMetric::MRR,
                GroundTruthMrrMode::Qrels
            };
        case DatasetKind::MsMarco:
            return {
                DatasetKind::MsMarco,
                "msmarco",
                mess::data_path("dataset/msmarco_bert"),
                mess::data_path("dataset/msmarco_bert/queries.fvecs"),
                mess::data_path("dataset/msmarco_bert/gt_10.ivecs"),
                mess::data_path("dataset/msmarco_bert/passages/collection.tsv"),
                mess::data_path("dataset/msmarco_bert/passages/queries.dev.small.tsv"),
                mess::data_path("dataset/msmarco_bert/passages/qrels.dev.small.tsv"),
                FeatureFileType::FVECS,
                GroundTruthFileType::IVECS,
                EvalMetric::MRR,
                GroundTruthMrrMode::Qrels
            };
        case DatasetKind::Sift100M:
            return {
                DatasetKind::Sift100M,
                "sift100m",
                mess::data_path("dataset/sift100m"),
                mess::data_path("dataset/sift100m/query.bin"),
                mess::data_path("dataset/sift100m/gt.bin"),
                "",
                "",
                "",
                FeatureFileType::BIN_UINT8,
                GroundTruthFileType::BIN_INT32,
                EvalMetric::Recall,
                GroundTruthMrrMode::FirstGroundTruth
            };
    }
    throw std::runtime_error("Unsupported dataset kind.");
}

static void apply_client_overrides(ClientDatasetConfig& cfg, const RuntimeOptions& opt) {
    if (!opt.query_path_override.empty()) cfg.query_path = opt.query_path_override;
    if (!opt.gt_path_override.empty()) cfg.gt_path = opt.gt_path_override;
    if (!opt.collection_tsv_override.empty()) cfg.collection_tsv = opt.collection_tsv_override;
    if (!opt.queries_tsv_override.empty()) cfg.queries_tsv = opt.queries_tsv_override;
    if (!opt.qrels_tsv_override.empty()) cfg.qrels_tsv = opt.qrels_tsv_override;
}

static void print_client_usage(const char* prog) {
    std::cout << "Usage: " << prog
              << " [--dataset sift|laion|trip|msmarco|sift100m]"
              << " [--metric recall|mrr] [--test_queries N]"
              << " [--candidate_pool N] [--hash_bits N] [--num_tables N]"
              << " [--p_base X] [--p_inst X]"
              << " [--server_ip IP] [--port N]"
              << " [--use_query_batch 0|1] [--query_batch_size N]"
              << " [--query_path PATH] [--gt_path PATH]"
              << " [--collection_tsv PATH] [--queries_tsv PATH] [--qrels_tsv PATH]"
              << " [--omp_threads N]" << std::endl;
}

static RuntimeOptions parse_runtime_options(int argc, char** argv) {
    RuntimeOptions opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) {
            opt.dataset = parse_dataset_kind(argv[++i]);
        } else if (arg == "--metric" && i + 1 < argc) {
            const std::string metric = to_lower_copy(argv[++i]);
            opt.metric_overridden = true;
            if (metric == "recall") {
                opt.eval_metric = EvalMetric::Recall;
            } else if (metric == "mrr") {
                opt.eval_metric = EvalMetric::MRR;
            } else {
                throw std::runtime_error("Unsupported --metric value: " + metric + " (use recall or mrr)");
            }
        } else if (arg == "--test_queries" && i + 1 < argc) {
            opt.test_queries = static_cast<size_t>(std::max(1, std::stoi(argv[++i])));
        } else if (arg == "--candidate_pool" && i + 1 < argc) {
            opt.candidate_pool = std::stoi(argv[++i]);
        } else if (arg == "--top_k" && i + 1 < argc) {
            opt.top_k = std::stoi(argv[++i]);
        } else if (arg == "--hash_bits" && i + 1 < argc) {
            opt.hash_bits = std::stoi(argv[++i]);
        } else if (arg == "--num_tables" && i + 1 < argc) {
            opt.num_tables = std::stoi(argv[++i]);
        } else if (arg == "--p_base" && i + 1 < argc) {
            opt.p_base = std::stof(argv[++i]);
        } else if (arg == "--p_inst" && i + 1 < argc) {
            opt.p_inst = std::stof(argv[++i]);
        } else if (arg == "--use_query_batch" && i + 1 < argc) {
            opt.use_query_batch = (std::stoi(argv[++i]) != 0);
        } else if (arg == "--query_batch_size" && i + 1 < argc) {
            opt.query_batch_size = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--server_ip" && i + 1 < argc) {
            opt.server_ip = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            opt.port = std::stoi(argv[++i]);
        } else if (arg == "--omp_threads" && i + 1 < argc) {
            opt.omp_threads = std::stoi(argv[++i]);
        } else if (arg == "--query_path" && i + 1 < argc) {
            opt.query_path_override = argv[++i];
        } else if (arg == "--gt_path" && i + 1 < argc) {
            opt.gt_path_override = argv[++i];
        } else if (arg == "--collection_tsv" && i + 1 < argc) {
            opt.collection_tsv_override = argv[++i];
        } else if (arg == "--queries_tsv" && i + 1 < argc) {
            opt.queries_tsv_override = argv[++i];
        } else if (arg == "--qrels_tsv" && i + 1 < argc) {
            opt.qrels_tsv_override = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_client_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
    }
    return opt;
}

struct ScoredCandidate {
    float dist;
    int label;
};

static int recall_hits_at_k(
    const std::vector<ScoredCandidate>& ranked_results,
    const std::unordered_set<int>& relevant_labels,
    int k
) {
    const size_t limit = std::min(ranked_results.size(), static_cast<size_t>(k));
    int hits = 0;
    for (size_t rank = 0; rank < limit; ++rank) {
        if (relevant_labels.count(ranked_results[rank].label)) {
            ++hits;
        }
    }
    return hits;
}

static double reciprocal_rank_first_ground_truth(
    const std::vector<ScoredCandidate>& ranked_results,
    const std::vector<int>& gt_data,
    int gt_dim,
    size_t query_offset,
    int k
) {
    if (gt_dim <= 0) return 0.0;
    const size_t gt_idx = query_offset * static_cast<size_t>(gt_dim);
    if (gt_idx >= gt_data.size()) return 0.0;

    const int first_gt = gt_data[gt_idx];
    const size_t limit = std::min(ranked_results.size(), static_cast<size_t>(k));
    for (size_t rank = 0; rank < limit; ++rank) {
        if (ranked_results[rank].label == first_gt) {
            return 1.0 / static_cast<double>(rank + 1);
        }
    }
    return 0.0;
}

static double reciprocal_rank_msmarco_compass(
    const std::vector<ScoredCandidate>& ranked_results,
    size_t query_offset,
    const MsMarcoCompassMeta& meta,
    int k
) {
    if (query_offset >= meta.offset_to_qid.size()) {
        throw std::runtime_error("query offset out of range for queries.dev.small.tsv");
    }

    const int qid = meta.offset_to_qid[query_offset];
    const auto it = meta.qrels.find(qid);
    if (it == meta.qrels.end()) {

        return -1.0;
    }

    const auto& relevant_pids = it->second;
    const size_t limit = std::min(ranked_results.size(), static_cast<size_t>(k));

    for (size_t rank = 0; rank < limit; ++rank) {
        const int passage_offset = ranked_results[rank].label;
        if (passage_offset < 0 ||
            static_cast<size_t>(passage_offset) >= meta.offset_to_pid.size()) {
            continue;
        }

        const int pid = meta.offset_to_pid[static_cast<size_t>(passage_offset)];
        if (relevant_pids.count(pid)) {
            return 1.0 / static_cast<double>(rank + 1);
        }
    }
    return 0.0;
}

static std::vector<ScoredCandidate> rerank_single_query(
    const float* raw_query,
    const std::vector<std::pair<int, std::vector<uint8_t>>>& candidates,
    SecureSearchContext& payload_secure_ctx,
    int input_dim,
    int k,
    bool use_inner_product,
    long long& rerank_us_out
) {
    const auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<ScoredCandidate> scored(candidates.size());
    #pragma omp parallel for schedule(dynamic)
    for (int64_t c = 0; c < static_cast<int64_t>(candidates.size()); ++c) {
        const std::vector<float> plain_vec =
            payload_secure_ctx.decrypt_vector(0, candidates[static_cast<size_t>(c)].second, input_dim);
        const float exact_value = use_inner_product
            ? exact_inner_product(raw_query, plain_vec.data(), input_dim)
            : exact_l2_sq(raw_query, plain_vec.data(), input_dim);
        scored[static_cast<size_t>(c)] = {exact_value, candidates[static_cast<size_t>(c)].first};
    }

    std::unordered_map<int, float> best_value_by_label;
    best_value_by_label.reserve(scored.size());
    for (const auto& item : scored) {
        auto it_best = best_value_by_label.find(item.label);
        if (it_best == best_value_by_label.end() ||
            (use_inner_product ? item.dist > it_best->second : item.dist < it_best->second)) {
            best_value_by_label[item.label] = item.dist;
        }
    }

    std::vector<ScoredCandidate> unique_scored;
    unique_scored.reserve(best_value_by_label.size());
    for (const auto& kv : best_value_by_label) {
        unique_scored.push_back({kv.second, kv.first});
    }

    const auto cmp_scored = [use_inner_product](const ScoredCandidate& a, const ScoredCandidate& b) {
        return use_inner_product ? (a.dist > b.dist) : (a.dist < b.dist);
    };

    const size_t keep = std::min(static_cast<size_t>(k), unique_scored.size());
    if (unique_scored.size() > keep) {
        std::nth_element(
            unique_scored.begin(),
            unique_scored.begin() + static_cast<std::ptrdiff_t>(keep),
            unique_scored.end(),
            cmp_scored
        );
        unique_scored.resize(keep);
    }
    std::sort(unique_scored.begin(), unique_scored.end(), cmp_scored);

    const auto t1 = std::chrono::high_resolution_clock::now();
    rerank_us_out += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    return unique_scored;
}

int main(int argc, char** argv) {
    try {
        RuntimeOptions runtime = parse_runtime_options(argc, argv);
        if (runtime.omp_threads > 0) {
            omp_set_num_threads(runtime.omp_threads);
        }

        ClientDatasetConfig data_cfg = make_client_dataset_config(runtime.dataset);
        apply_client_overrides(data_cfg, runtime);
        if (!runtime.metric_overridden) {
            runtime.eval_metric = data_cfg.default_metric;
        }

        const std::string dataset_root = data_cfg.dataset_root;
        const std::string query_path = data_cfg.query_path;
        const std::string gt_path = data_cfg.gt_path;
        const FeatureFileType query_file_type = data_cfg.query_file_type;
        const GroundTruthFileType gt_file_type = data_cfg.gt_file_type;

        const int hash_bits = runtime.hash_bits;
        const int num_tables = runtime.num_tables;
        if (num_tables <= 0 || hash_bits <= 0 || hash_bits % (num_tables * 8) != 0) {
            throw std::runtime_error("hash_bits must be divisible by num_tables * 8");
        }
        const int sub_hash_bits = hash_bits / num_tables;
        const int sub_hash_bytes = sub_hash_bits / 8;
        const int k = runtime.top_k;
        const int candidate_pool = runtime.candidate_pool;
        const int per_shard_pool = std::max(1, candidate_pool / num_tables);
        const float p_base = runtime.p_base;
        const float p_inst = runtime.p_inst;

        std::vector<float> query_data;
        std::vector<int> gt_data;
        size_t query_num = 0;
        int query_dim = 0, gt_num = 0, gt_dim = 0;

        load_feature_file(query_path, query_file_type, query_data, query_num, query_dim);

        MsMarcoCompassMeta qrels_meta;
        bool use_qrels_mrr = false;
        if (runtime.eval_metric == EvalMetric::Recall) {
            load_ground_truth_file(gt_path, gt_file_type, gt_data, gt_num, gt_dim);
        } else if (data_cfg.mrr_mode == GroundTruthMrrMode::Qrels) {
            qrels_meta = load_msmarco_compass_meta(
                data_cfg.collection_tsv,
                data_cfg.queries_tsv,
                data_cfg.qrels_tsv
            );
            if (qrels_meta.offset_to_qid.size() < query_num) {
                throw std::runtime_error("query TSV has fewer queries than query vectors.");
            }
            use_qrels_mrr = true;
        } else {
            load_ground_truth_file(gt_path, gt_file_type, gt_data, gt_num, gt_dim);
        }

        const int input_dim = query_dim;

        std::cout << "Dataset name: " << dataset_kind_to_string(data_cfg.dataset) << std::endl;
        std::cout << "Dataset root: " << dataset_root << std::endl;
        std::cout << "Query path: " << query_path << std::endl;
        if (runtime.eval_metric == EvalMetric::Recall || !use_qrels_mrr) {
            std::cout << "Ground-truth path: " << gt_path << std::endl;
        } else {
            std::cout << "Collection TSV: " << data_cfg.collection_tsv << std::endl;
            std::cout << "Queries TSV: " << data_cfg.queries_tsv << std::endl;
            std::cout << "Qrels TSV: " << data_cfg.qrels_tsv << std::endl;
        }
        std::cout << "Query vectors: " << query_num << ", Dim=" << input_dim << std::endl;
        std::cout << "Server: " << runtime.server_ip << ":" << runtime.port << std::endl;
        std::cout << "Client rerank metric: "
                  << (dataset_uses_inner_product(data_cfg.dataset) ? "INNER_PRODUCT (descending)" : "L2 (ascending)")
                  << std::endl;

        LSHEncoder encoder(input_dim, hash_bits, HashMethod::ISOHASH, dataset_root);

        SearchContext search_cfg;
        search_cfg.k = k;
        search_cfg.ef_search = per_shard_pool;
        search_cfg.verbose = true;

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
        SecureSearchContext payload_secure_ctx(payload_cfg);

        int sockfd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) throw std::runtime_error("socket() failed");
        int flag = 1;
        ::setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
        int sndbuf = 4 * 1024 * 1024;
        int rcvbuf = 4 * 1024 * 1024;
        ::setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));
        ::setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

        sockaddr_in serv{};
        serv.sin_family = AF_INET;
        serv.sin_port = htons(static_cast<uint16_t>(runtime.port));
        if (::inet_pton(AF_INET, runtime.server_ip.c_str(), &serv.sin_addr) <= 0) {
            throw std::runtime_error("invalid --server_ip: " + runtime.server_ip);
        }
        if (::connect(sockfd, reinterpret_cast<sockaddr*>(&serv), sizeof(serv)) < 0) {
            throw std::runtime_error("connect() failed");
        }

        std::vector<HashCodeword> pure_query_codes = encoder.batch_encode(query_data.data(), query_num);
        std::vector<HashCodeword> query_codes(query_num);
        std::vector<std::vector<uint8_t>> query_shard_bytes(
            query_num,
            std::vector<uint8_t>(static_cast<size_t>(num_tables) * static_cast<size_t>(sub_hash_bytes))
        );

        #pragma omp parallel for schedule(dynamic)
        for (int64_t i = 0; i < static_cast<int64_t>(query_num); ++i) {
            HashCodeword q = pure_query_codes[static_cast<size_t>(i)];
            q.inject_noise(p_base);
            q.inject_noise(p_inst);
            query_codes[static_cast<size_t>(i)] = q;
            for (int s = 0; s < num_tables; ++s) {
                auto sub = q.split(s, num_tables);
                const auto bytes = sub.getBytes();
                std::memcpy(query_shard_bytes[static_cast<size_t>(i)].data() + static_cast<size_t>(s) * static_cast<size_t>(sub_hash_bytes),
                            bytes.data(),
                            static_cast<size_t>(sub_hash_bytes));
            }
        }

        std::cout << "\n>>> Client Phase 2: Search & Decrypt-Rerank" << std::endl;
        std::cout << "QueryBatch enabled: " << (runtime.use_query_batch ? "YES" : "NO")
                  << ", batch_size=" << (runtime.use_query_batch ? runtime.query_batch_size : 1) << std::endl;
        std::cout << "Evaluation metric: " << eval_metric_to_string(runtime.eval_metric)
                  << "@" << k << std::endl;

        long long recall_hit_count = 0;
        long long recall_total_count = 0;
        double mrr_sum = 0.0;
        long long evaluated_queries = 0;
        long long skipped_queries = 0;

        long long total_e2e_us = 0;
        long long total_server_us = 0;
        long long total_recv_us = 0;
        long long total_rerank_us = 0;

        uint64_t total_request_bytes = 0;
        uint64_t total_response_bytes = 0;

        size_t test_queries = (runtime.test_queries > 0)
            ? std::min<size_t>(query_num, runtime.test_queries)
            : query_num;
        if (runtime.eval_metric == EvalMetric::MRR) {
            if (use_qrels_mrr) {
                test_queries = std::min(test_queries, qrels_meta.offset_to_qid.size());
            } else {
                test_queries = std::min(test_queries, static_cast<size_t>(gt_num));
            }
        } else {
            test_queries = std::min(test_queries, static_cast<size_t>(gt_num));
        }

        const size_t batch_size = runtime.use_query_batch
            ? static_cast<size_t>(std::max(1, runtime.query_batch_size))
            : static_cast<size_t>(1);

        for (size_t q0 = 0; q0 < test_queries; q0 += batch_size) {
            const size_t cur_batch = std::min(batch_size, test_queries - q0);
            QueryBatchRequestHeader req{};
            req.magic = PROTOCOL_MAGIC;
            req.version = PROTOCOL_VERSION;
            req.flags = runtime.use_query_batch ? 1 : 0;
            req.batch_size = static_cast<int32_t>(cur_batch);
            req.top_k = k;
            req.per_shard_pool = per_shard_pool;
            req.num_tables = num_tables;
            req.sub_hash_bytes = sub_hash_bytes;
            req.body_bytes = static_cast<int32_t>(
                cur_batch * static_cast<size_t>(num_tables) * static_cast<size_t>(sub_hash_bytes)
            );

            std::vector<uint8_t> req_buf;
            req_buf.reserve(sizeof(QueryBatchRequestHeader) + static_cast<size_t>(req.body_bytes));
            append_pod(req_buf, req);
            for (size_t b = 0; b < cur_batch; ++b) {
                const auto& bytes = query_shard_bytes[q0 + b];
                req_buf.insert(req_buf.end(), bytes.begin(), bytes.end());
            }

            total_request_bytes += static_cast<uint64_t>(req_buf.size());

            const auto t0 = std::chrono::high_resolution_clock::now();
            if (!send_all(sockfd, req_buf.data(), req_buf.size())) {
                throw std::runtime_error("send batch request failed");
            }

            QueryBatchResponseHeader batch_resp{};
            if (!recv_all(sockfd, &batch_resp, sizeof(batch_resp))) {
                throw std::runtime_error("recv batch response header failed");
            }
            if (batch_resp.magic != PROTOCOL_MAGIC || batch_resp.version != PROTOCOL_VERSION) {
                throw std::runtime_error("protocol mismatch in batch response");
            }

            std::vector<uint8_t> resp_body(static_cast<size_t>(batch_resp.body_bytes));
            if (batch_resp.body_bytes > 0) {
                if (!recv_all(sockfd, resp_body.data(), resp_body.size())) {
                    throw std::runtime_error("recv batch response body failed");
                }
            }

            total_response_bytes += static_cast<uint64_t>(sizeof(QueryBatchResponseHeader))
                                  + static_cast<uint64_t>(resp_body.size());

            const auto t1 = std::chrono::high_resolution_clock::now();

            size_t off = 0;
            for (size_t b = 0; b < cur_batch; ++b) {
                QueryResponseHeader resp = read_pod<QueryResponseHeader>(resp_body, off);
                std::vector<std::pair<int, std::vector<uint8_t>>> candidates;
                candidates.reserve(static_cast<size_t>(resp.candidate_count));
                const size_t q_body_end = off + static_cast<size_t>(resp.body_bytes);
                if (q_body_end > resp_body.size()) {
                    throw std::runtime_error("corrupted batch body: query body out of range");
                }

                for (int c = 0; c < resp.candidate_count; ++c) {
                    CandidateHeader ch = read_pod<CandidateHeader>(resp_body, off);
                    if (off + static_cast<size_t>(ch.cipher_size) > q_body_end) {
                        throw std::runtime_error("corrupted batch body: payload out of range");
                    }
                    std::vector<uint8_t> cipher(static_cast<size_t>(ch.cipher_size));
                    std::memcpy(cipher.data(), resp_body.data() + off, static_cast<size_t>(ch.cipher_size));
                    off += static_cast<size_t>(ch.cipher_size);
                    candidates.push_back({ch.label, std::move(cipher)});
                }
                if (off != q_body_end) off = q_body_end;

                long long rerank_us_local = 0;
                auto unique_scored = rerank_single_query(
                    query_data.data() + (q0 + b) * static_cast<size_t>(input_dim),
                    candidates,
                    payload_secure_ctx,
                    input_dim,
                    k,
                    dataset_uses_inner_product(data_cfg.dataset),
                    rerank_us_local
                );
                total_rerank_us += rerank_us_local;

                if (runtime.eval_metric == EvalMetric::Recall) {
                    std::unordered_set<int> true_neighbors;
                    const int max_gt = std::min(k, gt_dim);
                    for (int j = 0; j < max_gt; ++j) {
                        const size_t idx = (q0 + b) * static_cast<size_t>(gt_dim) + static_cast<size_t>(j);
                        if (idx < gt_data.size()) {
                            true_neighbors.insert(gt_data[idx]);
                        }
                    }
                    const int recall_hits = recall_hits_at_k(unique_scored, true_neighbors, k);
                    recall_hit_count += recall_hits;
                    recall_total_count += max_gt;
                } else {
                    double rr = 0.0;
                    if (use_qrels_mrr) {
                        rr = reciprocal_rank_msmarco_compass(
                            unique_scored,
                            q0 + b,
                            qrels_meta,
                            k
                        );
                    } else {
                        rr = reciprocal_rank_first_ground_truth(
                            unique_scored,
                            gt_data,
                            gt_dim,
                            q0 + b,
                            k
                        );
                    }
                    if (rr >= 0.0) {
                        mrr_sum += rr;
                        evaluated_queries++;
                    } else {
                        skipped_queries++;
                    }
                }

                total_server_us += static_cast<long long>(resp.server_compute_us);
            }

            const auto t2 = std::chrono::high_resolution_clock::now();
            total_recv_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            total_e2e_us += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t0).count();
        }

        const double recall = (recall_total_count > 0)
            ? static_cast<double>(recall_hit_count) / static_cast<double>(recall_total_count)
            : 0.0;

        const double mrr = (evaluated_queries > 0)
            ? mrr_sum / static_cast<double>(evaluated_queries)
            : 0.0;

        std::cout << "\n>>> Phase 3: Evaluation" << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "Config: " << num_tables
                  << " Shards | " << hash_bits
                  << " bits LSH | Pool=" << candidate_pool
                  << " | Top-K=" << k << std::endl;

        if (runtime.eval_metric == EvalMetric::Recall) {
            std::cout << "Recall@" << k << ": "
                      << std::fixed << std::setprecision(2)
                      << recall * 100.0 << "%" << std::endl;
            std::cout << "Recall evaluated queries: " << test_queries
                      << ", total ground-truth labels: " << recall_total_count << std::endl;
        } else {
            std::cout << "MRR@" << k << ": "
                      << std::fixed << std::setprecision(6)
                      << mrr << std::endl;
            std::cout << "MRR evaluated queries: " << evaluated_queries
                      << ", skipped: " << skipped_queries << std::endl;
        }

        if (test_queries > 0) {
            const double search_time_ms =
                static_cast<double>(total_e2e_us) / static_cast<double>(test_queries) / 1000.0;

            const double server_eval_ms =
                static_cast<double>(total_server_us) / static_cast<double>(test_queries) / 1000.0;

            const double client_dec_rank_ms =
                static_cast<double>(total_rerank_us) / static_cast<double>(test_queries) / 1000.0;

            double trans_latency_ms = search_time_ms - server_eval_ms - client_dec_rank_ms;
            if (trans_latency_ms < 0.0) {
                trans_latency_ms = 0.0;
            }

            const uint64_t total_comm_bytes = total_request_bytes + total_response_bytes;
            const double request_kb_per_query =
                static_cast<double>(total_request_bytes) / static_cast<double>(test_queries) / 1024.0;
            const double response_kb_per_query =
                static_cast<double>(total_response_bytes) / static_cast<double>(test_queries) / 1024.0;
            const double communication_kb_per_query =
                static_cast<double>(total_comm_bytes) / static_cast<double>(test_queries) / 1024.0;

            std::cout << "Search Time: " << std::fixed << std::setprecision(4)
                      << search_time_ms << " ms/query." << std::endl;

            std::cout << "Server Eval. Time: " << std::fixed << std::setprecision(4)
                      << server_eval_ms << " ms/query." << std::endl;

            std::cout << "Trans. Latency: " << std::fixed << std::setprecision(4)
                      << trans_latency_ms << " ms/query." << std::endl;

            std::cout << "Client Dec.&Rank Time: " << std::fixed << std::setprecision(4)
                      << client_dec_rank_ms << " ms/query." << std::endl;

            std::cout << "Communication: " << std::fixed << std::setprecision(4)
                      << communication_kb_per_query << " KB/query"
                      << " (request=" << request_kb_per_query
                      << " KB/query, response=" << response_kb_per_query
                      << " KB/query)." << std::endl;
        }
        if (total_e2e_us > 0) {
            std::cout << "QPS: " << std::fixed << std::setprecision(2)
                      << (1000000.0 * static_cast<double>(test_queries)) / static_cast<double>(total_e2e_us)
                      << std::endl;
        }
        std::cout << "------------------------------------------------" << std::endl;

        ::close(sockfd);
    } catch (const std::exception& e) {
        std::cerr << "CLIENT ERROR: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
