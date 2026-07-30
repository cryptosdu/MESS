#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cctype>
#include <limits>

#include <sys/stat.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/tcp.h>

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

static std::string trim_copy(const std::string& s) {
    size_t l = 0;
    while (l < s.size() && std::isspace(static_cast<unsigned char>(s[l]))) ++l;
    size_t r = s.size();
    while (r > l && std::isspace(static_cast<unsigned char>(s[r - 1]))) --r;
    return s.substr(l, r - l);
}

enum class DatasetKind {
    Trip,
    MsMarco,
    Sift,
    Laion,
};

enum class EvalMetric {
    Auto,
    MRR,
    Recall,
};

static DatasetKind parse_dataset_kind(const std::string& name) {
    const std::string s = to_lower_copy(name);
    if (s == "trip" || s == "tripclick" || s == "trip_click") return DatasetKind::Trip;
    if (s == "msmarco" || s == "ms_marco" || s == "marco" || s == "mscro") return DatasetKind::MsMarco;
    if (s == "sift") return DatasetKind::Sift;
    if (s == "laion" || s == "laion1m") return DatasetKind::Laion;
    throw std::runtime_error("Unsupported --dataset value: " + name + " (use trip, msmarco, sift, or laion)");
}

static EvalMetric parse_eval_metric(const std::string& name) {
    const std::string s = to_lower_copy(name);
    if (s == "auto") return EvalMetric::Auto;
    if (s == "mrr") return EvalMetric::MRR;
    if (s == "recall") return EvalMetric::Recall;
    throw std::runtime_error("Unsupported --metric value: " + name + " (use auto, mrr, or recall)");
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

static std::string metric_name_for_dataset(DatasetKind dataset) {
    return dataset == DatasetKind::Sift ? "L2" : "INNER_PRODUCT";
}

static std::string eval_metric_name(EvalMetric metric) {
    switch (metric) {
        case EvalMetric::Auto: return "auto";
        case EvalMetric::MRR: return "mrr";
        case EvalMetric::Recall: return "recall";
    }
    return "unknown";
}

struct Paths {
    std::string data_root;
    std::string dataset_name;

    std::string query_fvecs;
    std::string collection_tsv;
    std::string queries_tsv;
    std::string qrels_tsv;
    std::string gt_ivecs;
    std::string output_ivecs;

    EvalMetric default_metric = EvalMetric::Recall;
    size_t default_test_queries = 10000;
};

static Paths make_paths(DatasetKind dataset) {
    Paths p;
    p.data_root = mess::find_data_root().string();
    p.dataset_name = dataset_kind_name(dataset);

    switch (dataset) {
        case DatasetKind::Trip:
            p.query_fvecs = mess::data_path("dataset/trip_distilbert/queries.fvecs");

            p.collection_tsv = mess::data_path("dataset/trip_distilbert/benchmark_tsv/documents/docs.tsv");
            p.queries_tsv = mess::data_path("dataset/trip_distilbert/benchmark_tsv/topics/topics.head.val.tsv");
            p.qrels_tsv = mess::data_path("dataset/trip_distilbert/benchmark_tsv/qrels/qrels.dctr.head.val.tsv");
            p.gt_ivecs = mess::data_path("dataset/trip_distilbert/gt_10.ivecs");
            p.output_ivecs = "hnswlib_trip_ip_top10.ivecs";
            p.default_metric = EvalMetric::MRR;
            p.default_test_queries = 1175;
            break;

        case DatasetKind::MsMarco:
            p.query_fvecs = mess::data_path("dataset/msmarco_bert/queries.fvecs");
            p.collection_tsv = mess::data_path("dataset/msmarco_bert/passages/collection.tsv");
            p.queries_tsv = mess::data_path("dataset/msmarco_bert/passages/queries.dev.small.tsv");
            p.qrels_tsv = mess::data_path("dataset/msmarco_bert/passages/qrels.dev.small.tsv");
            p.gt_ivecs = mess::data_path("dataset/msmarco_bert/gt_10.ivecs");
            p.output_ivecs = "hnswlib_msmarco_ip_top10.ivecs";
            p.default_metric = EvalMetric::MRR;
            p.default_test_queries = 6980;
            break;

        case DatasetKind::Sift:
            p.query_fvecs = mess::data_path("dataset/sift/query.fvecs");
            p.gt_ivecs = mess::data_path("dataset/sift/gt.ivecs");
            p.output_ivecs = "hnswlib_sift_l2_split_top10.ivecs";
            p.default_metric = EvalMetric::Recall;
            p.default_test_queries = 10000;
            break;

        case DatasetKind::Laion:
            p.query_fvecs = mess::data_path("dataset/laion1m/laion_query.fvecs");
            p.gt_ivecs = mess::data_path("dataset/laion1m/100k/gt.ivecs");
            p.output_ivecs = "hnswlib_laion_ip_top10.ivecs";
            p.default_metric = EvalMetric::Recall;
            p.default_test_queries = 1000;
            break;
    }
    return p;
}

static void require_file(const std::string& path, const std::string& name) {
    if (!file_exists(path)) throw std::runtime_error("Cannot find " + name + ": " + path);
}

struct RuntimeOptions {
    DatasetKind dataset = DatasetKind::Sift;
    EvalMetric eval_metric = EvalMetric::Auto;
    bool output_ivecs_overridden = false;

    int top_k = 10;
    int ef = 20;
    size_t test_queries = 0;
    size_t search_batch_size = 100;
    int threads = 16;

    std::string server_ip = "127.0.0.1";
    int port = 9080;

    std::string output_ivecs;
    std::string query_fvecs_override;
    std::string collection_tsv_override;
    std::string queries_tsv_override;
    std::string qrels_tsv_override;
    std::string gt_ivecs_override;
};

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog
              << " [--dataset trip|msmarco|sift|laion]"
              << " [--metric auto|mrr|recall]"
              << " [--k N] [--ef N] [--test_queries N] [--search_batch_size N]"
              << " [--threads N] [--server_ip IP] [--port N]"
              << " [--output_ivecs PATH]"
              << " [--query_fvecs PATH] [--collection_tsv PATH] [--queries_tsv PATH]"
              << " [--qrels_tsv PATH] [--gt_ivecs PATH]"
              << std::endl;
}

static RuntimeOptions parse_args(int argc, char** argv) {
    RuntimeOptions opt;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) {
            opt.dataset = parse_dataset_kind(argv[++i]);
        } else if (arg == "--metric" && i + 1 < argc) {
            opt.eval_metric = parse_eval_metric(argv[++i]);
        } else if ((arg == "--k" || arg == "--top_k") && i + 1 < argc) {
            opt.top_k = std::stoi(argv[++i]);
        } else if ((arg == "--ef" || arg == "--efSearch" || arg == "--ef_search") && i + 1 < argc) {
            opt.ef = std::stoi(argv[++i]);
        } else if (arg == "--test_queries" && i + 1 < argc) {
            opt.test_queries = static_cast<size_t>(std::max(1, std::stoi(argv[++i])));
        } else if (arg == "--search_batch_size" && i + 1 < argc) {
            opt.search_batch_size = static_cast<size_t>(std::max(1, std::stoi(argv[++i])));
        } else if ((arg == "--threads" || arg == "--omp_threads") && i + 1 < argc) {
            opt.threads = std::stoi(argv[++i]);
        } else if (arg == "--server_ip" && i + 1 < argc) {
            opt.server_ip = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            opt.port = std::stoi(argv[++i]);
        } else if (arg == "--output_ivecs" && i + 1 < argc) {
            opt.output_ivecs = argv[++i];
            opt.output_ivecs_overridden = true;
        } else if (arg == "--query_fvecs" && i + 1 < argc) {
            opt.query_fvecs_override = argv[++i];
        } else if (arg == "--collection_tsv" && i + 1 < argc) {
            opt.collection_tsv_override = argv[++i];
        } else if (arg == "--queries_tsv" && i + 1 < argc) {
            opt.queries_tsv_override = argv[++i];
        } else if (arg == "--qrels_tsv" && i + 1 < argc) {
            opt.qrels_tsv_override = argv[++i];
        } else if (arg == "--gt_ivecs" && i + 1 < argc) {
            opt.gt_ivecs_override = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
    }

    if (opt.top_k <= 0) opt.top_k = 10;
    if (opt.ef <= 0) opt.ef = opt.top_k;
    opt.ef = std::max(opt.ef, opt.top_k);
    if (opt.search_batch_size == 0) opt.search_batch_size = 100;
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

    std::cout << "[Load] " << name << ": " << filename << ", num=" << num << ", dim=" << dim << std::endl;
    const auto t0 = std::chrono::high_resolution_clock::now();
    const size_t report_every = std::max<size_t>(1, num / 20);

    for (size_t i = 0; i < num; ++i) {
        int cur_dim = 0;
        in.read(reinterpret_cast<char*>(&cur_dim), 4);
        if (!in || cur_dim != dim) throw std::runtime_error("Corrupted fvecs or dim mismatch: " + filename);
        in.read(reinterpret_cast<char*>(data.data() + i * static_cast<size_t>(dim)),
                static_cast<std::streamsize>(dim * sizeof(float)));
        if (!in) throw std::runtime_error("Failed while reading fvecs data: " + filename);
        if ((i + 1) % report_every == 0 || i + 1 == num) {
            print_progress("Load " + name, i + 1, num, t0);
        }
    }
}

static void load_ivecs(const std::string& filename, std::vector<int>& data, size_t& num, int& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Cannot open ivecs file: " + filename);

    in.read(reinterpret_cast<char*>(&dim), 4);
    if (!in) throw std::runtime_error("Failed to read ivecs dim: " + filename);

    in.seekg(0, std::ios::end);
    const size_t file_size = static_cast<size_t>(in.tellg());
    const size_t item_size = 4 + static_cast<size_t>(dim) * sizeof(int32_t);
    if (item_size == 0 || file_size % item_size != 0) {
        throw std::runtime_error("Invalid ivecs file size: " + filename);
    }

    num = file_size / item_size;
    data.resize(num * static_cast<size_t>(dim));
    in.seekg(0, std::ios::beg);

    for (size_t i = 0; i < num; ++i) {
        int cur_dim = 0;
        in.read(reinterpret_cast<char*>(&cur_dim), 4);
        if (!in || cur_dim != dim) throw std::runtime_error("Corrupted ivecs or dim mismatch: " + filename);
        in.read(reinterpret_cast<char*>(data.data() + i * static_cast<size_t>(dim)),
                static_cast<std::streamsize>(dim * sizeof(int32_t)));
        if (!in) throw std::runtime_error("Failed while reading ivecs data: " + filename);
    }

    std::cout << "[META] Loaded ivecs ground truth: " << filename
              << ", num=" << num << ", dim=" << dim << std::endl;
}

static bool read_pandas_like_tsv_record(std::istream& in, std::string& record) {
    record.clear();
    bool in_quotes = false;
    bool at_field_start = true;

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

            while (!record.empty() && (record.back() == '\r' || record.back() == '\n')) {
                record.pop_back();
            }
            return true;
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

    while (!record.empty() && (record.back() == '\r' || record.back() == '\n')) {
        record.pop_back();
    }
    return !record.empty();
}

static std::vector<std::string> split_tsv_fields_lightweight(const std::string& record, size_t max_fields = 0) {
    std::vector<std::string> fields;
    std::string cur;
    bool in_quotes = false;
    bool at_field_start = true;

    for (size_t i = 0; i < record.size(); ++i) {
        const char c = record[i];
        if (c == '"') {
            if (in_quotes) {
                if (i + 1 < record.size() && record[i + 1] == '"') {
                    cur.push_back('"');
                    ++i;
                    at_field_start = false;
                    continue;
                }
                in_quotes = false;
                at_field_start = false;
                continue;
            }
            if (at_field_start) {
                in_quotes = true;
                at_field_start = false;
                continue;
            }
        }

        if (c == '\t' && !in_quotes) {
            fields.push_back(cur);
            cur.clear();
            at_field_start = true;
            if (max_fields > 0 && fields.size() + 1 >= max_fields) {
                cur = record.substr(i + 1);
                break;
            }
            continue;
        }

        cur.push_back(c);
        if (!in_quotes && c != '\r') at_field_start = false;
    }
    fields.push_back(cur);
    return fields;
}

static int parse_first_int_field_from_tsv_record(const std::string& record) {
    const size_t pos = record.find('\t');
    const std::string token = trim_copy(pos == std::string::npos ? record : record.substr(0, pos));
    if (token.empty()) {
        throw std::runtime_error("Empty first field in TSV record.");
    }
    return std::stoi(token);
}

struct TextMeta {
    std::vector<int> offset_to_pid;
    std::vector<int> offset_to_qid;
    std::unordered_map<int, std::vector<int>> qrels;
};

static void load_collection_offset_to_pid(const std::string& filename, std::vector<int>& offset_to_pid) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Cannot open collection TSV: " + filename);

    offset_to_pid.clear();
    std::string record;
    while (read_pandas_like_tsv_record(in, record)) {
        if (trim_copy(record).empty()) continue;
        offset_to_pid.push_back(parse_first_int_field_from_tsv_record(record));
    }

    std::cerr << "[META] Loaded passage mapping: " << offset_to_pid.size()
              << " rows from " << filename << std::endl;

    std::cerr << "[META] first 20 offset_to_pid:";
    for (size_t i = 0; i < std::min<size_t>(20, offset_to_pid.size()); ++i) {
        std::cerr << " " << offset_to_pid[i];
    }
    std::cerr << std::endl;
}

static void load_query_offset_to_qid(const std::string& filename, std::vector<int>& offset_to_qid) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Cannot open queries TSV: " + filename);

    offset_to_qid.clear();
    std::string record;
    while (read_pandas_like_tsv_record(in, record)) {
        if (trim_copy(record).empty()) continue;
        offset_to_qid.push_back(parse_first_int_field_from_tsv_record(record));
    }

    std::cerr << "[META] Loaded query mapping: " << offset_to_qid.size()
              << " rows from " << filename << std::endl;
}

static void load_qrels_tsv(const std::string& filename, std::unordered_map<int, std::vector<int>>& qrels) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Cannot open qrels TSV: " + filename);

    qrels.clear();
    std::string record;
    int prev_qid = std::numeric_limits<int>::min();
    size_t rel_pairs = 0;

    while (read_pandas_like_tsv_record(in, record)) {
        if (trim_copy(record).empty()) continue;

        const std::vector<std::string> fields = split_tsv_fields_lightweight(record, 4);
        if (fields.size() < 4) {
            throw std::runtime_error("Malformed qrels TSV record: " + record);
        }

        const int qid = std::stoi(trim_copy(fields[0]));
        const int pid = std::stoi(trim_copy(fields[2]));
        const int relevance = std::stoi(trim_copy(fields[3]));

        if (qid != prev_qid) {
            prev_qid = qid;
            qrels[qid] = std::vector<int>();
        }
        if (relevance > 0) {
            qrels[qid].push_back(pid);
            ++rel_pairs;
        }
    }

    std::cerr << "[META] Loaded qrels: " << qrels.size()
              << " qids, " << rel_pairs << " relevant pairs" << std::endl;
}

static TextMeta load_text_meta(const Paths& paths) {
    TextMeta meta;
    load_collection_offset_to_pid(paths.collection_tsv, meta.offset_to_pid);
    load_query_offset_to_qid(paths.queries_tsv, meta.offset_to_qid);
    load_qrels_tsv(paths.qrels_tsv, meta.qrels);
    return meta;
}

static void write_ivecs(const std::string& filename, const std::vector<int64_t>& labels, size_t nq, int top_k) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) throw std::runtime_error("Cannot open output ivecs: " + filename);

    const auto t0 = std::chrono::high_resolution_clock::now();
    const size_t report_every = std::max<size_t>(1, nq / 10);

    for (size_t qi = 0; qi < nq; ++qi) {
        int32_t dim = static_cast<int32_t>(top_k);
        out.write(reinterpret_cast<const char*>(&dim), sizeof(int32_t));
        for (int j = 0; j < top_k; ++j) {
            const int64_t v = labels[qi * static_cast<size_t>(top_k) + j];
            int32_t label = -1;
            if (v >= 0 && v <= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
                label = static_cast<int32_t>(v);
            }
            out.write(reinterpret_cast<const char*>(&label), sizeof(int32_t));
        }
        if (!out) throw std::runtime_error("Failed while writing output ivecs: " + filename);
        if ((qi + 1) % report_every == 0 || qi + 1 == nq) {
            print_progress("Write ivecs", qi + 1, nq, t0);
        }
    }
    std::cout << "[Output] Wrote top-" << top_k << " results to " << filename << std::endl;
}

static double reciprocal_rank_at_k(
    const int64_t* labels,
    size_t query_offset,
    const TextMeta& meta,
    int top_k
) {
    if (query_offset >= meta.offset_to_qid.size()) {
        throw std::runtime_error("query offset out of range");
    }

    const int qid = meta.offset_to_qid[query_offset];
    const auto it = meta.qrels.find(qid);
    if (it == meta.qrels.end()) {

        return -1.0;
    }

    const std::vector<int>& relevant_pids = it->second;
    for (int rank = 0; rank < top_k; ++rank) {
        const int64_t passage_offset = labels[rank];
        if (passage_offset < 0) continue;
        if (static_cast<size_t>(passage_offset) >= meta.offset_to_pid.size()) continue;

        const int pid = meta.offset_to_pid[static_cast<size_t>(passage_offset)];
        if (std::find(relevant_pids.begin(), relevant_pids.end(), pid) != relevant_pids.end()) {
            return 1.0 / static_cast<double>(rank + 1);
        }
    }
    return 0.0;
}

static int recall_hits_at_k(const int64_t* labels, const int* gt_row, int gt_dim, int top_k) {
    std::unordered_set<int> truth;
    const int limit = std::min(top_k, gt_dim);
    truth.reserve(static_cast<size_t>(limit) * 2);
    for (int i = 0; i < limit; ++i) truth.insert(gt_row[i]);

    int hits = 0;
    for (int rank = 0; rank < top_k; ++rank) {
        const int64_t label = labels[rank];
        if (label >= 0 && label <= static_cast<int64_t>(std::numeric_limits<int>::max())) {
            if (truth.count(static_cast<int>(label))) ++hits;
        }
    }
    return hits;
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

template <typename T>
static T read_pod(const std::vector<uint8_t>& buf, size_t& offset) {
    if (offset + sizeof(T) > buf.size()) throw std::runtime_error("read_pod out of range");
    T value{};
    std::memcpy(&value, buf.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

static int connect_to_server(const std::string& ip, int port) {
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
    serv.sin_port = htons(static_cast<uint16_t>(port));
    if (::inet_pton(AF_INET, ip.c_str(), &serv.sin_addr) <= 0) throw std::runtime_error("inet_pton() failed");
    if (::connect(sockfd, reinterpret_cast<sockaddr*>(&serv), sizeof(serv)) < 0) throw std::runtime_error("connect() failed");
    return sockfd;
}

int main(int argc, char** argv) {
    try {
        RuntimeOptions opt = parse_args(argc, argv);
#ifdef _OPENMP
        if (opt.threads > 0) omp_set_num_threads(opt.threads);
#endif

        Paths paths = make_paths(opt.dataset);
        if (!opt.query_fvecs_override.empty()) paths.query_fvecs = opt.query_fvecs_override;
        if (!opt.collection_tsv_override.empty()) paths.collection_tsv = opt.collection_tsv_override;
        if (!opt.queries_tsv_override.empty()) paths.queries_tsv = opt.queries_tsv_override;
        if (!opt.qrels_tsv_override.empty()) paths.qrels_tsv = opt.qrels_tsv_override;
        if (!opt.gt_ivecs_override.empty()) paths.gt_ivecs = opt.gt_ivecs_override;
        if (!opt.output_ivecs_overridden) opt.output_ivecs = paths.output_ivecs;

        EvalMetric metric = opt.eval_metric == EvalMetric::Auto ? paths.default_metric : opt.eval_metric;

        require_file(paths.query_fvecs, "query fvecs");
        if (metric == EvalMetric::MRR) {
            require_file(paths.collection_tsv, "collection TSV");
            require_file(paths.queries_tsv, "queries TSV");
            require_file(paths.qrels_tsv, "qrels TSV");
        } else {
            require_file(paths.gt_ivecs, "ground-truth ivecs");
        }

        std::cout << "=== hnswlib Runtime Client Compass-Exact v1 ===" << std::endl;
        std::cout << "Dataset:               " << paths.dataset_name << std::endl;
        std::cout << "Resolved data dir:     " << paths.data_root << std::endl;
        std::cout << "Query fvecs:           " << paths.query_fvecs << std::endl;
        if (metric == EvalMetric::MRR) {
            std::cout << "Collection TSV:        " << paths.collection_tsv << std::endl;
            std::cout << "Queries TSV:           " << paths.queries_tsv << std::endl;
            std::cout << "Qrels TSV:             " << paths.qrels_tsv << std::endl;
        } else {
            std::cout << "Ground truth ivecs:    " << paths.gt_ivecs << std::endl;
        }
        std::cout << "Output ivecs:          " << opt.output_ivecs << std::endl;
        std::cout << "Server:                " << opt.server_ip << ":" << opt.port << std::endl;
        std::cout << "Metric:                " << metric_name_for_dataset(opt.dataset) << "(server-side)" << std::endl;
        std::cout << "Eval metric:           " << eval_metric_name(metric) << std::endl;
        std::cout << "Top-K / result dim:    " << opt.top_k << std::endl;
        std::cout << "ef(request):           " << opt.ef << std::endl;
        std::cout << "Test queries:          " << opt.test_queries << std::endl;
        std::cout << "Search batch size:     " << opt.search_batch_size << std::endl;
        std::cout << "Threads:               " << opt.threads << std::endl;
        std::cout << "Protocol version:      " << HNSWLIB_PROTOCOL_VERSION << std::endl;

        std::vector<float> query_data;
        size_t nq = 0;
        int query_dim = 0;
        load_fvecs(paths.query_fvecs, query_data, nq, query_dim, "queries");

        TextMeta meta;
        std::vector<int> gt_data;
        size_t gt_num = 0;
        int gt_dim = 0;

        if (metric == EvalMetric::MRR) {
            meta = load_text_meta(paths);
        } else {
            load_ivecs(paths.gt_ivecs, gt_data, gt_num, gt_dim);
        }

        size_t test_queries = opt.test_queries > 0 ? opt.test_queries : paths.default_test_queries;
        test_queries = std::min(test_queries, nq);
        if (metric == EvalMetric::MRR) {
            test_queries = std::min(test_queries, meta.offset_to_qid.size());
        } else {
            test_queries = std::min(test_queries, gt_num);
        }
        if (test_queries == 0) throw std::runtime_error("No query can be evaluated.");

        int sockfd = connect_to_server(opt.server_ip, opt.port);

        std::cout << "\n[Phase 1] Send queries to server and collect TopK" << std::endl;
        std::vector<int64_t> I(test_queries * static_cast<size_t>(opt.top_k), -1);
        std::vector<float> D(test_queries * static_cast<size_t>(opt.top_k), 0.0f);

        long long total_e2e_us = 0;
        long long total_recv_us = 0;
        long long total_server_us = 0;

        uint64_t total_request_bytes = 0;
        uint64_t total_response_bytes = 0;

        const auto search_t0 = std::chrono::high_resolution_clock::now();
        bool printed_server_ef = false;

        for (size_t start = 0; start < test_queries; start += opt.search_batch_size) {
            const size_t cur = std::min(opt.search_batch_size, test_queries - start);

            QueryBatchRequestHeader req{};
            req.magic = HNSWLIB_PROTOCOL_MAGIC;
            req.version = HNSWLIB_PROTOCOL_VERSION;
            req.flags = 1;
            req.batch_size = static_cast<int32_t>(cur);
            req.dim = query_dim;
            req.top_k = opt.top_k;
            req.ef = opt.ef;
            req.query_id_start = static_cast<uint64_t>(start);
            req.body_bytes = static_cast<int32_t>(cur * static_cast<size_t>(query_dim) * sizeof(float));

            std::vector<uint8_t> req_buf;
            req_buf.reserve(sizeof(QueryBatchRequestHeader) + static_cast<size_t>(req.body_bytes));
            append_pod(req_buf, req);
            const uint8_t* query_bytes = reinterpret_cast<const uint8_t*>(
                query_data.data() + start * static_cast<size_t>(query_dim)
            );
            req_buf.insert(req_buf.end(), query_bytes, query_bytes + static_cast<size_t>(req.body_bytes));

            total_request_bytes += static_cast<uint64_t>(req_buf.size());

            const auto t0 = std::chrono::high_resolution_clock::now();
            if (!send_all(sockfd, req_buf.data(), req_buf.size())) throw std::runtime_error("send batch request failed");

            QueryBatchResponseHeader resp{};
            if (!recv_all(sockfd, &resp, sizeof(resp))) throw std::runtime_error("recv batch response header failed");
            if (resp.magic != HNSWLIB_PROTOCOL_MAGIC || resp.version != HNSWLIB_PROTOCOL_VERSION) {
                throw std::runtime_error("protocol mismatch in response");
            }
            if (resp.batch_size != static_cast<int32_t>(cur)) throw std::runtime_error("response batch_size mismatch");
            if (resp.top_k != opt.top_k) throw std::runtime_error("response top_k mismatch");
            const int expected_ef = std::max(opt.ef, opt.top_k);
            if (resp.ef_used != expected_ef) {
                std::ostringstream oss;
                oss << "server did not apply requested ef: requested=" << opt.ef
                    << ", expected_used=" << expected_ef
                    << ", server_used=" << resp.ef_used;
                throw std::runtime_error(oss.str());
            }
            if (!printed_server_ef) {
                std::cout << "[Client] Server confirmed ef: requested=" << opt.ef
                          << ", used=" << resp.ef_used << std::endl;
                printed_server_ef = true;
            }

            std::vector<uint8_t> body(static_cast<size_t>(resp.body_bytes));
            if (resp.body_bytes > 0 && !recv_all(sockfd, body.data(), body.size())) {
                throw std::runtime_error("recv batch response body failed");
            }

            total_response_bytes += static_cast<uint64_t>(sizeof(QueryBatchResponseHeader))
                                  + static_cast<uint64_t>(body.size());

            const auto t1 = std::chrono::high_resolution_clock::now();

            size_t off = 0;
            for (size_t b = 0; b < cur; ++b) {
                QueryResultHeader qh = read_pod<QueryResultHeader>(body, off);
                if (qh.result_count != opt.top_k) throw std::runtime_error("result_count mismatch");
                const size_t q_body_end = off + static_cast<size_t>(qh.body_bytes);
                if (q_body_end > body.size()) throw std::runtime_error("corrupted response body");
                const size_t qi = static_cast<size_t>(qh.query_id);
                if (qi >= test_queries) throw std::runtime_error("query_id out of range");

                for (int j = 0; j < qh.result_count; ++j) {
                    ResultItem item = read_pod<ResultItem>(body, off);
                    I[qi * static_cast<size_t>(opt.top_k) + static_cast<size_t>(j)] = item.label;
                    D[qi * static_cast<size_t>(opt.top_k) + static_cast<size_t>(j)] = item.distance;
                }
                if (off != q_body_end) off = q_body_end;
                total_server_us += static_cast<long long>(qh.server_compute_us);
            }

            const auto t2 = std::chrono::high_resolution_clock::now();
            total_recv_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            total_e2e_us += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t0).count();
            print_progress("Search", start + cur, test_queries, search_t0);
        }

        ::close(sockfd);

        const double search_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - search_t0
        ).count();

        std::cout << "\n[Phase 2] Write TopK to ivecs" << std::endl;
        write_ivecs(opt.output_ivecs, I, test_queries, opt.top_k);

        std::cout << "\n[Phase 3] Compute " << (metric == EvalMetric::MRR ? "MRR" : "Recall") << std::endl;

        double mrr_sum = 0.0;
        long long evaluated_queries = 0;
        long long skipped_queries = 0;
        long long recall_hit_count = 0;
        long long recall_total_count = 0;

        const auto eval_t0 = std::chrono::high_resolution_clock::now();
        const size_t eval_report_every = std::max<size_t>(1, test_queries / 10);

        for (size_t qi = 0; qi < test_queries; ++qi) {
            if (metric == EvalMetric::MRR) {
                const double rr = reciprocal_rank_at_k(
                    I.data() + qi * static_cast<size_t>(opt.top_k),
                    qi,
                    meta,
                    opt.top_k
                );
                if (rr >= 0.0) {
                    mrr_sum += rr;
                    ++evaluated_queries;
                } else {
                    ++skipped_queries;
                }
            } else {
                const int max_gt = std::min(opt.top_k, gt_dim);
                const int hits = recall_hits_at_k(
                    I.data() + qi * static_cast<size_t>(opt.top_k),
                    gt_data.data() + qi * static_cast<size_t>(gt_dim),
                    gt_dim,
                    opt.top_k
                );
                recall_hit_count += hits;
                recall_total_count += max_gt;
                ++evaluated_queries;
            }
            if ((qi + 1) % eval_report_every == 0 || qi + 1 == test_queries) {
                print_progress(metric == EvalMetric::MRR ? "Eval MRR" : "Eval Recall", qi + 1, test_queries, eval_t0);
            }
        }

        const double search_time_ms = static_cast<double>(total_e2e_us) / 1000.0 / static_cast<double>(test_queries);
        const double request_response_ms = static_cast<double>(total_recv_us) / 1000.0 / static_cast<double>(test_queries);
        const double server_eval_ms = static_cast<double>(total_server_us) / 1000.0 / static_cast<double>(test_queries);
        const double client_dec_rank_ms = 0.0;

        double trans_latency_ms = search_time_ms - server_eval_ms - client_dec_rank_ms;
        if (trans_latency_ms < 0.0) trans_latency_ms = 0.0;

        const uint64_t total_comm_bytes = total_request_bytes + total_response_bytes;
        const double request_kb_per_query =
            static_cast<double>(total_request_bytes) / static_cast<double>(test_queries) / 1024.0;
        const double response_kb_per_query =
            static_cast<double>(total_response_bytes) / static_cast<double>(test_queries) / 1024.0;
        const double comm_kb_per_query =
            static_cast<double>(total_comm_bytes) / static_cast<double>(test_queries) / 1024.0;

        const double qps = static_cast<double>(test_queries) / (static_cast<double>(total_e2e_us) / 1e6);

        std::cout << "\n=== Result ===" << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "Config: Split hnswlib client/server | Dataset=" << paths.dataset_name
                  << " | Dim=" << query_dim
                  << " | Metric=" << metric_name_for_dataset(opt.dataset)
                  << " | Eval=" << eval_metric_name(metric)
                  << " | ef=" << opt.ef
                  << " | Top-K=" << opt.top_k
                  << " | Server=" << opt.server_ip << ":" << opt.port
                  << std::endl;
        std::cout << "Output ivecs: " << opt.output_ivecs << std::endl;
        if (metric == EvalMetric::MRR) {
            const double mrr = evaluated_queries > 0 ? mrr_sum / static_cast<double>(evaluated_queries) : 0.0;
            std::cout << "MRR@" << opt.top_k << ": " << std::fixed << std::setprecision(6) << mrr << std::endl;
            std::cout << "Evaluated queries: " << evaluated_queries << ", skipped: " << skipped_queries << std::endl;
        } else {
            const double recall = recall_total_count > 0 ?
                static_cast<double>(recall_hit_count) / static_cast<double>(recall_total_count) : 0.0;
            std::cout << "Recall@" << opt.top_k << ": " << std::fixed << std::setprecision(6) << recall << std::endl;
            std::cout << "Recall evaluated queries: " << evaluated_queries
                      << ", total ground-truth labels: " << recall_total_count << std::endl;
        }
        std::cout << "Total search time: " << std::fixed << std::setprecision(3)
                  << search_ms << " ms" << std::endl;
        std::cout << "Search Time: " << std::fixed << std::setprecision(4)
                  << search_time_ms << " ms/query" << std::endl;
        std::cout << "Server Eval. Time: " << std::fixed << std::setprecision(4)
                  << server_eval_ms << " ms/query" << std::endl;
        std::cout << "Trans. Latency: " << std::fixed << std::setprecision(4)
                  << trans_latency_ms << " ms/query" << std::endl;
        std::cout << "Client Dec.&Rank Time: " << std::fixed << std::setprecision(4)
                  << client_dec_rank_ms << " ms/query" << std::endl;
        std::cout << "Request/Response Time: " << std::fixed << std::setprecision(4)
                  << request_response_ms << " ms/query" << std::endl;
        std::cout << "Communication: " << std::fixed << std::setprecision(4)
                  << comm_kb_per_query << " KB/query"
                  << " (request=" << request_kb_per_query
                  << " KB/query, response=" << response_kb_per_query
                  << " KB/query)" << std::endl;
        std::cout << "QPS: " << std::fixed << std::setprecision(2) << qps << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "HNSWLIB CLIENT ERROR: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
