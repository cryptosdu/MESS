#include "search_context.hpp"
#include "securesearchcontext.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

namespace {

void normalize_vector(std::vector<float>& v) {
    double norm = 0.0;
    for (float x : v) {
        norm += static_cast<double>(x) * static_cast<double>(x);
    }

    norm = std::sqrt(norm);
    if (norm > 1e-9) {
        for (float& x : v) {
            x = static_cast<float>(x / norm);
        }
    }
}

float l2_distance_sq(const float* a, const float* b, int dim) {
    float dist = 0.0f;
    for (int i = 0; i < dim; ++i) {
        const float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

}

VectorSearchSystem::VectorSearchSystem(
    const Config& config,
    std::shared_ptr<SecureSearchContext> secure_ctx)
    : config_(config), secure_ctx_(std::move(secure_ctx)) {
    if (!secure_ctx_) {
        throw std::invalid_argument("VectorSearchSystem: secure_ctx is null.");
    }

    if (config_.input_dim <= 0) {
        throw std::invalid_argument("VectorSearchSystem: input_dim must be positive.");
    }
    if (config_.hash_bits <= 0) {
        throw std::invalid_argument("VectorSearchSystem: hash_bits must be positive.");
    }
    if (config_.num_graphs <= 0) {
        throw std::invalid_argument("VectorSearchSystem: num_graphs must be positive.");
    }
    if (config_.max_items == 0) {
        throw std::invalid_argument("VectorSearchSystem: max_items must be > 0.");
    }
    if (config_.hash_bits % config_.num_graphs != 0) {
        throw std::runtime_error("VectorSearchSystem: hash_bits must be divisible by num_graphs.");
    }

    if (secure_ctx_->num_graphs() != config_.num_graphs) {
        throw std::runtime_error("VectorSearchSystem: secure_ctx num_graphs mismatch.");
    }
    if (secure_ctx_->input_dim() != config_.input_dim) {
        throw std::runtime_error("VectorSearchSystem: secure_ctx input_dim mismatch.");
    }

    loader_ = std::make_unique<DataLoader>();
    encoder_ = std::make_unique<LSHEncoder>(config_.input_dim, config_.hash_bits);

    const int sub_bits = config_.hash_bits / config_.num_graphs;
    indices_.reserve(config_.num_graphs);

    for (int g = 0; g < config_.num_graphs; ++g) {
        indices_.push_back(std::make_unique<HNSWIndex>(
            sub_bits,
            config_.max_items,
            config_.M,
            config_.ef_construction));
    }

    encrypted_payloads_.resize(config_.num_graphs);
    for (int g = 0; g < config_.num_graphs; ++g) {
        encrypted_payloads_[g].resize(config_.max_items);
    }

    std::cout << "[System] Mode: Coarse(LSH Multi-Graph) -> Fine(Decrypt + Re-rank)\n";
    std::cout << "  - Input Dim  : " << config_.input_dim << "\n";
    std::cout << "  - Hash Bits  : " << config_.hash_bits << "\n";
    std::cout << "  - Num Graphs : " << config_.num_graphs << "\n";
}

void VectorSearchSystem::build_index(const std::string& base_file_path) {
    std::cout << "\n[Build] Loading base data...\n";

    std::vector<float> data;
    int num = 0;
    int dim = 0;
    loader_->load_fvecs(base_file_path, data, num, dim);

    if (dim != config_.input_dim) {
        throw std::runtime_error("build_index: input dimension mismatch.");
    }
    if (static_cast<size_t>(num) > config_.max_items) {
        throw std::runtime_error("build_index: dataset size exceeds max_items.");
    }

    std::cout << "[Build] Normalizing, encrypting, and indexing...\n";

    std::vector<float> temp_vec(static_cast<size_t>(dim));

    for (int i = 0; i < num; ++i) {
        const float* raw_ptr = data.data() + static_cast<size_t>(i) * dim;

        std::copy(raw_ptr, raw_ptr + dim, temp_vec.begin());
        normalize_vector(temp_vec);

        HashCodeword full_code = encoder_->encode(temp_vec.data());

        for (int g = 0; g < config_.num_graphs; ++g) {
            encrypted_payloads_[g][i] =
                secure_ctx_->encrypt_vector(g, temp_vec.data(), dim);

            HashCodeword sub_code = full_code.split(g, config_.num_graphs);
            indices_[g]->add_item(sub_code, i);
        }

        if ((i + 1) % 2000 == 0) {
            std::cout << "\r  Progress: " << (i + 1) << " / " << num << std::flush;
        }
    }

    std::cout << "\n[Build] Done.\n";
}

std::vector<SearchResult> VectorSearchSystem::search(
    const std::string& query_file_path,
    const SearchContext& ctx) {
    if (ctx.verbose) {
        std::cout << "\n[Search] Strategy: LSH Recall -> Decrypt -> Re-rank\n";
    }

    std::vector<float> queries;
    int num = 0;
    int dim = 0;
    loader_->load_fvecs(query_file_path, queries, num, dim);

    if (dim != config_.input_dim) {
        throw std::runtime_error("search: query dimension mismatch.");
    }

    for (auto& index : indices_) {
        index->set_ef(ctx.ef_search);
    }

    std::vector<SearchResult> all_results;
    all_results.reserve(static_cast<size_t>(num));

    std::vector<float> query_vec(static_cast<size_t>(dim));
    double total_time_us = 0.0;

    for (int qid = 0; qid < num; ++qid) {
        const auto t1 = std::chrono::high_resolution_clock::now();

        const float* raw_query = queries.data() + static_cast<size_t>(qid) * dim;
        std::copy(raw_query, raw_query + dim, query_vec.begin());
        normalize_vector(query_vec);

        HashCodeword full_code = encoder_->encode(query_vec.data());

        const int pool_size = std::max(ctx.k * 2, 100);
        std::unordered_map<int, int> candidate_source_graph;

        for (int g = 0; g < config_.num_graphs; ++g) {
            HashCodeword sub_code = full_code.split(g, config_.num_graphs);
            auto pq = indices_[g]->query(sub_code, pool_size);

            while (!pq.empty()) {
                const int label = pq.top().second;
                pq.pop();

                if (candidate_source_graph.find(label) == candidate_source_graph.end()) {
                    candidate_source_graph[label] = g;
                }
            }
        }

        std::vector<std::pair<int, float>> ranked_results;
        ranked_results.reserve(candidate_source_graph.size());

        for (const auto& kv : candidate_source_graph) {
            const int label = kv.first;
            const int graph_id = kv.second;

            const auto& payload = encrypted_payloads_[graph_id][label];
            if (payload.empty()) {
                continue;
            }

            std::vector<float> plain_vec =
                secure_ctx_->decrypt_vector(graph_id, payload, dim);

            const float dist = l2_distance_sq(query_vec.data(), plain_vec.data(), dim);
            ranked_results.push_back({label, dist});
        }

        std::sort(
            ranked_results.begin(),
            ranked_results.end(),
            [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                return a.second < b.second;
            });

        if (ranked_results.size() > static_cast<size_t>(ctx.k)) {
            ranked_results.resize(static_cast<size_t>(ctx.k));
        }

        const auto t2 = std::chrono::high_resolution_clock::now();
        const double time_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        total_time_us += time_us;

        SearchResult result;
        result.query_id = qid;
        result.results = std::move(ranked_results);
        result.time_us = time_us;

        all_results.push_back(std::move(result));
    }

    if (ctx.verbose && num > 0) {
        std::cout << "[Search] Finished. Avg Latency: "
                  << (total_time_us / static_cast<double>(num)) << " us.\n";
    }

    return all_results;
}
