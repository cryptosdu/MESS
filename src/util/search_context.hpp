#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "data_loader.hpp"
#include "hash_codeword.hpp"
#include "hnsw_index.hpp"
#include "lsh_encoder.hpp"

class SecureSearchContext;

struct SearchContext {
    int k = 10;
    int ef_search = 100;
    bool verbose = false;
};

struct SearchResult {
    int query_id = -1;
    std::vector<std::pair<int, float>> results;
    double time_us = 0.0;
};

class VectorSearchSystem {
public:
    struct Config {
        int input_dim = 0;
        int hash_bits = 0;
        int num_graphs = 2;
        size_t max_items = 0;
        int ef_construction = 200;
        int M = 16;
    };

    VectorSearchSystem(
        const Config& config,
        std::shared_ptr<SecureSearchContext> secure_ctx
    );

    ~VectorSearchSystem() = default;

    void build_index(const std::string& base_file_path);

    std::vector<SearchResult> search(
        const std::string& query_file_path,
        const SearchContext& ctx
    );

private:
    Config config_;
    std::shared_ptr<SecureSearchContext> secure_ctx_;

    std::unique_ptr<DataLoader> loader_;
    std::unique_ptr<LSHEncoder> encoder_;
    std::vector<std::unique_ptr<HNSWIndex>> indices_;

    std::vector<std::vector<std::vector<uint8_t>>> encrypted_payloads_;
};
