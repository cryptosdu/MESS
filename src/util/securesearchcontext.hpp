#pragma once

#include <cstdint>
#include <vector>

#include "enc_context.hpp"
#include "search_context.hpp"

class SecureSearchContext {
public:
    struct Config {
        int num_graphs = 2;
        int input_dim = 0;
        SearchContext search;

        bool use_pre_shared_master_key = false;
        std::vector<uint8_t> master_key;
    };

    explicit SecureSearchContext(const Config& cfg);

    SearchContext& search_options();
    const SearchContext& search_options() const;

    int num_graphs() const;
    int input_dim() const;

    std::vector<uint8_t> encrypt_vector(
        int graph_id,
        const float* vec,
        int dim
    ) const;

    std::vector<float> decrypt_vector(
        int graph_id,
        const std::vector<uint8_t>& payload,
        int dim
    ) const;

private:
    void check_graph_id(int graph_id) const;

    static std::vector<uint8_t> derive_graph_key(
        const std::vector<uint8_t>& master_key,
        int graph_id
    );

private:
    SearchContext search_ctx_;
    int num_graphs_ = 0;
    int input_dim_ = 0;

    std::vector<EncContext> graph_enc_contexts_;
};
