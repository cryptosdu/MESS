#pragma once

#include <vector>
#include <string>
#include <queue>
#include <utility>
#include <cstddef>

#include "hash_codeword.hpp"
#include "hnswlib/hnswlib.h"

class HammingSpace : public hnswlib::SpaceInterface<int> {
public:
    explicit HammingSpace(size_t dim_bytes);
    ~HammingSpace() override = default;

    size_t get_data_size() override;
    hnswlib::DISTFUNC<int> get_dist_func() override;
    void* get_dist_func_param() override;

    static int dist_func(const void* data_a, const void* data_b, const void* dist_func_param);

private:
    size_t dim_bytes_;
};

class HNSWIndex {
public:
    HNSWIndex(int dim_bits, size_t max_elements, int M = 16, int ef_construction = 200);
    ~HNSWIndex();

    void set_edge_dp_params(float p_drop, float p_add);

    void apply_edge_dp_topology();

    void add_item(const HashCodeword& code, size_t label);

    std::vector<std::pair<size_t, int>> search(const HashCodeword& code, int k);

    std::priority_queue<std::pair<int, size_t>> query(const HashCodeword& code, int k);

    void add_batch(const std::vector<HashCodeword>& codes, const std::vector<size_t>& labels);
    std::vector<std::vector<std::pair<size_t, int>>> search_batch(const std::vector<HashCodeword>& codes, int k);

    static void add_batch_random_routing(
            std::vector<HNSWIndex*>& indices,
            const std::vector<HashCodeword>& codes,
            const std::vector<size_t>& labels,
            int num_graphs_to_pick
    );

    void set_ef(int ef);
    void save(const std::string& path);
    void load(const std::string& path, size_t max_elements);

private:
    int dim_bits_;
    size_t dim_bytes_;

    float p_drop_;
    float p_add_;

    HammingSpace* space_;
    hnswlib::HierarchicalNSW<int>* alg_hnsw_;
};
