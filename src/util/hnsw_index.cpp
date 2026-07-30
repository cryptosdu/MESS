#include "hnsw_index.hpp"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <omp.h>
#include <random>
#include <numeric>

#ifdef _MSC_VER
#include <intrin.h>
#define POPCOUNT64 __popcnt64
#else
#include <x86intrin.h>
#define POPCOUNT64 __builtin_popcountll
#endif

HammingSpace::HammingSpace(size_t dim_bytes) : dim_bytes_(dim_bytes) {}

size_t HammingSpace::get_data_size() { return dim_bytes_; }
hnswlib::DISTFUNC<int> HammingSpace::get_dist_func() { return dist_func; }
void* HammingSpace::get_dist_func_param() { return &dim_bytes_; }

int HammingSpace::dist_func(const void* data_a, const void* data_b, const void* dist_func_param) {
    const auto* bytes_ptr = static_cast<const size_t*>(dist_func_param);
    size_t bytes = *bytes_ptr;

    int dist = 0;

#if defined(__AVX2__)
    cout<<"Is AVX2";

    const uint8_t* a = static_cast<const uint8_t*>(data_a);
    const uint8_t* b = static_cast<const uint8_t*>(data_b);
    size_t i = 0;

    __m256i lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    );
    __m256i low_mask = _mm256_set1_epi8(0x0F);
    __m256i acc = _mm256_setzero_si256();

    for (; i + 31 < bytes; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));

        __m256i v_xor = _mm256_xor_si256(va, vb);

        __m256i lo = _mm256_and_si256(v_xor, low_mask);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v_xor, 4), low_mask);

        __m256i pop_lo = _mm256_shuffle_epi8(lookup, lo);
        __m256i pop_hi = _mm256_shuffle_epi8(lookup, hi);

        acc = _mm256_add_epi8(acc, _mm256_add_epi8(pop_lo, pop_hi));
    }

    __m256i sum_64 = _mm256_sad_epu8(acc, _mm256_setzero_si256());
    dist += _mm256_extract_epi64(sum_64, 0);
    dist += _mm256_extract_epi64(sum_64, 1);
    dist += _mm256_extract_epi64(sum_64, 2);
    dist += _mm256_extract_epi64(sum_64, 3);

    for (; i < bytes; ++i) {

        dist += _popcnt32(a[i] ^ b[i]);
    }

#else

    const auto* a = static_cast<const uint64_t*>(data_a);
    const auto* b = static_cast<const uint64_t*>(data_b);

    size_t n_blocks = bytes / 8;

    for (size_t i = 0; i < n_blocks; i++) {
        dist += static_cast<int>(POPCOUNT64(a[i] ^ b[i]));
    }

    if (bytes % 8 != 0) {
        const auto* a8 = static_cast<const uint8_t*>(data_a);
        const auto* b8 = static_cast<const uint8_t*>(data_b);
        size_t offset = n_blocks * 8;
        for (size_t i = offset; i < bytes; i++) {
            uint8_t xor_val = a8[i] ^ b8[i];
            while (xor_val) { dist++; xor_val &= (xor_val - 1); }
        }
    }

#endif

    return dist;
}

HNSWIndex::HNSWIndex(int dim_bits, size_t max_elements, int M, int ef_construction)
        : dim_bits_(dim_bits), dim_bytes_(dim_bits / 8) {

    if (dim_bits % 8 != 0) throw std::runtime_error("Bits must be multiple of 8");

    space_ = new HammingSpace(dim_bytes_);
    alg_hnsw_ = new hnswlib::HierarchicalNSW<int>(space_, max_elements, M, ef_construction);
}

HNSWIndex::~HNSWIndex() {
    delete alg_hnsw_;
    delete space_;
}

void HNSWIndex::set_edge_dp_params(float p_drop, float p_add) {
    p_drop_ = p_drop;
    p_add_ = p_add;
}

void HNSWIndex::apply_edge_dp_topology() {
    if (p_drop_ <= 0.0f && p_add_ <= 0.0f) return;

    size_t num_elements = alg_hnsw_->cur_element_count;
    if (num_elements == 0) return;

    std::cout << "\n[Security] Initiating Edge-DP Topology Obfuscation (p_drop=" << p_drop_
              << ", p_add=" << p_add_ << ")...\n" << std::flush;

    int max_M0 = alg_hnsw_->maxM0_;

    size_t count_size = alg_hnsw_->size_links_level0_ - max_M0 * sizeof(hnswlib::tableint);

    float local_p_drop = p_drop_;
    float local_p_add = p_add_;
    auto* local_alg_hnsw = alg_hnsw_;

#pragma omp parallel default(none) shared(num_elements, max_M0, count_size, local_p_drop, local_p_add, local_alg_hnsw)
    {

        std::mt19937 local_rng(42 + omp_get_thread_num());
        std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
        std::uniform_int_distribution<hnswlib::tableint> id_dist(0, static_cast<hnswlib::tableint>(num_elements - 1));

#pragma omp for
        for (int i = 0; i < (int)num_elements; ++i) {
            char* data_ptr = local_alg_hnsw->data_level0_memory_ + static_cast<size_t>(i) * local_alg_hnsw->size_data_per_element_;

            size_t current_links = 0;
            if (count_size == 2) {
                current_links = *reinterpret_cast<unsigned short*>(data_ptr);
            } else if (count_size == 4) {
                current_links = *reinterpret_cast<unsigned int*>(data_ptr);
            } else {
                current_links = *reinterpret_cast<size_t*>(data_ptr);
            }

            if (current_links > static_cast<size_t>(max_M0)) {
                current_links = max_M0;
            }

            auto* links = reinterpret_cast<hnswlib::tableint*>(data_ptr + count_size);

            std::vector<hnswlib::tableint> new_links;
            new_links.reserve(max_M0);

            for (size_t j = 0; j < current_links; ++j) {
                if (prob_dist(local_rng) > local_p_drop) {
                    new_links.push_back(links[j]);
                }
            }

            size_t current_size = new_links.size();
            for (size_t j = current_size; j < static_cast<size_t>(max_M0); ++j) {
                if (prob_dist(local_rng) < local_p_add) {
                    hnswlib::tableint dummy_id = id_dist(local_rng);
                    if (dummy_id != static_cast<hnswlib::tableint>(i)) {
                        new_links.push_back(dummy_id);
                    }
                }
            }

            if (count_size == 2) {
                *reinterpret_cast<unsigned short*>(data_ptr) = static_cast<unsigned short>(new_links.size());
            } else if (count_size == 4) {
                *reinterpret_cast<unsigned int*>(data_ptr) = static_cast<unsigned int>(new_links.size());
            } else {
                *reinterpret_cast<size_t*>(data_ptr) = static_cast<size_t>(new_links.size());
            }

            for (size_t j = 0; j < new_links.size(); ++j) {
                links[j] = new_links[j];
            }
        }
    }

    std::cout << "[Security] Edge-DP topology obfuscation successfully executed. Physical graph isomorphism has been permanently disrupted!\n" << std::flush;
}

void HNSWIndex::add_item(const HashCodeword& code, size_t label) {
    const std::vector<uint8_t>& bytes = code.getBytes();
    if (bytes.size() != dim_bytes_) throw std::runtime_error("Size mismatch");
    alg_hnsw_->addPoint(bytes.data(), label);
}

std::vector<std::pair<size_t, int>> HNSWIndex::search(const HashCodeword& code, int k) {
    auto result_pq = query(code, k);

    std::vector<std::pair<size_t, int>> results;
    results.reserve(k);
    while (!result_pq.empty()) {
        auto& item = result_pq.top();
        results.push_back({ item.second, item.first });
        result_pq.pop();
    }
    std::reverse(results.begin(), results.end());
    return results;
}

std::priority_queue<std::pair<int, size_t>> HNSWIndex::query(const HashCodeword& code, int k) {
    const std::vector<uint8_t>& bytes = code.getBytes();
    if (bytes.size() != dim_bytes_) {
        throw std::runtime_error("Error: Query code size mismatch!");
    }
    return alg_hnsw_->searchKnn(bytes.data(), k);
}

void HNSWIndex::add_batch(const std::vector<HashCodeword>& codes, const std::vector<size_t>& labels) {
    if (codes.size() != labels.size()) throw std::runtime_error("Codes and labels size mismatch");

    int n = static_cast<int>(codes.size());

#pragma omp parallel for default(none) shared(codes, labels, n)
    for (int i = 0; i < n; ++i) {
        add_item(codes[i], labels[i]);
    }
}

std::vector<std::vector<std::pair<size_t, int>>> HNSWIndex::search_batch(const std::vector<HashCodeword>& codes, int k) {
    int n = static_cast<int>(codes.size());
    std::vector<std::vector<std::pair<size_t, int>>> all_results(n);

#pragma omp parallel for default(none) shared(codes, k, n, all_results)
    for (int i = 0; i < n; ++i) {
        all_results[i] = search(codes[i], k);
    }
    return all_results;
}

void HNSWIndex::add_batch_random_routing(
        std::vector<HNSWIndex*>& indices,
        const std::vector<HashCodeword>& codes,
        const std::vector<size_t>& labels,
        int num_graphs_to_pick)
{
    if (codes.size() != labels.size()) {
        throw std::runtime_error("Codes and labels size mismatch!");
    }
    if (indices.empty()) {
        throw std::runtime_error("Indices array is empty!");
    }

    int total_graphs = static_cast<int>(indices.size());

    if (num_graphs_to_pick <= 0 || num_graphs_to_pick > total_graphs) {
        throw std::runtime_error("Invalid num_graphs_to_pick: must be > 0 and <= total number of graphs");
    }

    int n = static_cast<int>(codes.size());

    std::cout << "\n[Routing] Starting Random Multi-Assignment: " << n << " vectors -> "
              << num_graphs_to_pick << "/" << total_graphs << " graphs per vector..." << std::endl;

#pragma omp parallel default(none) shared(n, total_graphs, num_graphs_to_pick, indices, codes, labels)
    {

        std::mt19937 local_rng(42 + omp_get_thread_num());

        std::vector<int> local_graph_ids(total_graphs);
        std::iota(local_graph_ids.begin(), local_graph_ids.end(), 0);

#pragma omp for
        for (int i = 0; i < n; ++i) {

            for (int j = 0; j < num_graphs_to_pick; ++j) {
                std::uniform_int_distribution<int> dist(j, total_graphs - 1);
                int swap_idx = dist(local_rng);

                std::swap(local_graph_ids[j], local_graph_ids[swap_idx]);

                int chosen_graph = local_graph_ids[j];

                indices[chosen_graph]->add_item(codes[i], labels[i]);
            }
        }
    }

    std::cout << "[Routing] Random Multi-Assignment completed successfully!\n" << std::endl;
}

void HNSWIndex::set_ef(int ef) { alg_hnsw_->setEf(ef); }
void HNSWIndex::save(const std::string& path) { alg_hnsw_->saveIndex(path); }
void HNSWIndex::load(const std::string& path, size_t max) {
    delete alg_hnsw_;
    alg_hnsw_ = new hnswlib::HierarchicalNSW<int>(space_, path, false, max);
}
