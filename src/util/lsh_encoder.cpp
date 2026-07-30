#include "lsh_encoder.hpp"

#include <iostream>
#include <omp.h>
#include <stdexcept>

LSHEncoder::LSHEncoder(int dim,
                       int bits,
                       HashMethod method,
                       const std::string& model_path,
                       float,
                       int)
    : input_dim_(dim), code_bits_(bits), method_(method) {
    if (method_ != HashMethod::ISOHASH) {
        throw std::invalid_argument("Only IsoHash is retained in this cleaned version.");
    }

    iso_engine_ = std::make_unique<IsoHasher>(dim, bits);
    std::cout << "[LSHEncoder] Initialized with IsoHash Engine. (" << bits << " bits)" << std::endl;

    if (!model_path.empty()) {
        load_isohash_model(model_path + "/isohash_weights.bin");
    }
}

bool LSHEncoder::load_isohash_model(const std::string& filepath) {
    if (method_ != HashMethod::ISOHASH) {
        std::cerr << "[LSHEncoder] Warning: Current method is not ISOHASH. Ignored model loading." << std::endl;
        return false;
    }
    return iso_engine_->load_model(filepath);
}

HashCodeword LSHEncoder::encode(const std::vector<float>& vec) const {
    return encode(vec.data());
}

HashCodeword LSHEncoder::encode(const float* data) const {
    if (!iso_engine_) {
        throw std::runtime_error("[LSHEncoder] IsoHash engine is not initialized.");
    }
    return HashCodeword(iso_engine_->hash(data));
}

std::vector<HashCodeword> LSHEncoder::batch_encode(const float* data, size_t num_vectors) const {
    std::vector<HashCodeword> all_codewords(num_vectors);
    int n = static_cast<int>(num_vectors);
#pragma omp parallel for schedule(static) default(none) shared(data, all_codewords, n)
    for (int i = 0; i < n; ++i) {
        const float* vec_ptr = data + static_cast<size_t>(i) * input_dim_;
        all_codewords[i] = encode(vec_ptr);
    }
    return all_codewords;
}
