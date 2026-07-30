#include "iso_lsh.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

IsoHasher::IsoHasher(int input_dim, int code_bits)
        : input_dim_(input_dim), code_bits_(code_bits), is_model_loaded_(false) {
}

bool IsoHasher::load_model(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[IsoHasher] Error: Cannot open IsoHash model file: " << filepath << std::endl;
        return false;
    }

    mean_vector_.resize(input_dim_);
    in.read(reinterpret_cast<char*>(mean_vector_.data()), input_dim_ * sizeof(float));

    size_t matrix_size = static_cast<size_t>(code_bits_) * input_dim_;
    projection_matrix_.resize(matrix_size);
    in.read(reinterpret_cast<char*>(projection_matrix_.data()), matrix_size * sizeof(float));

    if (!in) {
        std::cerr << "[IsoHasher] Error: Failed to read complete model data. File size mismatch!" << std::endl;
        return false;
    }

    is_model_loaded_ = true;
    std::cout << "[IsoHasher] Successfully loaded model from " << filepath << std::endl;
    return true;
}

std::vector<uint8_t> IsoHasher::hash(const float* vec) const {
    if (!is_model_loaded_) {
        throw std::runtime_error("[IsoHasher] Error: Model is not loaded! Call load_model() first.");
    }

    int num_bytes = (code_bits_ + 7) / 8;
    std::vector<uint8_t> raw_bytes(num_bytes, 0);

    for (int i = 0; i < code_bits_; ++i) {
        float dot = 0.0f;
        int row_offset = i * input_dim_;

        for (int j = 0; j < input_dim_; ++j) {

            float centered_val = vec[j] - mean_vector_[j];
            dot += projection_matrix_[row_offset + j] * centered_val;
        }

        if (dot > 0.0f) {
            int byte_idx = i / 8;
            int bit_idx = i % 8;
            raw_bytes[byte_idx] |= (1 << bit_idx);
        }
    }

    return raw_bytes;
}
