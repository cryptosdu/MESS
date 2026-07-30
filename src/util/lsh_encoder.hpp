#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "hash_codeword.hpp"
#include "LSH/iso_lsh.hpp"

enum class HashMethod {
    ISOHASH
};

class LSHEncoder {
public:
    LSHEncoder(int dim,
               int bits,
               HashMethod method = HashMethod::ISOHASH,
               const std::string& model_path = "",
               float w = 4.0f,
               int hybrid_e2lsh_bits = 16);

    bool load_isohash_model(const std::string& filepath);

    HashCodeword encode(const std::vector<float>& vec) const;
    HashCodeword encode(const float* data) const;

    std::vector<HashCodeword> batch_encode(const float* data, size_t num_vectors) const;

    int get_input_dim() const { return input_dim_; }
    int get_code_bits() const { return code_bits_; }

private:
    int input_dim_;
    int code_bits_;
    HashMethod method_;
    std::unique_ptr<IsoHasher> iso_engine_;
};
