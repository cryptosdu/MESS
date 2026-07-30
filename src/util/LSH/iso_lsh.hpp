#pragma once
#include <vector>
#include <string>
#include <cstdint>

class IsoHasher {
public:
    IsoHasher(int input_dim, int code_bits);

    bool load_model(const std::string& filepath);

    std::vector<uint8_t> hash(const float* vec) const;

private:
    int input_dim_;
    int code_bits_;

    std::vector<float> mean_vector_;

    std::vector<float> projection_matrix_;

    bool is_model_loaded_;
};
