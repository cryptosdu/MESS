#include "hash_codeword.hpp"
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <random>

#ifdef _MSC_VER
#include <intrin.h>
#define POPCOUNT64 __popcnt64
#else
#include <x86intrin.h>
#define POPCOUNT64 __builtin_popcountll
#endif

HashCodeword::HashCodeword(size_t num_bytes) : bytes_(num_bytes, 0) {}

HashCodeword::HashCodeword(const std::vector<uint8_t>& data) : bytes_(data) {}

int HashCodeword::hamming_distance(const HashCodeword& other) const {
    if (bytes_.size() != other.bytes_.size()) {
        throw std::invalid_argument("HashCodeword dimension mismatch");
    }
    size_t dim = bytes_.size();
    return dist_func_static(bytes_.data(), other.bytes_.data(), &dim);
}

int HashCodeword::dist_func_static(const void* data_a, const void* data_b, const void* size_param) {
    const uint64_t* a = (const uint64_t*)data_a;
    const uint64_t* b = (const uint64_t*)data_b;

    size_t bytes = *((const size_t*)size_param);
    size_t n_blocks = bytes / 8;
    int dist = 0;

    for (size_t i = 0; i < n_blocks; ++i) {
        dist += (int)POPCOUNT64(a[i] ^ b[i]);
    }

    size_t remainder = bytes % 8;
    if (remainder) {
        const uint8_t* a8 = (const uint8_t*)data_a;
        const uint8_t* b8 = (const uint8_t*)data_b;
        size_t offset = n_blocks * 8;

        for (size_t i = 0; i < remainder; ++i) {
            uint8_t xor_val = a8[offset + i] ^ b8[offset + i];
            while (xor_val) {
                dist++;
                xor_val &= (xor_val - 1);
            }
        }
    }
    return dist;
}

bool HashCodeword::operator==(const HashCodeword& other) const {
    return bytes_ == other.bytes_;
}

bool HashCodeword::operator!=(const HashCodeword& other) const {
    return !(*this == other);
}

std::string HashCodeword::to_hex() const {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (uint8_t b : bytes_) {
        ss << std::setw(2) << static_cast<int>(b);
    }
    return ss.str();
}

void HashCodeword::inject_noise(double flip_probability) {
    if (flip_probability <= 0.0) return;
    if (flip_probability >= 1.0) {

        for (auto& b : bytes_) b = ~b;
        return;
    }

    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());

    std::bernoulli_distribution d(flip_probability);

    for (size_t i = 0; i < bytes_.size(); ++i) {
        uint8_t noise_mask = 0;

        for (int bit = 0; bit < 8; ++bit) {

            if (d(gen)) {

                noise_mask |= (1 << bit);
            }
        }

        bytes_[i] ^= noise_mask;
    }
}
