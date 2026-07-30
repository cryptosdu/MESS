#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <iostream>

#ifdef _MSC_VER
#include <intrin.h>
#define POPCOUNT64 __popcnt64
#else
#include <x86intrin.h>
#define POPCOUNT64 __builtin_popcountll
#endif

class HashCodeword {
public:

    HashCodeword() = default;

    explicit HashCodeword(size_t num_bytes);

    explicit HashCodeword(const std::vector<uint8_t>& data);

    const std::vector<uint8_t>& getBytes() const {
        return bytes_;
    }

    int hamming_distance(const HashCodeword& other) const;

    bool operator==(const HashCodeword& other) const;
    bool operator!=(const HashCodeword& other) const;

    const uint8_t* data() const { return bytes_.data(); }
    uint8_t* data() { return bytes_.data(); }

    size_t size_bytes() const { return bytes_.size(); }
    size_t size_bits() const { return bytes_.size() * 8; }

    std::string to_hex() const;

    static int dist_func_static(const void* a, const void* b, const void* size_param);

    HashCodeword split(int part_idx, int total_parts) const {
        size_t total_len = bytes_.size();
        size_t part_len = total_len / total_parts;

        size_t start = part_idx * part_len;
        size_t end = start + part_len;

        if (end > total_len) end = total_len;

        std::vector<uint8_t> sub_bytes(bytes_.begin() + start, bytes_.begin() + end);
        return HashCodeword(sub_bytes);
    }

    static int distance(const HashCodeword& a, const HashCodeword& b) {
        const uint64_t* pa = (const uint64_t*)a.bytes_.data();
        const uint64_t* pb = (const uint64_t*)b.bytes_.data();

        size_t n_blocks = a.bytes_.size() / 8;
        int dist = 0;
        for (size_t i = 0; i < n_blocks; ++i) {
            dist += (int)POPCOUNT64(pa[i] ^ pb[i]);
        }
        return dist;
    }

    void inject_noise(double flip_probability);

private:
    std::vector<uint8_t> bytes_;
};

struct HashCodewordHasher {
    std::size_t operator()(const HashCodeword& hc) const {
        std::size_t hash_val = 0;
        for (uint8_t byte : hc.getBytes()) {

            hash_val ^= std::hash<uint8_t>()(byte) + 0x9e3779b9 + (hash_val << 6) + (hash_val >> 2);
        }
        return hash_val;
    }
};
