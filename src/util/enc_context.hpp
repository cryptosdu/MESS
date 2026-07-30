#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <stdexcept>

class EncContext {
private:
    std::vector<uint8_t> key_;

    static constexpr int KEY_SIZE = 32;
    static constexpr int IV_SIZE = 12;
    static constexpr int TAG_SIZE = 16;

public:

    EncContext();

    explicit EncContext(const std::vector<uint8_t>& key);

    void generate_random_key();

    const std::vector<uint8_t>& get_key() const;

    std::vector<uint8_t> encrypt(const uint8_t* plaintext, size_t len) const;

    bool decrypt(const std::vector<uint8_t>& payload, uint8_t* out_plaintext, size_t expected_len) const;
};
