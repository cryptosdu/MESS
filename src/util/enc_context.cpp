#include "enc_context.hpp"
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <iostream>
#include <cstring>

EncContext::EncContext() {
    generate_random_key();
}

EncContext::EncContext(const std::vector<uint8_t>& key) {
    if (key.size() != KEY_SIZE) {
        throw std::invalid_argument("AES-256 requires exactly a 32-byte key.");
    }
    key_ = key;
}

void EncContext::generate_random_key() {
    key_.resize(KEY_SIZE);

    if (RAND_bytes(key_.data(), KEY_SIZE) != 1) {
        throw std::runtime_error("Failed to generate secure random key via OpenSSL.");
    }
}

const std::vector<uint8_t>& EncContext::get_key() const {
    return key_;
}

std::vector<uint8_t> EncContext::encrypt(const uint8_t* plaintext, size_t len) const {

    std::vector<uint8_t> payload(IV_SIZE + TAG_SIZE + len);

    uint8_t* iv_ptr = payload.data();
    uint8_t* tag_ptr = payload.data() + IV_SIZE;
    uint8_t* ciphertext_ptr = payload.data() + IV_SIZE + TAG_SIZE;

    if (RAND_bytes(iv_ptr, IV_SIZE) != 1) {
        throw std::runtime_error("Failed to generate random IV.");
    }

    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (!ctx) throw std::runtime_error("Failed to create EVP context.");

    int out_len1 = 0, out_len2 = 0;

    EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, nullptr, nullptr);

    EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, IV_SIZE, nullptr);

    EVP_EncryptInit_ex(ctx, nullptr, nullptr, key_.data(), iv_ptr);

    EVP_EncryptUpdate(ctx, ciphertext_ptr, &out_len1, plaintext, len);

    EVP_EncryptFinal_ex(ctx, ciphertext_ptr + out_len1, &out_len2);

    EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, TAG_SIZE, tag_ptr);

    EVP_CIPHER_CTX_free(ctx);

    return payload;
}

bool EncContext::decrypt(const std::vector<uint8_t>& payload, uint8_t* out_plaintext, size_t expected_len) const {

    if (payload.size() != IV_SIZE + TAG_SIZE + expected_len) {
        return false;
    }

    const uint8_t* iv_ptr = payload.data();
    const uint8_t* tag_ptr = payload.data() + IV_SIZE;
    const uint8_t* ciphertext_ptr = payload.data() + IV_SIZE + TAG_SIZE;

    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return false;

    int out_len1 = 0, out_len2 = 0;

    EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, nullptr, nullptr);
    EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, IV_SIZE, nullptr);
    EVP_DecryptInit_ex(ctx, nullptr, nullptr, key_.data(), iv_ptr);

    EVP_DecryptUpdate(ctx, out_plaintext, &out_len1, ciphertext_ptr, expected_len);

    EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, TAG_SIZE, const_cast<uint8_t*>(tag_ptr));

    int ret = EVP_DecryptFinal_ex(ctx, out_plaintext + out_len1, &out_len2);

    EVP_CIPHER_CTX_free(ctx);

    return ret > 0;
}
