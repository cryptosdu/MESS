#include "securesearchcontext.hpp"

#include <cstddef>
#include <stdexcept>
#include <openssl/evp.h>

SecureSearchContext::SecureSearchContext(const Config& cfg)
    : search_ctx_(cfg.search),
      num_graphs_(cfg.num_graphs),
      input_dim_(cfg.input_dim) {
    if (num_graphs_ <= 0) {
        throw std::invalid_argument("SecureSearchContext: num_graphs must be positive.");
    }

    if (input_dim_ <= 0) {
        throw std::invalid_argument("SecureSearchContext: input_dim must be positive.");
    }

    graph_enc_contexts_.reserve(static_cast<size_t>(num_graphs_));

    if (!cfg.use_pre_shared_master_key) {
        for (int i = 0; i < num_graphs_; ++i) {
            graph_enc_contexts_.emplace_back();
        }
        return;
    }

    if (cfg.master_key.size() != 32) {
        throw std::invalid_argument(
            "SecureSearchContext: master_key must be exactly 32 bytes for AES-256."
        );
    }

    for (int i = 0; i < num_graphs_; ++i) {
        graph_enc_contexts_.emplace_back(derive_graph_key(cfg.master_key, i));
    }
}

std::vector<uint8_t> SecureSearchContext::derive_graph_key(
    const std::vector<uint8_t>& master_key,
    int graph_id
) {

    EVP_MD_CTX* mdctx = EVP_MD_CTX_new();
    if (!mdctx) {
        throw std::runtime_error(
            "SecureSearchContext::derive_graph_key: EVP_MD_CTX_new failed."
        );
    }

    std::vector<uint8_t> out(32, 0);
    unsigned int out_len = 0;

    if (EVP_DigestInit_ex(mdctx, EVP_sha256(), nullptr) != 1) {
        EVP_MD_CTX_free(mdctx);
        throw std::runtime_error(
            "SecureSearchContext::derive_graph_key: EVP_DigestInit_ex failed."
        );
    }

    if (EVP_DigestUpdate(mdctx, master_key.data(), master_key.size()) != 1) {
        EVP_MD_CTX_free(mdctx);
        throw std::runtime_error(
            "SecureSearchContext::derive_graph_key: EVP_DigestUpdate(master_key) failed."
        );
    }

    if (EVP_DigestUpdate(mdctx, &graph_id, sizeof(graph_id)) != 1) {
        EVP_MD_CTX_free(mdctx);
        throw std::runtime_error(
            "SecureSearchContext::derive_graph_key: EVP_DigestUpdate(graph_id) failed."
        );
    }

    if (EVP_DigestFinal_ex(mdctx, out.data(), &out_len) != 1) {
        EVP_MD_CTX_free(mdctx);
        throw std::runtime_error(
            "SecureSearchContext::derive_graph_key: EVP_DigestFinal_ex failed."
        );
    }

    EVP_MD_CTX_free(mdctx);

    if (out_len != 32) {
        throw std::runtime_error(
            "SecureSearchContext::derive_graph_key: unexpected digest length."
        );
    }

    return out;
}

SearchContext& SecureSearchContext::search_options() {
    return search_ctx_;
}

const SearchContext& SecureSearchContext::search_options() const {
    return search_ctx_;
}

int SecureSearchContext::num_graphs() const {
    return num_graphs_;
}

int SecureSearchContext::input_dim() const {
    return input_dim_;
}

void SecureSearchContext::check_graph_id(int graph_id) const {
    if (graph_id < 0 || graph_id >= num_graphs_) {
        throw std::out_of_range("SecureSearchContext: graph_id out of range.");
    }
}

std::vector<uint8_t> SecureSearchContext::encrypt_vector(
    int graph_id,
    const float* vec,
    int dim
) const {
    check_graph_id(graph_id);

    if (vec == nullptr) {
        throw std::invalid_argument(
            "SecureSearchContext::encrypt_vector: input vec is null."
        );
    }

    if (dim <= 0) {
        throw std::invalid_argument(
            "SecureSearchContext::encrypt_vector: dim must be positive."
        );
    }

    const uint8_t* plaintext = reinterpret_cast<const uint8_t*>(vec);
    const size_t byte_len = static_cast<size_t>(dim) * sizeof(float);

    return graph_enc_contexts_[graph_id].encrypt(plaintext, byte_len);
}

std::vector<float> SecureSearchContext::decrypt_vector(
    int graph_id,
    const std::vector<uint8_t>& payload,
    int dim
) const {
    check_graph_id(graph_id);

    if (dim <= 0) {
        throw std::invalid_argument(
            "SecureSearchContext::decrypt_vector: dim must be positive."
        );
    }

    std::vector<float> out(static_cast<size_t>(dim), 0.0f);
    const size_t byte_len = static_cast<size_t>(dim) * sizeof(float);

    const bool ok = graph_enc_contexts_[graph_id].decrypt(
        payload,
        reinterpret_cast<uint8_t*>(out.data()),
        byte_len
    );

    if (!ok) {
        throw std::runtime_error(
            "SecureSearchContext::decrypt_vector: AES-GCM authentication failed."
        );
    }

    return out;
}
