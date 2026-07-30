add_subdirectory(LSH)

add_library(Utils STATIC
        lsh_encoder.cpp
        lsh_encoder.hpp
        data_paths.hpp
        data_loader.cpp
        data_loader.hpp
        hnsw_index.cpp
        hnsw_index.hpp
        hash_codeword.cpp
        hash_codeword.hpp
        search_context.hpp
        search_context.cpp
        securesearchcontext.cpp
        securesearchcontext.hpp
        enc_context.cpp
        enc_context.hpp
)

find_package(OpenSSL REQUIRED)

target_link_libraries(Utils
        PUBLIC
        OpenSSL::Crypto
        OpenSSL::SSL
        LSH
)

target_include_directories(Utils PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
