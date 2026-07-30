add_library(LSH STATIC
        iso_lsh.cpp
        iso_lsh.hpp
)

target_include_directories(LSH PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
