cmake_minimum_required(VERSION 3.26)

project(DPHNSW VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)

if(MINGW OR MSYS OR CMAKE_GENERATOR MATCHES "Unix Makefiles" OR CMAKE_GENERATOR MATCHES "Ninja")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")
endif()

add_subdirectory(src/util)

add_executable(client_origin src/clientorigin.cpp)
add_executable(server_origin src/serverorigin.cpp)
add_executable(hnswlib_client src/examples/hnswlib_client.cpp)
add_executable(hnswlib_server src/examples/hnswlib_server.cpp)
add_executable(update_example src/examples/update_example.cpp)

set(COMMON_LIBS
        Utils
        OpenMP::OpenMP_CXX
        Threads::Threads
        OpenSSL::Crypto
)

target_link_libraries(client_origin PRIVATE ${COMMON_LIBS})
target_link_libraries(server_origin PRIVATE ${COMMON_LIBS})

set(HNSWLIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/util")

if(NOT EXISTS "${HNSWLIB_INCLUDE_DIR}/hnswlib/hnswlib.h")
    message(FATAL_ERROR "Cannot find hnswlib header: ${HNSWLIB_INCLUDE_DIR}/hnswlib/hnswlib.h")
endif()

target_include_directories(hnswlib_client PRIVATE ${HNSWLIB_INCLUDE_DIR})
target_include_directories(hnswlib_server PRIVATE ${HNSWLIB_INCLUDE_DIR})

target_link_libraries(hnswlib_client PRIVATE
        OpenMP::OpenMP_CXX
        Threads::Threads
)

target_link_libraries(hnswlib_server PRIVATE
        OpenMP::OpenMP_CXX
        Threads::Threads
)

target_link_libraries(update_example PRIVATE
        Threads::Threads
)
