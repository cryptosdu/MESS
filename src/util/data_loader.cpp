#include "data_loader.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <filesystem>

namespace fs = std::filesystem;

DataLoader::DataLoader(const std::string& folder_path) {
    load(folder_path);
}

void DataLoader::load(const std::string& folder_path) {
    std::cout << "[DataLoader] Loading data from: " << folder_path << std::endl;

    fs::path root(folder_path);

    base_data    = load_fvecs((root / "base.fvecs").string(), base_dim, base_num);
    query_data   = load_fvecs((root / "query.fvecs").string(), query_dim, query_num);
    ground_truth = load_ivecs((root / "gt.ivecs").string(), gt_dim, gt_num);

    fs::path learn_path = root / "learn.fvecs";
    if (fs::exists(learn_path)) {
        learn_data = load_fvecs(learn_path.string(), learn_dim, learn_num);
    }

    print_stats();
}

void DataLoader::load_fvecs(const std::string& filename,
                            std::vector<float>& data,
                            int& num,
                            int& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    in.read(reinterpret_cast<char*>(&dim), sizeof(int));

    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    size_t vec_size = sizeof(int) + dim * sizeof(float);
    num = (int)(file_size / vec_size);

    data.resize((size_t)num * dim);

    in.seekg(0, std::ios::beg);
    for (int i = 0; i < num; ++i) {
        int d;
        in.read(reinterpret_cast<char*>(&d), sizeof(int));
        in.read(reinterpret_cast<char*>(data.data() + (size_t)i * dim),
                (size_t)dim * sizeof(float));
    }
    in.close();
}

const float* DataLoader::get_base_vector(int index) const {
    if (index < 0 || index >= base_num) throw std::out_of_range("Base index out of range");
    return &base_data[index * base_dim];
}

const float* DataLoader::get_query_vector(int index) const {
    if (index < 0 || index >= query_num) throw std::out_of_range("Query index out of range");
    return &query_data[index * query_dim];
}

const int* DataLoader::get_ground_truth_vector(int index) const {
    if (index < 0 || index >= gt_num) throw std::out_of_range("GroundTruth index out of range");
    return &ground_truth[index * gt_dim];
}

std::vector<float> DataLoader::load_fvecs(const std::string& filename, int& dim, int& num) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    in.read((char*)&dim, 4);

    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    size_t vector_size = 4 + dim * 4;
    num = (int)(file_size / vector_size);

    in.seekg(0, std::ios::beg);

    std::vector<float> data(num * dim);
    for (int i = 0; i < num; ++i) {
        int d;
        in.read((char*)&d, 4);
        if (d != dim) throw std::runtime_error("Dimension mismatch in file: " + filename);
        in.read((char*)(data.data() + i * dim), dim * 4);
    }
    return data;
}

std::vector<int> DataLoader::load_ivecs(const std::string& filename, int& dim, int& num) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    in.read((char*)&dim, 4);
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    size_t vector_size = 4 + dim * 4;
    num = (int)(file_size / vector_size);
    in.seekg(0, std::ios::beg);

    std::vector<int> data(num * dim);
    for (int i = 0; i < num; ++i) {
        int d;
        in.read((char*)&d, 4);
        if (d != dim) throw std::runtime_error("Dimension mismatch in file: " + filename);
        in.read((char*)(data.data() + i * dim), dim * 4);
    }
    return data;
}

void DataLoader::print_stats() {
    std::cout << "=== Data Stats ===" << std::endl;
    std::cout << " Base:  " << base_num << " x " << base_dim << std::endl;
    std::cout << " Query: " << query_num << " x " << query_dim << std::endl;
    std::cout << " Truth: " << gt_num << " x " << gt_dim << std::endl;
    if (learn_num > 0) std::cout << " Learn: " << learn_num << " x " << learn_dim << std::endl;
    std::cout << "==================" << std::endl;
}
