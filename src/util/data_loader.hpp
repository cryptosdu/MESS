#pragma once

#include <vector>
#include <string>

class DataLoader {
public:

    std::vector<float> base_data;
    std::vector<float> query_data;
    std::vector<float> learn_data;
    std::vector<int>   ground_truth;

    int base_dim = 0, base_num = 0;
    int query_dim = 0, query_num = 0;
    int learn_dim = 0, learn_num = 0;
    int gt_dim = 0, gt_num = 0;

    DataLoader() = default;

    explicit DataLoader(const std::string& folder_path);

    void load(const std::string& folder_path);

    void load_fvecs(const std::string& filename,
                    std::vector<float>& data,
                    int& num,
                    int& dim);

    const float* get_base_vector(int index) const;
    const float* get_query_vector(int index) const;
    const int* get_ground_truth_vector(int index) const;

private:

    std::vector<float> load_fvecs(const std::string& filename, int& dim, int& num);

    std::vector<int> load_ivecs(const std::string& filename, int& dim, int& num);

    void print_stats();
};
