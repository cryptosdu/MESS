#pragma once

#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mess {

inline std::filesystem::path find_data_root() {
    namespace fs = std::filesystem;

    if (const char* configured = std::getenv("MESS_DATA_ROOT");
        configured != nullptr && *configured != '\0') {
        const fs::path root(configured);
        if (fs::is_directory(root / "dataset")) {
            return fs::weakly_canonical(root);
        }
        throw std::runtime_error(
            "MESS_DATA_ROOT does not contain a dataset directory: " +
            root.string()
        );
    }

    const fs::path cwd = fs::current_path();
    std::vector<fs::path> candidates = {
        cwd / "data",
        cwd / ".." / "data",
        cwd / ".." / ".." / "data",
    };

#if defined(__linux__)
    std::error_code executable_error;
    const fs::path executable = fs::read_symlink("/proc/self/exe", executable_error);
    if (!executable_error) {
        candidates.push_back(executable.parent_path() / "data");
        candidates.push_back(executable.parent_path() / ".." / "data");
    }
#endif

    const fs::path source_file(__FILE__);
    candidates.push_back(
        source_file.parent_path().parent_path().parent_path() / "data"
    );

    for (const fs::path& candidate : candidates) {
        std::error_code directory_error;
        if (fs::is_directory(candidate / "dataset", directory_error)) {
            return fs::weakly_canonical(candidate);
        }
    }

    std::ostringstream message;
    message << "Cannot find the repository data directory. Checked:";
    for (const fs::path& candidate : candidates) {
        message << "\n  - " << candidate.lexically_normal().string();
    }
    message << "\nRun from the repository root/build directory or set MESS_DATA_ROOT.";
    throw std::runtime_error(message.str());
}

inline std::string data_path(const std::string& relative_path) {
    return (find_data_root() / relative_path).lexically_normal().string();
}

}  // namespace mess
