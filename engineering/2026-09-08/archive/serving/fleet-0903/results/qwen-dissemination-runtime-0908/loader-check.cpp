
#include "ggml.h"
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <string>

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    if (!std::strcmp(argv[1], "pinned")) ggml_backend_load_all_from_path(argv[2]);
    else if (!std::strcmp(argv[1], "private")) ggml_backend_load_all();
    else return 2;
    llama_backend_init();
    Dl_info llama_info{};
    if (!dladdr(reinterpret_cast<void *>(llama_backend_init), &llama_info)) return 4;
    std::printf("LLAMA_SYMBOL %s\n", llama_info.dli_fname);
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void *>(ggml_backend_cpu_init), &info)) return 3;
    std::printf("CPU_SYMBOL %s\n", info.dli_fname);
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        std::printf("DEVICE %s\n", ggml_backend_dev_name(ggml_backend_dev_get(i)));
    }
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        if (line.find("/libllama.so.") != std::string::npos) {
            std::printf("LLAMA_MAP %s\n", line.c_str());
        }
        if (line.find("/libggml-cpu.so.") != std::string::npos) {
            std::printf("CPU_MAP %s\n", line.c_str());
        }
    }
    return 0;
}
