#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <linux/magic.h>
#include <openssl/evp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#include "nlohmann/json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using Counts = std::array<uint64_t, 4>;
static volatile sig_atomic_t interrupted = 0;
static void on_signal(int) { interrupted = 1; }

struct Options {
    bool apply = false;
    bool inventory = false;
    bool deepseek_cache = false;
    uint64_t sample_pages = 4096;
    uint64_t batch_pages = 4096;
    uint64_t max_move_bytes = 64ULL << 30;
    uint64_t max_seconds = 1200;
    std::string fixture;
    std::string output;
};

static void require(bool ok, const std::string &message) {
    if (!ok) throw std::runtime_error(message);
}

static std::string contents(const fs::path &path) {
    std::ifstream input(path);
    require(input.good(), "cannot read " + path.string());
    return std::string(std::istreambuf_iterator<char>(input), {});
}

struct Mapping {
    fs::path path;
    int fd = -1;
    struct stat original {};
    uint8_t *data = nullptr;
    uint64_t pages = 0;
    std::string expected_sha256;
    size_t page_size;
    Mapping(const fs::path &p, uint64_t expected_size, size_t ps) : path(p), page_size(ps) {
        fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        require(fd >= 0, "open failed: " + path.string() + ": " + strerror(errno));
        try {
            require(fstat(fd, &original) == 0, "fstat failed");
            require(S_ISREG(original.st_mode), "target is not a regular file");
            require(original.st_uid == geteuid(), "file is not owned by effective UID");
            require(uint64_t(original.st_size) == expected_size, "unexpected file size: " + path.string());
            struct statfs filesystem {};
            require(fstatfs(fd, &filesystem) == 0 && filesystem.f_type == TMPFS_MAGIC, "target is not tmpfs");
            void *mapped = mmap(nullptr, original.st_size, PROT_READ, MAP_SHARED, fd, 0);
            require(mapped != MAP_FAILED, "read-only shared mmap failed");
            data = static_cast<uint8_t *>(mapped);
            pages = (original.st_size + page_size - 1) / page_size;
        } catch (...) { close(fd); fd = -1; throw; }
    }
    ~Mapping() {
        if (data) munmap(data, original.st_size);
        if (fd >= 0) close(fd);
    }
    void stable() const {
        struct stat current {}, named {};
        require(fstat(fd, &current) == 0 && lstat(path.c_str(), &named) == 0, "file disappeared");
        for (const auto *s : {&current, &named}) {
            require(s->st_dev == original.st_dev && s->st_ino == original.st_ino && s->st_size == original.st_size &&
                    s->st_uid == original.st_uid && s->st_mtim.tv_sec == original.st_mtim.tv_sec &&
                    s->st_mtim.tv_nsec == original.st_mtim.tv_nsec && s->st_ctim.tv_sec == original.st_ctim.tv_sec &&
                    s->st_ctim.tv_nsec == original.st_ctim.tv_nsec,
                    "file identity, size or modification metadata changed: " + path.string());
        }
    }
};

static void no_llama_mappings(const std::vector<std::unique_ptr<Mapping>> &files) {
    for (const auto &entry : fs::directory_iterator("/proc")) {
        const std::string pid = entry.path().filename();
        if (pid.empty() || !std::all_of(pid.begin(), pid.end(), ::isdigit)) continue;
        std::ifstream comm_file(entry.path() / "comm");
        std::string comm;
        std::getline(comm_file, comm);
        if (comm.find("llama-server") == std::string::npos) continue;
        std::ifstream maps(entry.path() / "maps");
        require(maps.good(), "cannot inspect maps of llama-server PID " + pid);
        std::string line;
        while (std::getline(maps, line)) {
            std::istringstream fields(line);
            std::string address, perms, offset, dev;
            uint64_t inode = 0;
            fields >> address >> perms >> offset >> dev >> inode;
            unsigned maj = 0, min = 0;
            if (sscanf(dev.c_str(), "%x:%x", &maj, &min) != 2) continue;
            for (const auto &file : files) {
                require(!(inode == file->original.st_ino && maj == major(file->original.st_dev) && min == minor(file->original.st_dev)),
                        "llama-server PID " + pid + " maps target " + file->path.string());
            }
        }
    }
}

static std::vector<uint64_t> sample_offsets(const Mapping &file, uint64_t count) {
    count = std::min(count, file.pages);
    std::vector<uint64_t> result;
    result.reserve(count);
    for (uint64_t i = 0; i < count; ++i) result.push_back(count == 1 ? 0 : i * (file.pages - 1) / (count - 1));
    return result;
}

static std::string sample_hash(const Mapping &file) {
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    require(ctx != nullptr, "digest allocation failed");
    EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);
    for (uint64_t page : sample_offsets(file, 1024)) {
        const uint64_t offset = page * file.page_size;
        const size_t bytes = std::min<uint64_t>(file.page_size, file.original.st_size - offset);
        EVP_DigestUpdate(ctx, &offset, sizeof(offset));
        EVP_DigestUpdate(ctx, file.data + offset, bytes);
    }
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned length = 0;
    EVP_DigestFinal_ex(ctx, digest, &length);
    EVP_MD_CTX_free(ctx);
    static const char hex[] = "0123456789abcdef";
    std::string result;
    for (unsigned i = 0; i < length; ++i) { result += hex[digest[i] >> 4]; result += hex[digest[i] & 15]; }
    return result;
}

static std::string full_hash(const Mapping &file) {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned length = 0;
    require(EVP_Digest(file.data, file.original.st_size, digest, &length, EVP_sha256(), nullptr) == 1,
            "full digest failed");
    static const char hex[] = "0123456789abcdef";
    std::string result;
    for (unsigned i = 0; i < length; ++i) { result += hex[digest[i] >> 4]; result += hex[digest[i] & 15]; }
    return result;
}

struct Runner {
    Options options;
    std::vector<std::unique_ptr<Mapping>> files;
    json report;
    Clock::time_point start = Clock::now(), next_log = start, next_guard = start;
    uint64_t moved_bytes = 0;
    volatile uint8_t touched = 0;

    void checkpoint(const std::string &phase, uint64_t done, uint64_t total) {
        require(!interrupted, "interrupted; completed page moves are retained");
        require(std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - start).count() < int64_t(options.max_seconds),
                "time limit reached; completed page moves are retained");
        if (Clock::now() >= next_guard) {
            for (const auto &file : files) file->stable();
            no_llama_mappings(files);
            next_guard = Clock::now() + std::chrono::seconds(1);
        }
        if (Clock::now() >= next_log) {
            std::cerr << json{{"phase", phase}, {"pages_done", done}, {"pages_total", total}, {"moved_bytes", moved_bytes}}.dump() << '\n';
            next_log = Clock::now() + std::chrono::seconds(15);
        }
    }

    std::vector<int> query(Mapping &file, const std::vector<uint64_t> &pages, json &record) {
        std::vector<void *> pointers;
        pointers.reserve(pages.size());
        for (uint64_t page : pages) {
            auto *ptr = file.data + page * file.page_size;
            touched = uint8_t(touched ^ *static_cast<volatile uint8_t *>(ptr));
            pointers.push_back(ptr);
        }
        std::vector<int> status(pages.size(), -9999);
        long rc = syscall(SYS_move_pages, 0, pointers.size(), pointers.data(), nullptr, status.data(), 0);
        record["query_syscalls"] = record.value("query_syscalls", 0ULL) + 1;
        record["queried_pages"] = record.value("queried_pages", 0ULL) + pages.size();
        require(rc >= 0, "move_pages query failed: " + std::string(strerror(errno)));
        return status;
    }

    Counts inventory(Mapping &file, json &record, bool full, const std::string &phase) {
        Counts counts {};
        const uint64_t n = full ? file.pages : std::min(file.pages, options.sample_pages);
        const auto sampled = full ? std::vector<uint64_t>() : sample_offsets(file, n);
        uint64_t errors = 0;
        for (uint64_t begin = 0; begin < n; begin += options.batch_pages) {
            checkpoint(phase, begin, n);
            std::vector<uint64_t> batch;
            for (uint64_t i = begin; i < std::min(n, begin + options.batch_pages); ++i) batch.push_back(full ? i : sampled[i]);
            for (int node : query(file, batch, record)) {
                if (node >= 0 && node < 4) ++counts[node];
                else { ++errors; const auto key = std::to_string(node); record["query_status_errors"][key] = record["query_status_errors"].value(key, 0ULL) + 1; }
            }
        }
        record[phase] = {{"pages_queried", n}, {"complete_inventory", full}, {"node_pages", counts}, {"errors", errors}};
        if (full) require(errors == 0, "complete placement inventory contains query errors or nodes outside 0-3");
        return counts;
    }

    void balance(Mapping &file, json &record, Counts counts) {
        Counts targets;
        for (size_t node = 0; node < 4; ++node) targets[node] = file.pages / 4 + (node < file.pages % 4);
        uint64_t minimum_moves = 0;
        for (size_t node = 0; node < 4; ++node) if (counts[node] > targets[node]) minimum_moves += counts[node] - targets[node];
        record["target_node_pages"] = targets;
        record["minimum_pages_to_move"] = minimum_moves;
        require(minimum_moves * file.page_size <= options.max_move_bytes - moved_bytes,
                "planned moves exceed remaining --max-move-gib budget; no moves made for this file");
        for (uint64_t begin = 0; begin < file.pages; begin += options.batch_pages) {
            if (counts == targets) break;
            checkpoint("migrate", begin, file.pages);
            std::vector<uint64_t> batch;
            for (uint64_t i = begin; i < std::min(file.pages, begin + options.batch_pages); ++i) batch.push_back(i);
            const auto placement = query(file, batch, record);
            Counts planned = counts;
            std::vector<void *> pointers;
            std::vector<int> destinations, sources;
            for (size_t i = 0; i < batch.size(); ++i) {
                int source = placement[i];
                require(source >= 0 && source < 4, "page query failed during migration scan");
                if (planned[source] <= targets[source]) continue;
                int destination = -1;
                uint64_t greatest_deficit = 0;
                for (int node = 0; node < 4; ++node) {
                    if (planned[node] < targets[node] && targets[node] - planned[node] > greatest_deficit) {
                        greatest_deficit = targets[node] - planned[node]; destination = node;
                    }
                }
                if (destination < 0) continue;
                --planned[source]; ++planned[destination];
                pointers.push_back(file.data + batch[i] * file.page_size);
                destinations.push_back(destination); sources.push_back(source);
            }
            if (pointers.empty()) continue;
            std::vector<int> status(pointers.size(), -9999);
            // pid=0 and flags=0: only this mapping, never privileged MOVE_ALL.
            long rc = syscall(SYS_move_pages, 0, pointers.size(), pointers.data(), destinations.data(), status.data(), 0);
            record["move_syscalls"] = record.value("move_syscalls", 0ULL) + 1;
            record["attempted_pages"] = record.value("attempted_pages", 0ULL) + pointers.size();
            require(rc >= 0, "move_pages migration failed: " + std::string(strerror(errno)));
            for (size_t i = 0; i < status.size(); ++i) {
                if (status[i] == destinations[i]) {
                    --counts[sources[i]]; ++counts[destinations[i]];
                    moved_bytes += file.page_size;
                    record["moved_pages"] = record.value("moved_pages", 0ULL) + 1;
                } else {
                    const auto key = std::to_string(status[i]);
                    record["move_status_errors"][key] = record["move_status_errors"].value(key, 0ULL) + 1;
                }
            }
            record["tracked_node_pages"] = counts;
        }
        const Counts after = inventory(file, record, true, "after");
        record["balanced"] = after == targets;
        require(after == targets, "placement remains uneven after one bounded pass; inspect errors before an explicit retry");
    }

    void run() {
        require(geteuid() != 0, "do not run this utility as root");
        const long ps = sysconf(_SC_PAGESIZE);
        require(ps == 4096, "this reviewed implementation requires the host 4096-byte page size");
        for (int node = 0; node < 4; ++node) require(fs::exists("/sys/devices/system/node/node" + std::to_string(node)), "expected NUMA node is absent");
        if (!options.fixture.empty()) {
            const fs::path fixture = fs::absolute(options.fixture).lexically_normal();
            require(fixture.parent_path() == "/dev/shm" && fixture.filename().string().rfind("numa-rebalance-fixture-", 0) == 0,
                    "fixture must be /dev/shm/numa-rebalance-fixture-NAME");
            struct stat st {};
            require(lstat(fixture.c_str(), &st) == 0 && st.st_size >= (2 << 20) && st.st_size <= (8 << 20), "fixture must be 2-8 MiB");
            files.emplace_back(new Mapping(fixture, st.st_size, ps));
        } else if (options.deepseek_cache) {
            const fs::path root = "/dev/shm/deepseek-v41-native-fb2764-0910";
            const std::string revision = "fb2764a5cf321eaa5070ca8f9e892818f477c16d";
            require(!fs::is_symlink(root), "cache root must not be a symlink");
            const auto owner = json::parse(contents(root / "owner.json"));
            require(owner.at("task") == "deepseek-v41-native-0910" && owner.at("revision") == revision,
                    "DeepSeek cache ownership or checkpoint revision differs");
            std::istringstream manifest(contents(root / "tensors.jsonl"));
            std::string line;
            std::vector<std::string> names;
            while (std::getline(manifest, line)) {
                if (line.empty()) continue;
                const auto record = json::parse(line);
                const std::string name = record.at("name");
                const uint64_t size = record.at("bytes");
                if (size < (1ULL << 20)) continue;
                require(record.at("revision") == revision &&
                        std::regex_match(name, std::regex("[A-Za-z0-9_]+(\\.[A-Za-z0-9_]+)*")),
                        "invalid native cache tensor record");
                require(std::find(names.begin(), names.end(), name) == names.end(), "duplicate native cache tensor");
                names.push_back(name);
                const std::string hash = record.at("sha256");
                require(std::regex_match(hash, std::regex("[0-9a-f]{64}")), "invalid cache content digest");
                files.emplace_back(new Mapping(root / (name + ".bin"), size, ps));
                files.back()->expected_sha256 = hash;
            }
            require(!files.empty() && files.size() <= 12000, "unexpected native cache inventory size");
            report["target"] = "manifested native DeepSeek cache tensors at least 1 MiB";
            report["checkpoint_revision"] = revision;
        } else {
            const std::string directory = "/home/user/.local/share/ai-models/GLM-5.3-Flash-621d456e93e9/UD-Q4_K_XL/";
            const std::array<uint64_t, 6> sizes = {9429859ULL, 49198039200ULL, 49988626304ULL, 48700701344ULL, 49026026752ULL, 2784497888ULL};
            for (size_t i = 0; i < sizes.size(); ++i) {
                const std::string path = directory + "GLM-5.3-Flash-UD-Q4_K_XL-0000" + std::to_string(i + 1) + "-of-00006.gguf";
                files.emplace_back(new Mapping(path, sizes[i], ps));
            }
        }
        no_llama_mappings(files);
        report["mode"] = options.apply ? "apply" : (options.inventory ? "inventory" : "sample");
        report["page_size"] = ps;
        report["euid"] = geteuid();
        report["move_pages_pid"] = 0;
        report["move_pages_flags"] = 0;
        report["sample_integrity_scope"] = "SHA-256 of up to 1024 evenly spaced pages per file, including page offsets; not a full-file digest";
        report["files"] = json::array();
        std::vector<Counts> before;
        for (auto &file : files) {
            report["files"].push_back({{"path", file->path.string()}, {"size", file->original.st_size}, {"inode", file->original.st_ino},
                                      {"device", file->original.st_dev}, {"pages", file->pages}, {"query_status_errors", json::object()},
                                      {"move_status_errors", json::object()}});
            auto &record = report["files"].back();
            if (!file->expected_sha256.empty()) {
                record["manifest_sha256"] = file->expected_sha256;
                record["full_sha256_before"] = full_hash(*file);
                require(record["full_sha256_before"] == file->expected_sha256,
                        "native cache file does not match its recorded full digest: " + file->path.string());
            }
            record["sample_sha256_before"] = sample_hash(*file);
            before.push_back(inventory(*file, record, options.apply || options.inventory, "before"));
        }
        if (options.apply) {
            uint64_t planned_bytes = 0;
            for (size_t i = 0; i < files.size(); ++i) {
                for (size_t node = 0; node < 4; ++node) {
                    uint64_t target = files[i]->pages / 4 + (node < files[i]->pages % 4);
                    if (before[i][node] > target) planned_bytes += (before[i][node] - target) * ps;
                }
            }
            report["planned_minimum_move_bytes"] = planned_bytes;
            require(planned_bytes <= options.max_move_bytes, "total plan exceeds --max-move-gib; no model pages migrated");
            no_llama_mappings(files);
            for (size_t i = 0; i < files.size(); ++i) balance(*files[i], report["files"][i], before[i]);
        }
    }

    void finish() {
        report["moved_bytes"] = moved_bytes;
        report["elapsed_seconds"] = std::chrono::duration<double>(Clock::now() - start).count();
        if (!report.contains("files")) return;
        for (size_t i = 0; i < report["files"].size(); ++i) {
            auto &record = report["files"][i];
            files[i]->stable();
            if (!files[i]->expected_sha256.empty()) {
                record["full_sha256_after"] = full_hash(*files[i]);
                record["full_bytes_unchanged"] = record["full_sha256_after"] == files[i]->expected_sha256;
                require(record["full_bytes_unchanged"].get<bool>(), "full native cache digest changed");
            }
            record["sample_sha256_after"] = sample_hash(*files[i]);
            record["sample_bytes_unchanged"] = record["sample_sha256_before"] == record["sample_sha256_after"];
            require(record["sample_bytes_unchanged"].get<bool>(), "sample content digest changed");
        }
    }
};

int main(int argc, char **argv) {
    Runner runner;
    int result = 0;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto value = [&]() { require(i + 1 < argc, "missing value for " + arg); return std::string(argv[++i]); };
            if (arg == "--apply") runner.options.apply = true;
            else if (arg == "--inspect") {}
            else if (arg == "--inventory") runner.options.inventory = true;
            else if (arg == "--deepseek-cache") runner.options.deepseek_cache = true;
            else if (arg == "--fixture") runner.options.fixture = value();
            else if (arg == "--json") runner.options.output = value();
            else if (arg == "--sample-pages") runner.options.sample_pages = std::stoull(value());
            else if (arg == "--batch-pages") runner.options.batch_pages = std::stoull(value());
            else if (arg == "--max-move-gib") {
                const uint64_t gib = std::stoull(value());
                require(gib <= 256, "max-move-gib must be 0..256");
                runner.options.max_move_bytes = gib << 30;
            }
            else if (arg == "--max-seconds") runner.options.max_seconds = std::stoull(value());
            else throw std::runtime_error("unknown argument " + arg);
        }
        require(runner.options.sample_pages > 0 && runner.options.sample_pages <= 1048576, "sample-pages must be 1..1048576");
        require(runner.options.batch_pages > 0 && runner.options.batch_pages <= 65536, "batch-pages must be 1..65536");
        require(runner.options.max_seconds > 0 && runner.options.max_seconds <= 7200, "max-seconds must be 1..7200");
        require(runner.options.max_move_bytes <= (256ULL << 30), "max-move-gib must be 0..256");
        require(!runner.options.deepseek_cache || runner.options.fixture.empty(), "choose one target inventory");
        signal(SIGINT, on_signal); signal(SIGTERM, on_signal);
        runner.run();
        runner.report["completed"] = true;
    } catch (const std::exception &error) {
        runner.report["completed"] = false;
        runner.report["error"] = error.what();
        std::cerr << error.what() << '\n';
        result = 1;
    }
    try { runner.finish(); }
    catch (const std::exception &error) { runner.report["integrity_error"] = error.what(); runner.report["completed"] = false; result = 1; }
    const std::string serialized = runner.report.dump(2) + "\n";
    std::cout << serialized;
    if (!runner.options.output.empty()) {
        // Never truncate or follow an existing path, including any weight file.
        int output = open(runner.options.output.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (output < 0) { std::cerr << "report path must be new: " << strerror(errno) << '\n'; return 1; }
        size_t done = 0;
        while (done < serialized.size()) {
            const ssize_t n = write(output, serialized.data() + done, serialized.size() - done);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) { close(output); std::cerr << "cannot write report\n"; return 1; }
            done += n;
        }
        close(output);
    }
    return result;
}
