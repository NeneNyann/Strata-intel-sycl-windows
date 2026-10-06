// src/core/gguf_expert_source.cpp - see the header. Plain C++, no device code.
#include "strata/core/gguf_expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cstring>
#include <fcntl.h>
#ifdef _WIN32
#include <filesystem>
#include <io.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#include <thread>
#include <atomic>

namespace strata::core {

namespace {
constexpr size_t kRing = 512;   // blobs alive at once: the prompt path holds a layer's worth of streamed experts

int open_shard(const std::string& name) {
#ifdef _WIN32
    return _wopen(std::filesystem::u8path(name).c_str(), _O_RDONLY | _O_BINARY | _O_NOINHERIT);
#else
    return ::open(name.c_str(), O_RDONLY | O_CLOEXEC);
#endif
}

void close_shard(int fd) {
#ifdef _WIN32
    _close(fd);
#else
    ::close(fd);
#endif
}

int64_t read_shard_at(int fd, void* dst, size_t bytes, uint64_t offset) {
#ifdef _WIN32
    // Explicit 64-bit offsets, as in the shared GGUF loader. No seek/read race between stager threads.
    OVERLAPPED ov{};
    ov.Offset = (DWORD) offset;
    ov.OffsetHigh = (DWORD) (offset >> 32);
    DWORD got = 0;
    const DWORD count = (DWORD) std::min<size_t>(bytes, MAXDWORD);
    if (!ReadFile((HANDLE) _get_osfhandle(fd), dst, count, &got, &ov))
        return GetLastError() == ERROR_HANDLE_EOF ? 0 : -1;
    return got;
#else
    return ::pread(fd, dst, bytes, (off_t) offset);
#endif
}
}

GgufExpertSource::~GgufExpertSource() { close(); }

void GgufExpertSource::close() {
    for (auto* segment : mirror_segments_) sycl::free(segment, dpct::get_in_order_queue());
    mirror_segments_.clear(); mirror_ = nullptr; mirror_segment_bytes_ = 0;
    mirror_bytes_ = 0; mirror_off_.clear(); layer_first_.clear();
    for (int fd : fds_) if (fd >= 0) close_shard(fd);
    fds_.clear(); names_.clear(); layer_fd_.clear(); ring_.clear(); ring_key_.clear(); where_.clear();
    ring_next_ = 0;
}

bool GgufExpertSource::open(const std::string& shard1, int64_t n_layers, int64_t n_expert, std::string& err) {
    close();
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native || lay.gguf_off.size() < (size_t) (3 * n_layers)) {
        err = "--stream-experts needs a native (IQ) pack whose native_experts.txt carries the GGUF tensor offsets";
        return false;
    }
    for (int64_t l = 0; l < n_layers; ++l)
        for (int r = 0; r < 3; ++r)
            if (lay.gguf_off[(size_t) (3 * l + r)] == 0) {
                err = "--stream-experts: layer " + std::to_string(l) + " has no GGUF offset for its experts";
                return false;
            }
    shard_ = shard1;
    const size_t cut = shard1.find_last_of("/\\");
    dir_ = cut == std::string::npos ? std::string() : shard1.substr(0, cut + 1);
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    layer_fd_.assign((size_t) (3 * n_layers), -1);
    for (int64_t l = 0; l < n_layers; ++l)
        for (int r = 0; r < 3; ++r)
            if (fd_of(l, r, err) < 0) { close(); return false; }
    ring_.resize(kRing);
    for (auto& b : ring_) b.resize((size_t) lay.max_blob);
    ring_key_.assign(kRing, -1);
    return true;
}

// The file of role `role` (0 gate, 1 up, 2 down) of `layer`. ExpertLayout::gguf_file is per layer AND role
// (`3 * layer + role`) since upstream 0.1.31; indexing it by layer alone read a shard-2 layer from shard 1 (Swift 1.5,
// whose layers 13-47 are in shard 2: garbage IQ1_M scales, NaN logits - 2026-10-01).
int GgufExpertSource::fd_of(int64_t layer, int role, std::string& err) {
    const size_t i = (size_t) (3 * layer + role);
    if (layer_fd_[i] >= 0) return fds_[(size_t) layer_fd_[i]];
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::string name = shard_;
    if (lay.gguf_file.size() > i && !lay.gguf_file[i].empty()) name = dir_ + lay.gguf_file[i];
    for (size_t k = 0; k < names_.size(); ++k)
        if (names_[k] == name) { layer_fd_[i] = (int) k; return fds_[k]; }
    const int fd = open_shard(name);
    if (fd < 0) { err = "--stream-experts: cannot open " + name; return -1; }
    names_.push_back(name);
    fds_.push_back(fd);
    layer_fd_[i] = (int) fds_.size() - 1;
    return fd;
}

const uint8_t* GgufExpertSource::blob(int64_t layer, int64_t expert) {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_ || ring_.empty()) return nullptr;
    if (mirror_ != nullptr) {
        const int64_t o = mirror_off_[(size_t) (layer * n_expert_ + expert)];
        if (o >= 0) return mirror_at((uint64_t) o);
    }
    const auto& lay = strata::kernels::cpu::expert_layout();
    const auto& fm = lay.fmt[(size_t) layer];
    const uint64_t blob = lay.bytes[(size_t) layer];
    // the arena loader's gather, for one expert: [gate | up | down] from the three tensors
    const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
    const uint64_t at[3] = {0, fm.up_off, fm.down_off};
    const int64_t key = ((int64_t) layer << 20) | expert;
    size_t slot;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = where_.find(key);
        if (it != where_.end()) { ++reads_; return ring_[it->second].data(); }   // still resident
        slot = ring_next_;
        ring_next_ = (ring_next_ + 1) % ring_.size();
        if (ring_key_[slot] >= 0) where_.erase(ring_key_[slot]);
        ring_key_[slot] = key;
        where_[key] = slot;
    }
    std::vector<uint8_t>& buf = ring_[slot];
    for (int r = 0; r < 3; ++r) {
        const int fd = fds_[(size_t) layer_fd_[(size_t) (3 * layer + r)]];
        const uint64_t src = lay.gguf_off[(size_t) (3 * layer + r)] + per[r] * (uint64_t) expert;
        uint64_t done = 0;
        while (done < per[r]) {
            const int64_t n = read_shard_at(fd, buf.data() + at[r] + done, (size_t) (per[r] - done), src + done);
            if (n <= 0) { std::lock_guard<std::mutex> lk(mu_); where_.erase(key); ring_key_[slot] = -1; return nullptr; }
            done += (uint64_t) n;
        }
    }
    ++reads_;
    return buf.data();
}

int64_t GgufExpertSource::mirror(const std::vector<std::pair<int64_t, int64_t>>& pairs, uint64_t cap, int threads,
                                 std::string& err) {
    if (cap == 0) return 0;
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::vector<std::pair<int64_t, int64_t>> take;
    std::vector<uint64_t> offs;
    uint64_t total = 0;
#ifdef _WIN32
    // B580 Windows: malloc_host(8 GiB) succeeds but 16 GiB returns null. Keep each expert inside one segment.
    const uint64_t segment_bytes = 2ull << 30;
#else
    const uint64_t segment_bytes = cap;
#endif
    for (const auto& [l, e] : pairs) {
        if (l < 0 || l >= n_layers_ || e < 0 || e >= n_expert_) continue;
        const uint64_t b = (lay.bytes[(size_t) l] + 255) / 256 * 256;
        if (b > segment_bytes) { err = "mirror: expert exceeds the host segment size"; return -1; }
        uint64_t at = total;
        const uint64_t used = at % segment_bytes;
        if (used + b > segment_bytes) at += segment_bytes - used;
        if (at > cap || b > cap - at) break;
        take.push_back({l, e});
        offs.push_back(at);
        total = at + b;
    }
    if (take.empty()) return 0;
    std::vector<uint8_t*> segments;
    auto release_segments = [&] {
        for (auto* segment : segments) sycl::free(segment, dpct::get_in_order_queue());
    };
    try {
        for (uint64_t at = 0; at < total; at += segment_bytes) {
            auto* segment = (uint8_t*) sycl::malloc_host(std::min(segment_bytes, total - at), dpct::get_in_order_queue());
            if (!segment) {
                release_segments();
                err = "mirror: no pinned host memory for segment " + std::to_string(segments.size());
                return -1;
            }
            segments.push_back(segment);
        }
    } catch (const sycl::exception& ex) {
        release_segments();
        err = std::string("mirror: ") + ex.what();
        return -1;
    }
    std::atomic<size_t> next{0};
    std::atomic<bool> bad{false};
    std::vector<std::thread> ts;
    for (int i = 0; i < std::max(1, threads); ++i)
        ts.emplace_back([&] {
            for (size_t j; (j = next.fetch_add(1)) < take.size() && !bad.load();) {
                const auto [l, e] = take[j];
                auto* dst = segments[offs[j] / segment_bytes] + offs[j] % segment_bytes;
                if (!read_into(l, e, dst, (size_t) lay.bytes[(size_t) l])) bad.store(true);
            }
        });
    for (auto& th : ts) th.join();
    if (bad.load()) {
        release_segments();
        err = "mirror: reading an expert from the GGUF failed";
        return -1;
    }
    for (auto* segment : mirror_segments_) sycl::free(segment, dpct::get_in_order_queue());
    mirror_segments_ = std::move(segments);
    mirror_segment_bytes_ = segment_bytes;
    mirror_ = mirror_segments_.front();
    mirror_bytes_ = total;
    mirror_off_.assign((size_t) (n_layers_ * n_expert_), -1);
    layer_first_.assign((size_t) n_layers_, -1);
    for (size_t j = 0; j < take.size(); ++j) {
        const auto [l, e] = take[j];
        mirror_off_[(size_t) (l * n_expert_ + e)] = (int64_t) offs[j];
        if (layer_first_[(size_t) l] < 0) layer_first_[(size_t) l] = (int64_t) offs[j];
    }
    return (int64_t) take.size();
}

bool GgufExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (mirror_ == nullptr || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return false;
    return mirror_off_[(size_t) (layer * n_expert_ + expert)] >= 0;
}

uint8_t* GgufExpertSource::mirror_at(uint64_t offset) const {
    return mirror_segments_[offset / mirror_segment_bytes_] + offset % mirror_segment_bytes_;
}

const uint8_t* GgufExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (mirror_ == nullptr || layer < 0 || layer >= n_layers_) return nullptr;
    if (expert >= 0 && expert < n_expert_) {
        const int64_t o = mirror_off_[(size_t) (layer * n_expert_ + expert)];
        if (o >= 0) return mirror_at((uint64_t) o);
    }
    const int64_t f = layer_first_[(size_t) layer];
    return f >= 0 ? mirror_at((uint64_t) f) : nullptr;
}

bool GgufExpertSource::read_into(int64_t layer, int64_t expert, uint8_t* dst, size_t bytes) const {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_ || dst == nullptr) return false;
    const auto& lay = strata::kernels::cpu::expert_layout();
    const auto& fm = lay.fmt[(size_t) layer];
    const uint64_t blob = lay.bytes[(size_t) layer];
    if (bytes < blob) return false;
    const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
    const uint64_t at[3] = {0, fm.up_off, fm.down_off};
    for (int r = 0; r < 3; ++r) {
        const int fd = fds_[(size_t) layer_fd_[(size_t) (3 * layer + r)]];
        const uint64_t src = lay.gguf_off[(size_t) (3 * layer + r)] + per[r] * (uint64_t) expert;
        uint64_t done = 0;
        while (done < per[r]) {
            const int64_t n = read_shard_at(fd, dst + at[r] + done, (size_t) (per[r] - done), src + done);
            if (n <= 0) return false;
            done += (uint64_t) n;
        }
    }
    return true;
}

}  // namespace strata::core
