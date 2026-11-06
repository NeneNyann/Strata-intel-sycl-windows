#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/weights.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <set>

namespace strata::core {
namespace {
int g_layer_lb = -1, g_layer_le = -1;   // set_layer_range; -1: every layer
bool in_range(const std::string& name) {
    if (g_layer_lb < 0 || name.rfind("blk.", 0) != 0) return true;
    const int l = std::atoi(name.c_str() + 4);
    return l >= g_layer_lb && l < g_layer_le;
}
bool eligible(const strata::TensorInfo& tensor, bool include_ple_key) {
    const auto& name = tensor.name;
    if (name.rfind("blk.", 0) != 0) return false;
    // Match the native PLE kernel: Q2_0, IQ3_XXS, IQ4_XS and Q8_0 (UD-Q4_K_XL). Other keys retain the packed BF16
    // fallback.
    if (name == "blk.1.ple_key.weight")
        return include_ple_key && (tensor.type == 42 || tensor.type == 18 || tensor.type == 23 || tensor.type == 8);
    static const char* suffixes[] = {".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight",
        ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_output.weight",
        ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight", ".ffn_down_shexp.weight"};
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
// The native dense matrices (one allocation each, ~300 of them) are carved out of a few big blocks. On an Arc A-series card
// (i915) every kernel launch costs the host time proportional to the number of live device allocations (12 us with none,
// 131 us with 300, 1.8 ms with 3000) and the kernels themselves run slower; the allocation size does not matter.
// STRATA_DENSE_SLAB_MIB=<n>: block size in MiB (default 256 under the i915 driver, 0 = one allocation per matrix, which is
// what every other driver does); matrices of half a block or more keep their own allocation.
struct DenseSlab {
    struct Block { void* base; size_t size; size_t used; };
    std::mutex mu;
    std::vector<Block> blocks;
};
DenseSlab& dense_slab() { static DenseSlab s; return s; }
size_t dense_slab_bytes() {
    static const size_t v = [] {
        if (const char* e = std::getenv("STRATA_DENSE_SLAB_MIB")) return (size_t) (std::atol(e) > 0 ? std::atol(e) : 0) << 20;
        // on by default where it is measured: an Arc A-series card under the i915 driver (A750: 4K-token prompt 112.6 ->
        // 185.6 tok/s, output bit-identical); other drivers and cards keep one allocation per matrix
        return strata::intel_gpu_driver() == "i915" ? ((size_t) 256 << 20) : (size_t) 0;
    }();
    return v;
}
bool dense_slab_owns(const void* p) {
    DenseSlab& s = dense_slab();
    std::lock_guard<std::mutex> lk(s.mu);
    for (const auto& b : s.blocks) if ((uintptr_t) p >= (uintptr_t) b.base && (uintptr_t) p < (uintptr_t) b.base + b.size) return true;
    return false;
}
void* dense_alloc(size_t bytes) {
    const size_t sb = dense_slab_bytes();
    const size_t need = (bytes + 255) & ~(size_t) 255;
    if (sb == 0 || need >= sb / 2) return strata::malloc_device_guarded(bytes, dpct::get_in_order_queue());
    DenseSlab& s = dense_slab();
    std::lock_guard<std::mutex> lk(s.mu);
    for (auto& b : s.blocks)   // first fit: a gap left by a big matrix is filled by a later small one
        if (b.size - b.used >= need) { void* p = (uint8_t*) b.base + b.used; b.used += need; return p; }
    void* base = strata::malloc_device_guarded(sb, dpct::get_in_order_queue(), "native dense block");
    if (base == nullptr) return nullptr;
    s.blocks.push_back({base, sb, need});
    return base;
}
void dense_slab_free_blocks() {
    DenseSlab& s = dense_slab();
    std::lock_guard<std::mutex> lk(s.mu);
    for (auto& b : s.blocks) { strata::device_alloc_unregister(b.base); sycl::free(b.base, dpct::get_in_order_queue()); }
    s.blocks.clear();
}
struct DeviceFree {
    void
    operator()(void *p) const {
        if (p && !dense_slab_owns(p)) sycl::free(p, dpct::get_in_order_queue());
    }
};
using DevicePtr = std::unique_ptr<void, DeviceFree>;
struct Pending {
    WeightRef* ref;
    int type;
    uint64_t bytes;
    DevicePtr data;
};
// The block with the most unused room is copied into an allocation of exactly its used size (and freed): the VRAM the
// last, part-filled block would waste is the expert cache's (192 slots became 53 on an 8 GB card without this).
// Called before the matrices' pointers are published, so the Pending items are the only holders of them.
bool dense_slab_trim(std::vector<Pending>& pending) {
    DenseSlab& s = dense_slab();
    size_t at = s.blocks.size(), best = 0;
    for (size_t i = 0; i < s.blocks.size(); ++i)
        if (s.blocks[i].size - s.blocks[i].used > best) { best = s.blocks[i].size - s.blocks[i].used; at = i; }
    if (at == s.blocks.size() || best < ((size_t) 4 << 20)) return true;
    DenseSlab::Block old = s.blocks[at];
    void* fresh = strata::malloc_device_guarded(old.used, dpct::get_in_order_queue(), "native dense block");
    if (fresh == nullptr) return true;   // keep the part-filled block
    dpct::get_in_order_queue().memcpy(fresh, old.base, old.used).wait();
    auto rebase = [&](DevicePtr& d) {
        const uintptr_t a = (uintptr_t) d.get();
        if (d && a >= (uintptr_t) old.base && a < (uintptr_t) old.base + old.used)
            (void) d.release(), d.reset((uint8_t*) fresh + (a - (uintptr_t) old.base));
    };
    for (auto& item : pending) rebase(item.data);
    {
        std::lock_guard<std::mutex> lk(s.mu);
        s.blocks[at] = {fresh, old.used, old.used};   // the new block now owns those pointers
    }
    strata::device_alloc_unregister(old.base);
    sycl::free(old.base, dpct::get_in_order_queue());
    return true;
}
}

// The preserved Windows verifier does not use the experimental HC Q8 projections.
bool NativeDense::hc_q8_requested() { return false; }

bool NativeDense::served_names(const std::vector<std::string>& shards, bool include_ple_key,
                               std::set<std::string>& out, std::string& err) {
    try {
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            for (const auto& tensor : gguf.tensors())
                if (eligible(tensor, include_ple_key) && strata::kernels::native_mmvq_supported(tensor.type) &&
                    tensor.shape.size() == 2)
                    out.insert(tensor.name);
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}

void NativeDense::set_layer_range(int lb, int le) { g_layer_lb = lb; g_layer_le = le; }
bool NativeDense::keep_unquantized_ple_key(const std::string& pack_dir, std::set<std::string>& skip,
                                           std::string& err) {
    const std::string key = "blk.1.ple_key.weight";
    if (!skip.count(key)) return true;
    int code_bits = -1;
    if (!WeightTable::index_code_bits(pack_dir, key, code_bits, err)) return false;
    if (code_bits == 0) skip.erase(key);
    return true;
}

NativeDense::~NativeDense() {
    if (scratch_) sycl::free(scratch_, dpct::get_in_order_queue());
    for (void *p : weights_)
        if (!dense_slab_owns(p)) DPCT_CHECK_ERROR(sycl::free(p, dpct::get_in_order_queue()));
    dense_slab_free_blocks();
}

bool NativeDense::load(const std::vector<std::string> &shards,
                       WeightTable &table, std::string &err,
                       bool include_ple_key, int64_t layer_lo, int64_t layer_hi) try {
    auto outside = [&](const std::string &name) {   // a blk.<l>. tensor of another stage's layers
        if (layer_hi < 0 || name.rfind("blk.", 0) != 0) return false;
        const long l = std::strtol(name.c_str() + 4, nullptr, 10);
        return l < layer_lo || l >= layer_hi;
    };
    if (scratch_ || !weights_.empty()) { err = "native dense: already loaded"; return false; }
    if (shards.empty()) { err = "native dense: at least one GGUF shard is required"; return false; }
    try {
        std::vector<Pending> pending;
        std::set<std::string> seen;
        int max_in = 0;
        uint64_t total = 0;
        uint64_t split_count = 0, split_tensors = 0;
        std::set<uint64_t> split_numbers;
        bool have_architecture = false;
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            const auto* count = gguf.get("split.count");
            const auto* number = gguf.get("split.no");
            const auto* tensors = gguf.get("split.tensors.count");
            if (gguf.get("general.architecture")) {
                err = strata::check_engine_architecture(gguf);
                if (!err.empty()) return false;
                have_architecture = true;
                if (count && number && tensors && number->u == 0 && count->u > 1) {
                    split_count = count->u;
                    split_tensors = tensors->u;
                }
            } else if (!have_architecture || !split_count || !count || !number || !tensors ||
                       count->u != split_count || number->u == 0 || number->u >= split_count ||
                       tensors->u != split_tensors) {
                err = "native dense: additional shard must match the architecture-validated first shard's split metadata";
                return false;
            }
            if (number && !split_numbers.insert(number->u).second) {
                err = "native dense: duplicate split shard number"; return false;
            }
            std::vector<uint64_t> offsets;
            for (const auto& tensor : gguf.tensors()) offsets.push_back(tensor.offset);
            std::sort(offsets.begin(), offsets.end());
            if (std::adjacent_find(offsets.begin(), offsets.end()) != offsets.end()) {
                err = "native dense: tensor payload offsets overlap"; return false;
            }
            // Validate every directory span, including tensors we do not upload:
            // an ignored tensor must not overlap the native matrix that follows it.
            const uint64_t payload = gguf.file_size() - gguf.data_start();
            for (const auto& tensor : gguf.tensors()) {
                int block_elements = 0, block_bytes = 0;
                uint64_t elements = 1;
                if (tensor.shape.empty() || !strata::block_geometry(tensor.type, block_elements, block_bytes) ||
                    tensor.shape[0] % (uint64_t) block_elements != 0) {
                    err = "native dense: invalid block geometry " + tensor.name; return false;
                }
                for (uint64_t dimension : tensor.shape) {
                    if (!dimension || elements > (std::numeric_limits<uint64_t>::max)() / dimension) {
                        err = "native dense: invalid tensor extent " + tensor.name; return false;
                    }
                    elements *= dimension;
                }
                const uint64_t blocks = elements / (uint64_t) block_elements;
                if (blocks > (std::numeric_limits<uint64_t>::max)() / (uint64_t) block_bytes) {
                    err = "native dense: tensor byte count overflow " + tensor.name; return false;
                }
                const uint64_t bytes = blocks * (uint64_t) block_bytes;
                if (tensor.offset > payload || bytes > payload - tensor.offset) {
                    err = "native dense: truncated payload " + tensor.name; return false;
                }
                const auto next = std::upper_bound(offsets.begin(), offsets.end(), tensor.offset);
                if (next != offsets.end() && bytes > *next - tensor.offset) {
                    err = "native dense: overlapping payload " + tensor.name; return false;
                }
            }
            for (const auto& tensor : gguf.tensors()) {
                if (!eligible(tensor, include_ple_key) || outside(tensor.name)) continue;
                if (!in_range(tensor.name) && tensor.name.find("ple") == std::string::npos) continue;
                if (!seen.insert(tensor.name).second) {
                    err = "native dense: duplicate tensor " + tensor.name; return false;
                }
                auto found = table.table_.find(tensor.name);
                if (found == table.table_.end()) {
                    err = "native dense: tensor absent from canonical table: " + tensor.name; return false;
                }
                auto& ref = found->second;
                if (ref.native_data) { err = "native dense: override already attached"; return false; }
                if (!strata::kernels::native_mmvq_supported(tensor.type)) continue;
                // #326: the pack keeps an unquantized (--compat-bf16) key, which the PLE reads from the arena
                if (tensor.name == "blk.1.ple_key.weight" && !ref.quantized()) continue;
                if (!ref.quantized() || tensor.shape.size() != 2 ||
                    ref.ne0 <= 0 || ref.ne0 > INT_MAX || ref.ne1 <= 0 || ref.ne1 > INT_MAX ||
                    tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1) {
                    err = "native dense: incompatible matrix " + tensor.name; return false;
                }
                const auto bytes = strata::kernels::native_mmvq_weight_bytes(
                    tensor.type, (int) ref.ne0, (int) ref.ne1);
                void* allocation = nullptr;
                auto status =
                    DPCT_CHECK_ERROR(allocation = dense_alloc(bytes));
                DevicePtr data(allocation);
                if (status == 0)
                    status = DPCT_CHECK_ERROR(
                        dpct::get_in_order_queue()
                            .memcpy(data.get(), gguf.tensor_data(tensor), bytes)
                            .wait());
                /*
                DPCT1000: Error handling if-stmt was detected but could not
                be rewritten.
                */
                if (status != 0) {
                    /*
                    DPCT1009: SYCL reports errors using exceptions and does
                    not use error codes. Please replace the
                    "get_error_string_dummy(...)" with a real error-handling
                    function.
                    */
                    /*
                    DPCT1001: The statement could not be removed.
                    */
                    err = "native dense upload " + tensor.name + ": " +
                          dpct::get_error_string_dummy(status);
                        return false;
                }
                max_in = (std::max)(max_in, (int) ref.ne0);
                total += bytes;
                pending.push_back(Pending{&ref, (int) tensor.type, bytes, std::move(data)});
            }
        }
        if (pending.empty()) { err = "native dense: no supported GDN/QSA matrices in supplied shards"; return false; }
        void* allocation = nullptr;
        const auto status =
            DPCT_CHECK_ERROR(allocation = (void *)strata::malloc_device_guarded(
                                 strata::kernels::native_q8_1_bytes(max_in),
                                 dpct::get_in_order_queue()));
        DevicePtr scratch(allocation);
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001: The statement could not be removed.
        */
        /*
        DPCT1000: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (status != 0) {
            err = std::string("native dense scratch: ") +
                  dpct::get_error_string_dummy(status);
            return false;
        }
        // All checks and allocations finish before publishing any reference.
        dense_slab_trim(pending);
        weights_.reserve(pending.size());
        for (auto& item : pending) {
            item.ref->native_data = item.data.get();
            item.ref->native_type = item.type;
            item.ref->native_q8_1 = scratch.get();
            weights_.push_back(item.data.release());
        }
        scratch_ = scratch.release();
        bytes_ = total;
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
} // namespace strata::core
