// Level Zero device probe for the Windows installer and release package.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <cstdio>
#include <string>

namespace {
std::string json_string(const std::string& value) {
    std::string out = "\"";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') { out += '\\'; out += (char) c; }
        else if (c < 32) {
            char escaped[7];
            std::snprintf(escaped, sizeof escaped, "\\u%04x", c);
            out += escaped;
        } else out += (char) c;
    }
    return out + '"';
}
}

int main() try {
    int index = 0;
    std::printf("{\"devices\":[");
    for (const auto& platform : sycl::platform::get_platforms()) {
        if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& device : platform.get_devices(sycl::info::device_type::gpu)) {
            const auto name = json_string(device.get_info<sycl::info::device::name>());
            const auto bytes = device.get_info<sycl::info::device::global_mem_size>();
            const bool integrated = device.get_info<sycl::info::device::host_unified_memory>();
            std::string pci = "null", uuid = "null", id = "null", alchemist = "null";
            try {
                namespace exp = sycl::ext::oneapi::experimental;
                const auto arch = device.get_info<exp::info::device::architecture>();
                alchemist = arch == exp::architecture::intel_gpu_acm_g10 || arch == exp::architecture::intel_gpu_acm_g11 ||
                            arch == exp::architecture::intel_gpu_acm_g12 ? "true" : "false";
            } catch (const sycl::exception&) {}
            try {
                if (device.has(sycl::aspect::ext_intel_pci_address))
                    pci = json_string(device.get_info<sycl::ext::intel::info::device::pci_address>());
            } catch (const sycl::exception&) {}
            try {
                if (device.has(sycl::aspect::ext_intel_device_id))
                    id = std::to_string(device.get_info<sycl::ext::intel::info::device::device_id>());
            } catch (const sycl::exception&) {}
            try {
                if (device.has(sycl::aspect::ext_intel_device_info_uuid)) {
                    const auto raw = device.get_info<sycl::ext::intel::info::device::uuid>();
                    std::string hex;
                    for (unsigned char byte : raw) {
                        char pair[3];
                        std::snprintf(pair, sizeof pair, "%02x", byte);
                        hex += pair;
                    }
                    uuid = json_string(hex);
                }
            } catch (const sycl::exception&) {}
            std::printf("%s{\"index\":%d,\"name\":%s,\"memory_bytes\":%llu,\"integrated\":%s,"
                        "\"vendor_id\":%u,\"device_id\":%s,\"pci_address\":%s,\"uuid\":%s,"
                        "\"usm_host\":%s,\"usm_device\":%s,\"alchemist\":%s}",
                        index ? "," : "", index, name.c_str(), (unsigned long long) bytes,
                        integrated ? "true" : "false", device.get_info<sycl::info::device::vendor_id>(),
                        id.c_str(), pci.c_str(), uuid.c_str(),
                        device.has(sycl::aspect::usm_host_allocations) ? "true" : "false",
                        device.has(sycl::aspect::usm_device_allocations) ? "true" : "false", alchemist.c_str());
            ++index;
        }
    }
    std::printf("]}\n");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "strata-device: %s\n", e.what());
    return 1;
}
