// Level Zero device probe for the Windows installer and release package.
#include <sycl/sycl.hpp>
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
            std::printf("%s{\"index\":%d,\"name\":%s,\"memory_bytes\":%llu,\"integrated\":%s}",
                        index ? "," : "", index, name.c_str(), (unsigned long long) bytes,
                        integrated ? "true" : "false");
            ++index;
        }
    }
    std::printf("]}\n");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "strata-device: %s\n", e.what());
    return 1;
}
