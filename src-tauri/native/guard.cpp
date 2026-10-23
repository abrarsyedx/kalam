// Small native helpers for Kalam; the Rust side is src/native.rs.
//
// Built with C++ exceptions enabled and without assuming that extern "C"
// functions never throw (MSVC: /EHs /EHc-), because both the guarded callback
// and the Rust code around it can unwind through here.

#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>

#ifdef _WIN32
#include <io.h>
#include <string>
#include <windows.h>
#endif

static void copy_message(char *out, size_t out_len, const char *text) {
    if (out == nullptr || out_len == 0) return;
    std::strncpy(out, text != nullptr ? text : "", out_len - 1);
    out[out_len - 1] = '\0';
}

extern "C" {

typedef void (*kalam_callback)(void *);

// Runs f(data). A C++ exception escaping it (for example from whisper.cpp's
// Vulkan backend when the graphics driver fails) is caught here and its
// message copied to msg, instead of unwinding into Rust, which aborts the
// whole app ("Rust cannot catch foreign exceptions"). Returns 0 when f
// returned normally and 1 when it threw. Rust panics are not std::exception
// and pass through unchanged.
int kalam_guarded_call(kalam_callback f, void *data, char *msg, size_t msg_len) {
    try {
        f(data);
        return 0;
    } catch (const std::exception &e) {
        copy_message(msg, msg_len, e.what());
        return 1;
    }
}

// For tests: throws a std::runtime_error with this message.
void kalam_test_throw(const char *message) {
    throw std::runtime_error(message != nullptr ? message : "");
}

// Sends the process's standard error (C/C++ output such as whisper.cpp's and
// the graphics backend's messages, and Rust's panic messages) to a file.
// A windowed app started from the Start menu has no console, so this output
// would otherwise be lost. Returns 0 on success.
int kalam_redirect_stderr(const char *path_utf8) {
#ifdef _WIN32
    int n = MultiByteToWideChar(CP_UTF8, 0, path_utf8, -1, nullptr, 0);
    if (n <= 0) return 1;
    std::wstring path(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path_utf8, -1, &path[0], n);
    if (_wfreopen(path.c_str(), L"w", stderr) == nullptr) return 1;
    setvbuf(stderr, nullptr, _IONBF, 0);
    // Rust writes to the Win32 handle, not the C stream.
    intptr_t handle = _get_osfhandle(_fileno(stderr));
    if (handle != -1) SetStdHandle(STD_ERROR_HANDLE, reinterpret_cast<HANDLE>(handle));
    return 0;
#else
    if (std::freopen(path_utf8, "w", stderr) == nullptr) return 1;
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    return 0;
#endif
}

// Memory that new allocations can still get, in MB (Windows: commit limit
// minus commit charge, i.e. RAM plus page file still free). -1 if unknown.
long long kalam_available_memory_mb(void) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) return -1;
    return static_cast<long long>(status.ullAvailPageFile / (1024 * 1024));
#else
    return -1;
#endif
}

} // extern "C"
