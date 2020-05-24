#pragma once

template <typename... Args>
std::string string_format(const std::string &format, Args... args)
{
    size_t size = snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
    if (size <= 0)
    {
        throw std::runtime_error("Error during formatting.");
    }
    std::unique_ptr<char[]> buf(new char[size]);
    snprintf(buf.get(), size, format.c_str(), args...);
    return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

#ifdef CUDAERRORCHECKS
#define cudaErrorCheck() __cudaErrorCheck(__func__, __FILE__, __LINE__)
#else
#define cudaErrorCheck()
#endif

void __cudaErrorCheck(char const *const func, const char *const file, int const line)
{
    auto code = cudaGetLastError();
    if (cudaSuccess == code)
    {
        cudaDeviceSynchronize();
        code = cudaGetLastError();
    }
    if (cudaSuccess != code)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, code, cudaGetErrorString(code), func);
        exit(code);
    }
}