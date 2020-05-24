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
#define cudaErrorCheck(stream) __cudaErrorCheck(__func__, __FILE__, __LINE__, stream)
#else
#define cudaErrorCheck(stream)
#endif

void __cudaErrorCheck(char const *const func, const char *const file, int const line, cudaStream_t stream)
{
    auto code = cudaGetLastError();
    if (cudaSuccess == code)
    {
        if (stream == nullptr)
            cudaDeviceSynchronize();
        else
            cudaStreamSynchronize(stream);
        code = cudaGetLastError();
    }
    if (cudaSuccess != code)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, code, cudaGetErrorString(code), func);
        exit(code);
    }
}

/** Round-up integer division. */
int distribute(int workSize, int blockSize)
{
    return (workSize - 1) / blockSize + 1;
}

/** Round-up integer division. */
dim3 distribute(dim3 workSize, dim3 blockSize)
{
    return dim3(
        (workSize.x - 1) / blockSize.x + 1,
        (workSize.y - 1) / blockSize.y + 1,
        (workSize.z - 1) / blockSize.z + 1);
}
