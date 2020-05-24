/*
Current version: Artem Chirkin

Based on: Jiahui Huang (https://github.com/heiwang1997/DenseCRF)

Based on: Miguel Monteiro at al (https://github.com/MiguelMonteiro/permutohedral_lattice)

----------------------------------------------------------------------------

Copyright (c) 2018 Miguel Monteiro, Andrew Adams, Jongmin Baek, Abe Davis

Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#define BLOCK_SIZE 256

#include <atomic>
#include "util.hpp"

namespace crf
{

    template <typename T>
    struct MatrixEntry
    {
        int index;
        T weight;
    };

    // GPU version Hash Table (with fixed size)
    template <typename T, int pd, int vd>
    class HashTableGPU
    {
    private:
        const cudaStream_t mainStream;

        /** A primitive way to keep track of the object copies. */
        std::atomic<int> *refCount;

    public:
        const int capacity;
        short *keys;
        int *entries;
        T *values;
        /** When doing some filtering, such as blur, this serves as a temporary memory for values. */
        T *valuesCache;

        __host__ HashTableGPU(const cudaStream_t mainStream, const int capacity)
            : mainStream(mainStream), capacity(capacity), entries(nullptr), keys(nullptr), values(nullptr), valuesCache(nullptr),
              refCount(new std::atomic<int>(1))
        {
            cudaMalloc((void **)&entries, capacity * 2 * sizeof(int));
            cudaErrorCheck(nullptr);
            cudaMalloc((void **)&keys, capacity * pd * sizeof(short));
            cudaErrorCheck(nullptr);
            cudaMalloc((void **)&values, capacity * vd * sizeof(T));
            cudaErrorCheck(nullptr);
            cudaMalloc((void **)&valuesCache, capacity * vd * sizeof(T));
            cudaErrorCheck(nullptr);
        }

        __host__ HashTableGPU(const HashTableGPU &o)
            : mainStream(o.mainStream), capacity(o.capacity), entries(o.entries), keys(o.keys), values(o.values), valuesCache(o.valuesCache), refCount(o.refCount)
        {
            (*refCount)++;
        }

        __host__ ~HashTableGPU()
        {
            if (refCount->fetch_sub(1) > 1)
                return;
            delete refCount;
            cudaFree(keys);
            cudaErrorCheck(nullptr);
            cudaFree(entries);
            cudaErrorCheck(nullptr);
            cudaFree(values);
            cudaErrorCheck(nullptr);
            cudaFree(valuesCache);
            cudaErrorCheck(nullptr);
        }

        void clear()
        {
            cudaMemsetAsync((void *)entries, -1, capacity * 2 * sizeof(int), mainStream);
            cudaErrorCheck(mainStream);
            cudaMemsetAsync((void *)keys, 0, capacity * pd * sizeof(short), mainStream);
            cudaErrorCheck(mainStream);
        }

        __device__ int modHash(unsigned int n)
        {
            return (n % (2 * capacity));
        }

        __device__ unsigned int hash(short *key)
        {
            unsigned int k = 0;
            for (int i = 0; i < pd; i++)
            {
                k += key[i];
                k = k * 2531011;
            }
            return k;
        }

        // Insert key into slot. Return bucket id.
        __device__ int insert(short *key, unsigned int slot)
        {
            int h = modHash(hash(key));
            while (1)
            {
                int *e = entries + h;

                // If the cell is empty (-1), lock it (-2)
                int contents = atomicCAS(e, -1, -2);

                if (contents == -2)
                {
                    // If it was locked already, move on to the next cell
                    // However, we are not sure whether other threads are writing the same key to it.
                    // So a post-processing cleaning step is necessary.
                }
                else if (contents == -1)
                {
                    // If it was empty, we successfully locked it. Write our key.
                    for (int i = 0; i < pd; i++)
                    {
                        keys[slot * pd + i] = key[i];
                    }
                    // Unlock
                    atomicExch(e, slot);
                    return h;
                }
                else
                {
                    // The cell is unlocked and has a key in it, check if it matches
                    bool match = true;
                    for (int i = 0; i < pd && match; i++)
                    {
                        match = (keys[contents * pd + i] == key[i]);
                    }
                    if (match)
                        return h;
                }
                // increment the bucket with wraparound
                h++;
                if (h == capacity * 2)
                    h = 0;
            }
        }

        // Find key. Return slot id.
        __device__ int retrieve(short *key)
        {
            int h = modHash(hash(key));
            while (1)
            {
                int *e = entries + h;

                if (*e == -1)
                    return -1;

                bool match = true;
                for (int i = 0; i < pd && match; i++)
                {
                    match = (keys[(*e) * pd + i] == key[i]);
                }
                if (match)
                    return *e;

                h++;
                if (h == capacity * 2)
                    h = 0;
            }
        }

        __host__ void blur(MatrixEntry<T> *matrix);
    };

    template <typename T, int pd, int vd>
    __global__ void blurKernel(MatrixEntry<T> *matrix, int color, HashTableGPU<T, pd, vd> table)
    {

        const int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= table.capacity)
            return;

        // Check if I'm valid
        if (matrix[idx].index != idx)
            return;

        // find my key and the keys of my neighbors
        short myKey[pd + 1];
        short np[pd + 1];
        short nm[pd + 1];

        for (int i = 0; i < pd; i++)
        {
            myKey[i] = table.keys[idx * pd + i];
            np[i] = myKey[i] + 1;
            nm[i] = myKey[i] - 1;
        }
        np[color] -= pd + 1;
        nm[color] += pd + 1;

        int offNp = table.retrieve(np);
        int offNm = table.retrieve(nm);

        T *valMe = table.values + vd * idx;
        T *valOut = table.valuesCache + vd * idx;

        //in case neighbours don't exist (lattice edges) offNp and offNm are -1
        T *valNp = valMe;
        T *valNm = valMe;
        if (offNp >= 0)
            valNp = table.values + vd * offNp;
        if (offNm >= 0)
            valNm = table.values + vd * offNm;

        for (int i = 0; i < vd; i++)
            valOut[i] = 0.25 * valNp[i] + 0.5 * valMe[i] + 0.25 * valNm[i];
    }

    template <typename T, int pd, int vd>
    __host__ void HashTableGPU<T, pd, vd>::blur(MatrixEntry<T> *matrix)
    {
        const int blockSize(128);
        const int blocks((capacity - 1) / blockSize + 1);
        for (int i = 0; i <= pd; i++)
        {
            blurKernel<T, pd, vd><<<blocks, blockSize, 0, mainStream>>>(matrix, i, *this);
            cudaErrorCheck(mainStream);
            std::swap(values, valuesCache);
        }
    }

    template <typename T, int pd, int vd>
    __global__ static void createLattice(const int n,
                                         const T *positions,
                                         const T *scaleFactor,
                                         MatrixEntry<T> *matrix,
                                         HashTableGPU<T, pd, vd> table)
    {

        const int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= n)
            return;

        T elevated[pd + 1];
        const T *position = positions + idx * pd;
        int rem0[pd + 1];
        int rank[pd + 1];

        // embed position vector into the hyperplane
        // first rotate position into the (pd+1)-dimensional hyperplane
        // sm contains the sum of 1..n of our feature vector
        T sm = 0;
        for (int i = pd; i > 0; i--)
        {
            T cf = position[i - 1] * scaleFactor[i - 1];
            elevated[i] = sm - i * cf;
            sm += cf;
        }
        elevated[0] = sm;

        // Find the closest 0-colored simplex through rounding
        // greedily search for the closest zero-colored lattice point
        short sum = 0;
        for (int i = 0; i <= pd; i++)
        {
            T v = elevated[i] * (1.0 / (pd + 1));
            T up = ceil(v) * (pd + 1);
            T down = floor(v) * (pd + 1);
            if (up - elevated[i] < elevated[i] - down)
            {
                rem0[i] = (short)up;
            }
            else
            {
                rem0[i] = (short)down;
            }
            sum += rem0[i];
        }
        sum /= pd + 1;

        // Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the features values)
        for (int i = 0; i <= pd; i++)
            rank[i] = 0;
        for (int i = 0; i < pd; i++)
        {
            double di = elevated[i] - rem0[i];
            for (int j = i + 1; j <= pd; j++)
                if (di < elevated[j] - rem0[j])
                    rank[i]++;
                else
                    rank[j]++;
        }

        // If the point doesn't lie on the plane (sum != 0) bring it back
        for (int i = 0; i <= pd; i++)
        {
            rank[i] += sum;
            if (rank[i] < 0)
            {
                rank[i] += pd + 1;
                rem0[i] += pd + 1;
            }
            else if (rank[i] > pd)
            {
                rank[i] -= pd + 1;
                rem0[i] -= pd + 1;
            }
        }

        T barycentric[pd + 2]{0};
        // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
        for (int i = 0; i <= pd; i++)
        {
            T delta = (elevated[i] - rem0[i]) * (1.0 / (pd + 1));
            barycentric[pd - rank[i]] += delta;
            barycentric[pd + 1 - rank[i]] -= delta;
        }
        // Wrap around
        barycentric[0] += 1.0 + barycentric[pd + 1];

        short key[pd];
        for (int remainder = 0; remainder <= pd; remainder++)
        {
            // Compute the location of the lattice point explicitly (all but
            // the last coordinate - it's redundant because they sum to zero)
            for (int i = 0; i < pd; i++)
            {
                key[i] = static_cast<short>(rem0[i] + remainder);
                if (rank[i] > pd - remainder)
                    key[i] -= (pd + 1);
            }

            MatrixEntry<T> r;
            unsigned int slot = static_cast<unsigned int>(idx * (pd + 1) + remainder);
            r.index = table.insert(key, slot);
            r.weight = barycentric[remainder];
            matrix[idx * (pd + 1) + remainder] = r;
        }
    }

    template <typename T, int pd, int vd>
    __global__ static void cleanHashTable(int n, HashTableGPU<T, pd, vd> table)
    {
        const int idx = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * (threadIdx.y + blockDim.y * blockIdx.y));
        // (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.x;

        if (idx >= n)
            return;

        // find my hash table entry
        int *e = table.entries + idx;

        // Check if I created my own key in the previous phase
        if (*e >= 0)
        {
            // Rehash my key and reset the pointer in order to merge with
            // any other pixel that created a different entry under the
            // same key. If the computation was serial this would never
            // happen, but sometimes race conditions can make the same key
            // be inserted twice. hashTableRetrieve always returns the
            // earlier, so it's no problem as long as we rehash now.
            *e = table.retrieve(table.keys + *e * pd);
        }
    }

    template <typename T, int pd, int vd>
    __global__ static void splatCache(const int n, const T *values, MatrixEntry<T> *matrix, HashTableGPU<T, pd, vd> table, bool isInit)
    {

        const int idx = threadIdx.x + blockIdx.x * blockDim.x;
        const int threadId = threadIdx.x;
        const int color = blockIdx.y;
        const bool outOfBounds = (idx >= n);

        __shared__ int sharedOffsets[BLOCK_SIZE];
        __shared__ T sharedValues[BLOCK_SIZE * vd];
        int myOffset = -1;
        T *myValue = sharedValues + threadId * vd;

        if (!outOfBounds)
        {

            T *value = const_cast<T *>(values + idx * (vd - 1));

            MatrixEntry<T> r = matrix[idx * (pd + 1) + color];

            if (isInit)
            {
                // convert the matrix entry from a pointer into the entries array to a pointer into the keys/values array
                matrix[idx * (pd + 1) + color].index = r.index = table.entries[r.index];
            }

            // record the offset into the keys/values array in shared space
            myOffset = sharedOffsets[threadId] = r.index * vd;

            for (int j = 0; j < vd - 1; j++)
            {
                myValue[j] = value[j] * r.weight;
            }
            myValue[vd - 1] = r.weight;
        }
        else
        {
            sharedOffsets[threadId] = -1;
        }

        __syncthreads();

        // am I the first thread in this block to care about this key?
        if (outOfBounds)
            return;

        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            if (i < threadId)
            {
                if (myOffset == sharedOffsets[i])
                {
                    // somebody else with higher priority cares about this key
                    return;
                }
            }
            else if (i > threadId)
            {
                if (myOffset == sharedOffsets[i])
                {
                    // someone else with lower priority cares about this key, accumulate it into mine
                    for (int j = 0; j < vd; j++)
                    {
                        sharedValues[threadId * vd + j] += sharedValues[i * vd + j];
                    }
                }
            }
        }

        // only the threads with something to write to main memory are still going
        T *val = table.values + myOffset;
        for (int j = 0; j < vd; j++)
        {
            atomicAdd(val + j, myValue[j]);
        }
    }

    template <typename T, int pd, int vd>
    __global__ static void slice(const int n, T *values, MatrixEntry<T> *matrix, HashTableGPU<T, pd, vd> table)
    {

        const int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= n)
            return;

        T value[vd - 1]{0};
        T weight = 0;

        for (int i = 0; i <= pd; i++)
        {
            MatrixEntry<T> r = matrix[idx * (pd + 1) + i];
            T *val = table.values + r.index * vd;
            for (int j = 0; j < vd - 1; j++)
            {
                value[j] += r.weight * val[j];
            }
            weight += r.weight * val[vd - 1];
        }

        weight = 1.0 / weight;
        for (int j = 0; j < vd - 1; j++)
            values[idx * (vd - 1) + j] = value[j] * weight;
    }

    template <typename T, int pd, int vd>
    class PermutohedralLattice
    {
    private:
        const cudaStream_t mainStream;

    public:
        int n; //number of pixels/voxels etc..
        T *scaleFactor;
        MatrixEntry<T> *matrix;
        HashTableGPU<T, pd, vd> hashTable;
        int filterTimes; // Lazy mark

        PermutohedralLattice(const cudaStream_t mainStream, int n)
            : mainStream(mainStream),
              n(n),
              scaleFactor(nullptr),
              matrix(nullptr),
              hashTable(HashTableGPU<T, pd, vd>(mainStream, n * (pd + 1)))
        {
            T hostScaleFactor[pd];
            T invStdDev = (pd + 1) * sqrt(2.0f / 3);
            for (int i = 0; i < pd; i++)
            {
                hostScaleFactor[i] = 1.0f / (sqrt((T)(i + 1) * (i + 2))) * invStdDev;
            }

            cudaMalloc((void **)&scaleFactor, sizeof(T) * pd);
            cudaErrorCheck(nullptr);
            cudaMemcpyAsync(scaleFactor, hostScaleFactor, sizeof(T) * pd, cudaMemcpyHostToDevice, mainStream);
            cudaErrorCheck(mainStream);

            cudaMalloc((void **)&matrix, sizeof(MatrixEntry<T>) * n * (pd + 1));
            cudaErrorCheck(nullptr);
        }

        ~PermutohedralLattice()
        {
            // tear aux variables.
            cudaFree(scaleFactor);
            cudaErrorCheck(nullptr);
            cudaFree(matrix);
            cudaErrorCheck(nullptr);
        }

        PermutohedralLattice(const PermutohedralLattice &o) = delete;

        // values and position must already be device pointers
        void prepare(const T *positions)
        {
            filterTimes = 0;
            hashTable.clear();
            dim3 blocks((n - 1) / BLOCK_SIZE + 1, 1, 1);
            dim3 blockSize(BLOCK_SIZE, 1, 1);
            dim3 cleanBlockSize(128, 1, 1);
            dim3 cleanBlocks((n - 1) / cleanBlockSize.x + 1, (2 * pd + 1) / cleanBlockSize.y + 1, 1);

            createLattice<T, pd, vd><<<blocks, blockSize, 0, mainStream>>>(n, positions, scaleFactor, matrix, hashTable);
            cudaErrorCheck(mainStream);

            cleanHashTable<T, pd, vd><<<cleanBlocks, cleanBlockSize, 0, mainStream>>>(2 * n * (pd + 1), hashTable);
            cudaErrorCheck(mainStream);
        }

        // values and position must already be device pointers
        void filter(T *output, const T *inputs)
        {
            dim3 blocks((n - 1) / BLOCK_SIZE + 1, pd + 1, 1);
            dim3 blockSize(BLOCK_SIZE, 1, 1);
            int cleanBlockSize = 128;
            dim3 cleanBlocks((n - 1) / cleanBlockSize + 1, 2 * (pd + 1), 1);

            cudaMemsetAsync((void *)(hashTable.values), 0, hashTable.capacity * vd * sizeof(T), mainStream);
            cudaErrorCheck(mainStream);
            cudaMemsetAsync((void *)(hashTable.valuesCache), 0, hashTable.capacity * vd * sizeof(T), mainStream);
            cudaErrorCheck(mainStream);

            splatCache<T, pd, vd><<<blocks, blockSize, 0, mainStream>>>(n, inputs, matrix, hashTable, filterTimes == 0);
            cudaErrorCheck(mainStream);

            hashTable.blur(matrix);

            blockSize.y = 1;
            slice<T, pd, vd><<<blocks, blockSize, 0, mainStream>>>(n, output, matrix, hashTable);
            cudaErrorCheck(mainStream);
            ++filterTimes;
        }
    };

} // namespace crf