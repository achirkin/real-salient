/*
Current version: Artem Chirkin

Based on: Jiahui Huang (https://github.com/heiwang1997/DenseCRF)

*/

#pragma once

#include <vector>
#include "crf_permutohedral.cuh"
#include "util.hpp"

namespace salient
{
    namespace crf
    {

        // Weight applying kernels for potts potential.
        template <int M>
        __global__ static void pottsWeight(float* out, const float* in, const int n, const float pw)
        {
            const int ni = threadIdx.x + blockIdx.x * blockDim.x;
            const int vi = blockIdx.y;
            if (ni >= n)
                return;
            out[ni * M + vi] += pw * in[ni * M + vi];
        }

        // Initializing kernels for potts potential.
        template <int F>
        __global__ static void assembleImageFeature(int w, int h, const float* features, float posdev, float featuredev, float* out)
        {
            const int wi = threadIdx.x + blockIdx.x * blockDim.x;
            const int hi = threadIdx.y + blockIdx.y * blockDim.y;
            if (wi >= w || hi >= h)
                return;

            const int idx = hi * w + wi;
            const int posDims = posdev == 0 ? 0 : 2;
            if (posDims != 0)
            {
                out[idx * F + 0] = (float)wi / posdev;
                out[idx * F + 1] = (float)hi / posdev;
            }
#pragma unroll
            for (int i = posDims; i < F; ++i)
            {
                out[idx * F + i] = (float)features[idx * (F - posDims) + (i - posDims)] / featuredev;
            }
        }

        class IPairwisePotential
        {
        public:
            virtual ~IPairwisePotential() {}
            virtual void loadImage(const float weight, const float div_position, const float div_feature = 0.0f, const float* img_features = nullptr) = 0;
            virtual void apply(float* out_values, const float* in_values, float* tmp) = 0;
        };

        template <int M, int F>
        class PairwisePotential : public IPairwisePotential
        {
        private:
            const cudaStream_t mainStream;

            /** An intermediate buffer to copy the spatial and image features. */
            float* features;
            /**
             * Actual inference happence in PermutohedralLattice class.
             * Number of parameter dimensions is the same as number of features.
             * Number of output dimensions is the number of classes plus one;
             *   that is for the purpose of normalization.
             */
            PermutohedralLattice<float, F, M + 1>* lattice;

        protected:
            const int N;
            float weight;
            const int w;
            const int h;

        public:
            PairwisePotential(const cudaStream_t mainStream, int w, int h)
                : mainStream(mainStream),
                N(w* h), w(w), h(h),
                features(nullptr)
            {
                lattice = new PermutohedralLattice<float, F, M + 1>(mainStream, N);
                cudaMalloc((void**)&features, sizeof(float) * F * N);
                cudaErrorCheck(nullptr);
            }

            ~PairwisePotential() override
            {
                delete lattice;
                cudaFree(features);
            }

            PairwisePotential(const PairwisePotential& o) = delete;

            void loadImage(const float weight, const float div_position, const float div_feature = 0.0f, const float* img_features = nullptr) override
            {
                dim3 blockSize(32, 32, 1);
                dim3 workSize(w, h, 1);
                dim3 blocks = distribute(workSize, blockSize);
                this->weight = weight;
                assembleImageFeature<F> << <blocks, blockSize, 0, mainStream >> > (w, h, img_features, div_position, div_feature, features);
                cudaErrorCheck(mainStream);
                lattice->prepare(features); // const float *features,
            }

            void apply(float* out_values, const float* in_values, float* tmp) override
            {
                lattice->filter(tmp, in_values);
                dim3 blockSize(BLOCK_SIZE, 1, 1);
                dim3 workSize(N, M, 1);
                dim3 blocks = distribute(workSize, blockSize);
                pottsWeight<M> << <blocks, blockSize, 0, mainStream >> > (out_values, tmp, N, weight);
                cudaErrorCheck(mainStream);
            }
        };

        // GPU CUDA Implementation
        template <int M>
        class DenseCRF
        {

        private:
            const cudaStream_t mainStream;

            // Pre-allocated host/device memory
            float* current, * next, * tmp;

            // Negative unary potential (i.e. log-likelihood) for all variables and labels (memory order is [x0l0 x0l1 x0l2 .. x1l0 x1l1 ...])
            const float* unary;

            // Store all pairwise potentials
            std::vector<IPairwisePotential*> pairwise;

            void expAndNormalize(float* out, const float* in, float scale = 1.0, float relax = 1.0);
            void stepInit();

        public:
            // Number of variables and labels
            const int N;

            // Create a dense CRF model of size N with M labels
            explicit DenseCRF(const cudaStream_t mainStream, int N, const float* lhood)
                : mainStream(mainStream), N(N), unary(lhood), current(nullptr), next(nullptr), tmp(nullptr)
            {
                cudaMalloc((void**)&current, sizeof(float) * N * M);
                cudaErrorCheck(nullptr);
                cudaMalloc((void**)&next, sizeof(float) * N * M);
                cudaErrorCheck(nullptr);
                cudaMalloc((void**)&tmp, sizeof(float) * N * M);
                cudaErrorCheck(nullptr);
            }

            ~DenseCRF()
            {
                cudaFree(current);
                cudaErrorCheck(nullptr);
                cudaFree(next);
                cudaErrorCheck(nullptr);
                cudaFree(tmp);
                cudaErrorCheck(nullptr);
                for (auto* pPairwise : pairwise)
                {
                    delete pPairwise;
                }
                pairwise.clear();
            }

            DenseCRF(DenseCRF& o) = delete;

            // Add your own favorite pairwise potential (ownwership will be transfered to this class)
            void addPairwiseEnergy(IPairwisePotential* potential) { pairwise.push_back(potential); }

            // Run inference and return the probabilities
            // All returned values are managed by class
            virtual void inference(int n_iterations, float relax = 1.0)
            {
                startInference();
                for (int it = 0; it < n_iterations; ++it)
                {
                    stepInference(relax);
                }
            }
            float* logLikelihoodPtr() const { return current; }

            // Step by step inference
            void startInference()
            {
                expAndNormalize(current, unary, 1);
            }

            void stepInference(float relax = 1.0)
            {
                // Set the unary potential
                stepInit();
                // Add up all pairwise potentials
                for (unsigned int i = 0; i < pairwise.size(); i++)
                {
                    pairwise[i]->apply(next, current, tmp);
                }
                // Exponentiate and normalize
                expAndNormalize(current, next, 1.0, relax);
            }
        };

        template <int M>
        __global__ static void expNormKernel(int N, float* out, const float* in, float scale, float relax)
        {
            const int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx >= N)
                return;
            const float* b = in + idx * M;
            // Find the max and subtract it so that the exp doesn't explode
            float mx = scale * b[0];
            for (int j = 1; j < M; ++j)
            {
                if (mx < scale * b[j])
                {
                    mx = scale * b[j];
                }
            }
            float tt = 0.0;
            float V[M]{ 0 };
            for (int j = 0; j < M; ++j)
            {
                V[j] = __expf(scale * b[j] - mx);
                tt += V[j];
            }
            // Make it a probability
            for (int j = 0; j < M; ++j)
            {
                V[j] /= tt;
            }
            float* a = out + idx * M;
            for (int j = 0; j < M; ++j)
            {
                a[j] = (1 - relax) * a[j] + relax * V[j];
            }
        }

        template <int M>
        void DenseCRF<M>::expAndNormalize(float* out, const float* in, float scale /* = 1.0 */, float relax /* = 1.0 */)
        {
            expNormKernel<M> << <distribute(N, BLOCK_SIZE), BLOCK_SIZE, 0, mainStream >> > (N, out, in, scale, relax);
            cudaErrorCheck(mainStream);
        }

        template <int M>
        void DenseCRF<M>::stepInit()
        {
            cudaMemcpyAsync(next, unary, sizeof(float) * N * M, cudaMemcpyDeviceToDevice, mainStream);
            cudaErrorCheck(mainStream);
        }

    } // namespace crf
}  // namespace salient