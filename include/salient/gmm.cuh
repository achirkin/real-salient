/*
Author: Artem Chirkin

*/

#pragma once

#include "util.hpp"

#ifndef EPS
#define EPS 0.00001f
#endif

namespace salient
{
    namespace gmm
    {

        /**
         * Train M Gaussian Mixture Models containing K Gaussian components each.
         *
         * @param C Number of feature channels in the data.
         * @param K Number of Gaussians in each GMM.
         * @param M Number of GMMs.
         */
        template <int C, int K, int M>
        class GaussianMixtures
        {
        private:
            const cudaStream_t mainStream;

            // size: N * C
            float* features;
            // size: N
            int8_t* labels;
            // size: N * K
            // w_ij - probability of pixel i (in M) being in class j (in K) under GMM model labels[i]
            float* probs_weights;
            // size: M * K
            // phi_mj - weight of a component j (in K) in GMM m (in M) (NB: not normalized by the number of points)
            float* mixture_weights;
            float* mixture_weights_longterm;
            // size: M * K * C
            // mu_mj - estimated mean of a component j (in K) in GMM m (in M).
            float* means;
            float* means_longterm;
            // size: M * K * C * C
            // sigma_mj - estimated covariance of a component j (in K) in GMM m (in M).
            float* covs;
            float* covs_longterm;
            // size: M * K * C * C
            float* covs_inv;
            // size: M * K
            float* covs_det;
            // size: N * M
            // llhoods[i*M + m] is the maximum log-likelihood of feature[i] being in one of the components of GMM m;
            // i.e. the log-likelihood of being in class m.
            float* llhoods;

            /**
             * Initialize the coefficients of all GMMs.
             *
             * NB: This is a one-time operation to be done after the features and labels are set at lest once.
             */
            void initModels();

            // Whether the models are already initialized.
            bool initModelsDone = false;

            void expectationStep();

            void maximizationStep(bool doMovingAverage);

            const int blocksForN;
            const int blockSizeForN;
            // temporary array for aggregating estimated values.
            // to fit many covs matrices, the maximum required size is: blockSizeForN * M * K * C * C
            float* aggr_temp;

            float cpu_mixture_weights_temp[M * K];

        public:
            /** Number of data points. */
            const int N;

            /** Exponential moving average constant.  */
            float alpha;

            /** A multiplier for imputed labels (label for class 'm' imputed is 'm - 128'). */
            float imputed_weight;

            /**
             * Initialize the instance and allocate GPU arrays.
             * The input data is expected to contain (N*C) points of type float, where
             * C is a compile-time constant indicating the number of dimensions (channels) of a single point and
             * N is a runtime constant indicating the number of points.
             *
             * @param N number of data points.
             * @param alpha constant for smoothing the GMM parameters over time (frames).
             * @param imputed_weight how important the imputed label values compared to real labels.
             */
            explicit GaussianMixtures(const cudaStream_t mainStream, int N, float alpha = 0.1f, float imputed_weight = 0.3f)
                : mainStream(mainStream),
                N(N),
                alpha(alpha),
                imputed_weight(imputed_weight),
                blocksForN((N - 1) / BLOCK_SIZE + 1),
                blockSizeForN(BLOCK_SIZE)
            {
                cudaMalloc((void**)&features, sizeof(float) * N * C);
                cudaMalloc((void**)&labels, sizeof(int8_t) * N);
                cudaMalloc((void**)&probs_weights, sizeof(float) * N * K);
                cudaMalloc((void**)&mixture_weights, sizeof(float) * M * K);
                cudaMalloc((void**)&mixture_weights_longterm, sizeof(float) * M * K);
                cudaMalloc((void**)&means, sizeof(float) * M * K * C);
                cudaMalloc((void**)&means_longterm, sizeof(float) * M * K * C);
                cudaMalloc((void**)&covs, sizeof(float) * M * K * C * C);
                cudaMalloc((void**)&covs_longterm, sizeof(float) * M * K * C * C);
                cudaMalloc((void**)&covs_inv, sizeof(float) * M * K * C * C);
                cudaMalloc((void**)&covs_det, sizeof(float) * M * K);
                cudaMalloc((void**)&llhoods, sizeof(float) * N * M);
                cudaMalloc((void**)&aggr_temp, sizeof(float) * blockSizeForN * M * K * C * C);
                cudaErrorCheck(nullptr);
            }

            ~GaussianMixtures()
            {
                cudaFree(features);
                cudaFree(labels);
                cudaFree(probs_weights);
                cudaFree(mixture_weights);
                cudaFree(mixture_weights_longterm);
                cudaFree(means);
                cudaFree(means_longterm);
                cudaFree(covs);
                cudaFree(covs_longterm);
                cudaFree(covs_inv);
                cudaFree(covs_det);
                cudaFree(llhoods);
                cudaFree(aggr_temp);
                cudaErrorCheck(nullptr);
            }

            GaussianMixtures(const GaussianMixtures& o) = delete;

            /** Perform a single EM estimation step. */
            void iterationStep(bool doMovingAverage)
            {
                auto doMAReally = doMovingAverage && initModelsDone; // don't do MA if the longterm state is not initialized yet
                if (!initModelsDone)
                {
                    initModels();
                    initModelsDone = true;
                }
                expectationStep();
                maximizationStep(doMAReally);
            }

            /**
             * Perform a few EM estimation steps.
             *
             * @param steps number of iterations to perform.
             */
            void iterate(int steps)
            {
                for (int i = steps; i > 0; i--)
                    iterationStep(i == 1);
            }

            /** Calculate likelihoods for the current dataset and put them into the llhoods array. */
            void infer();

            /** Get the pointer to the feature array */
            inline float* featuresPtr()
            {
                return features;
            }

            /** Get the pointer to the labels array.
             *
             *  Ground-truth labels must be within 0 (included) .. M (excluded).
             *
             *  Imputed labels are optional, if present must be within -128 (included) .. (M - 128) (excluded).
             *
             *  The rest of the values mean unused data (but -1 is the preferable value for the unused data).
             *
             */
            inline int8_t* labelsPtr()
            {
                return labels;
            }

            /** Get the pointer to the labels array */
            inline float* logLikelihoodPtr()
            {
                return llhoods;
            }
        };

        template <int C, int K, int M>
        __global__ void expectationKernel(
            int N,
            const float* features, const int8_t* labels,
            const float* means, const float* covs_inv, const float* mixture_weights, float* out_probs_weights)
        {
            const int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= N)
                return;

            const int m = labels[i] & 0x7F; // use both GT and imputed data.

            if (m >= M || m < 0)
                return;

            float x[C];  // data point
            float ws[K]; // collecting output here
            float w;     // temporary value for the exponent
            const int KC = K * C;
            const int KCC = KC * C;
            const int CC = C * C;
            const float penalty(0.0001f * N / (K * M));

            float totalOverK = 0;
#pragma unroll
            for (int k = 0; k < K; k++)
            { // go over all Gaussian components of a single selected GMM m.
#pragma unroll
                for (int c = 0; c < C; c++)
                    x[c] = features[i * C + c] - means[m * KC + k * C + c];

                w = 0.0f;
#pragma unroll
                for (int ci = 0; ci < C; ci++)
#pragma unroll
                    for (int cj = 0; cj < C; cj++)
                        w += x[ci] * x[cj] * covs_inv[m * KCC + k * CC + ci * C + cj];

                // the second component of max protects against zeros-nans.
                w = penalty + __expf(-0.5f * max(0.0f, w)) * mixture_weights[m * K + k];

                totalOverK += w;
                ws[k] = w;
            }

            w = 1.0f / totalOverK;
#pragma unroll
            for (int k = 0; k < K; k++)
                out_probs_weights[i * K + k] = ws[k] * w;
        }

        template <int C, int K, int M>
        void GaussianMixtures<C, K, M>::expectationStep()
        {
            expectationKernel<C, K, M><<<blocksForN, blockSizeForN, 0, mainStream>>>(N, features, labels, means, covs_inv, mixture_weights, probs_weights);
            cudaErrorCheck(mainStream);
        }

        /**
         * The last step of any aggregation in this module is to sum up exactly blockDim.x float vectors,
         * size (S*T) each.
         * This step is called with one block only.
         *
         * @param S is how many times to repeat the aggregation, so that the resulting array has length (S*T).
         * @param T (times the number of blocks) is the number of elements in the shared data.
         */
        template <int S, int T>
        __global__ void aggregateN(const float* aggr_temp, float* out)
        {
            extern __shared__ float sdata[];
            int tid = threadIdx.x;
            int stride = blockDim.x;
            float* ldata = sdata + tid * T;
            const float* aggr_block = aggr_temp + tid * S * T;

#pragma unroll
            for (int s = 0; s < S; s++)
            {
#pragma unroll
                for (int t = 0; t < T; t++)
                    ldata[t] = aggr_block[s * T + t];

                __syncthreads();
                for (unsigned int i = blockDim.x >> 1; i > 0; i >>= 1)
                {
                    if (tid < i)
#pragma unroll
                        for (int t = 0; t < T; t++)
                            ldata[t] += ldata[i * T + t];
                    __syncthreads();
                }

                for (int t = tid; t < T; t += stride)
                    out[s * T + t] = sdata[t];
                __syncthreads();
            }
        }

        template <int T, int K, int M, class GetShared, class WriteAggregated>
        __device__ void aggregateGeneric(int N, const int8_t* labels, GetShared getShared, WriteAggregated writeAggregated, const float imputed_weight)
        {
            extern __shared__ float sdata[];

            // init local storage
            int tid = threadIdx.x;
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            float* ldata = sdata + tid * T;
            int m;
            float weight;

#pragma unroll
            for (int k = 0; k < K; k++)
            {
#pragma unroll
                for (int t = 0; t < T; t++)
                    ldata[t] = 0;

                // sequential part: aggregate a few items in the global memory into the shared memory
                for (int i = index; i < N; i += stride)
                {
                    m = labels[i];
                    weight = m < 0 ? imputed_weight : 1.0f;
                    m &= 0x7F;
                    if (m >= M)
                        continue;

                    getShared(k, i, m, ldata, weight);
                }
                __syncthreads();

                // parallel reduction
                for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
                {
                    if (tid < s)
                    {
#pragma unroll
                        for (int t = 0; t < T; t++)
                            ldata[t] += ldata[s * T + t];
                    }
                    __syncthreads();
                }

                // copy result to the aggr_temp
                writeAggregated(k, blockIdx.x, blockDim.x, tid, sdata);
                __syncthreads();
            }
        }

        template <int K, int M>
        __global__ void maximization_aggregateWeights(
            int N,
            const int8_t* labels,
            const float* probs_weights /* N * K */,
            float* aggr_temp /* used memory, up to: blockSizeForN * M * K  */,
            const float imputed_weight)
        {
            const int MK(M * K);
            aggregateGeneric<M, K, M>(
                N, labels,
                [&probs_weights](int k, int i, int m, float* ldata, const float weight) {
                    ldata[m] += probs_weights[i * K + k] * weight;
                },
                [&aggr_temp, MK](int k, int blockI, int blockD, int threadI, float* sdata) {
                    const int aggr_off(blockI * MK);
                    for (int m = threadI; m < M; m += blockD)
                        aggr_temp[aggr_off + m * K + k] = sdata[m];
                },
                    imputed_weight);
        }

        template <int C, int K, int M>
        __global__ void maximization_aggregateMeans(
            int N,
            const float* features /* N * C */,
            const int8_t* labels /* N */,
            const float* probs_weights /* N * K */,
            const float* mixture_weights /* M * K */,
            float* aggr_temp /* used memory, up to: blockSizeForN * M * K * C  */,
            const float imputed_weight)
        {
            aggregateGeneric<M* C, K, M>(
                N, labels,
                [&probs_weights, &features](int k, int i, int m, float* ldata, const float weight) {
#pragma unroll
                    for (int c = 0; c < C; c++)
                        ldata[m * C + c] += probs_weights[i * K + k] * features[i * C + c] * weight;
                },
                [&aggr_temp, &mixture_weights](int k, int blockI, int blockD, int threadI, float* sdata) {
                    const int MK(M * K);
                    const int aggr_off(blockI * MK * C);
                    float mw;
                    for (int m = threadI; m < M; m += blockD)
                    {
                        mw = 1.0f / max(EPS, mixture_weights[m * K + k]);
#pragma unroll
                        for (int c = 0; c < C; c++)
                            aggr_temp[aggr_off + (m * K + k) * C + c] = sdata[m * C + c] * mw;
                    }
                },
                    imputed_weight);
        }

        template <int C, int K, int M>
        __global__ void maximization_aggregateCovs(
            int N,
            const float* features /* N * C */,
            const int8_t* labels /* N */,
            const float* probs_weights /* N * K */,
            const float* mixture_weights /* M * K */,
            const float* means /* M * K * C */,
            float* aggr_temp /* used memory, up to: blockSizeForN * M * K * C * C  */,
            const float imputed_weight)
        {
            float ex[C]; // local buffer for [x - mean(x)]
            aggregateGeneric<M* C* C, K, M>(
                N, labels,
                [&probs_weights, &features, &means, &ex](int k, int i, int m, float* ldata, const float weight) {
#pragma unroll
                    for (int c = 0; c < C; c++)
                        ex[c] = features[i * C + c] - means[(m * K + k) * C + c];
#pragma unroll
                    for (int ci = 0; ci < C; ci++)
#pragma unroll
                        for (int cj = 0; cj < C; cj++)
                            ldata[(m * C + ci) * C + cj] += probs_weights[i * K + k] * ex[ci] * ex[cj] * weight;
                },
                [&aggr_temp, &mixture_weights](int k, int blockI, int blockD, int threadI, float* sdata) {
                    const int MK(M * K);
                    const int CC(C * C);
                    const int aggr_off(blockI * MK * C * C);
                    float mw;
                    for (int m = threadI; m < M; m += blockD)
                    {
                        mw = 1.0f / max(EPS, mixture_weights[m * K + k]);
#pragma unroll
                        for (int c = 0; c < CC; c++)
                            aggr_temp[aggr_off + (m * K + k) * CC + c] = sdata[m * CC + c] * mw;
                    }
                },
                    imputed_weight);
        }

        template <int C>
        __global__ void maximization_doMovingAverage(
            int MK, float alpha,
            float* mixture_weights, float* mixture_weights_longterm /* MK */,
            float* means, float* means_longterm /* MK * C */,
            float* covs, float* covs_longterm /* MK * C * C */)
        {
            const int CC(C * C);
            const int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= MK)
                return;

            float mw1 = mixture_weights[i];
            float mw0 = mixture_weights_longterm[i];

            float w1 = alpha * mw1 / max(1.0f, mw0 + mw1);

            // trying different strategies to choose between the old and new values.
            // the multipliers should generally be greater than one, but the results look best
            // if equal to one.
            if (mw0 > mw1 * 3)
                w1 = 0;
            else if (mw1 > mw0 * 10)
                w1 = 1;

            float w0 = 1 - w1;

            float x = mw1 * w1 + mw0 * w0;
            int ix = i;
            mixture_weights[ix] = x;
            mixture_weights_longterm[ix] = x;

#pragma unfold
            for (int c = 0; c < C; c++)
            {
                ix = i * C + c;
                x = means[ix] * w1 + means_longterm[ix] * w0;
                means[ix] = x;
                means_longterm[ix] = x;
            }

#pragma unfold
            for (int c = 0; c < CC; c++)
            {
                ix = i * CC + c;
                x = covs[ix] * w1 + covs_longterm[ix] * w0;
                covs[ix] = x;
                covs_longterm[ix] = x;
            }
        }

        /**
         * Calculate inverse and determinant of the covariances matrices using Cholesky decomposition.
         *
         * https://en.wikipedia.org/wiki/Cholesky_decomposition#LDL_decomposition_2
         *
         * Distribution of the work: C threads per one matrix.
         * Required shared memory: blockDim.x * sizeof(float)
         *
         * @param C dimension of the covariance (i.e. number of channels in the data)
         * @param N number of covariance matrices.
         */
        template <int C, int N>
        __global__ void covariance_solve(
            float* in_covs /* N * C * C */,
            float* out_covs_inv /* N * C * C */,
            float* out_covs_det /* N */
        )
        {
            extern __shared__ float sdata[];

            const int matricesPerBlock(blockDim.x / C);

            // dimension (channel) worked by this thread
            const int c(threadIdx.x % C);
            // matrix index worked by this thread
            const int n(threadIdx.x / C + blockIdx.x * matricesPerBlock);

            const bool outOfBounds = (threadIdx.x >= matricesPerBlock * C) || (n >= N);

            float* cov = in_covs + C * C * n;
            float* cov_inv = out_covs_inv + C * C * n;
            // diagonal D of LDL' decomposition is stored in a shared memory for faster access
            float* matrix_D = sdata + threadIdx.x - c;

            // (1) calculate the LDL' decomposition.
            float detj = 1;
            float Dj, t;
            if (!outOfBounds)
                cov_inv[c * C + c] = 1; // by definition of LDL'
#pragma unfold
            for (int j = 0; j < C; j++)
            {
                if (!outOfBounds)
                {
                    // Extra constant here is to dumb-increase covariance diagonal values for better convergence
                    Dj = cov[j * (C + 1)] + 0.001f;
#pragma unfold
                    for (int k = 0; k < j; k++)
                    {
                        t = cov_inv[j * C + k];
                        Dj -= t * t * matrix_D[k];
                    }
                    Dj = max(EPS, Dj);
                    if (c == 0)
                    {
                        matrix_D[j] = Dj;
                        detj *= Dj;
                    }
                    else if (c > j)
                    {
                        t = cov[c * C + j];
#pragma unfold
                        for (int k = 0; k < j; k++)
                            t -= cov_inv[c * C + k] * cov_inv[j * C + k] * matrix_D[k];
                        cov_inv[c * C + j] = t / Dj;
                    }
                }
                __syncthreads();
            }

            if (!outOfBounds)
            {
                // write matrix determinant
                if (c == 0)
                {
                    out_covs_det[n] = detj;
                }

                // (2) inverse L, simple forward substitution
                // Diag elements ==1 by definition of LDL
                // The result is placed into the upper-triangular part of cov_inv.
#pragma unfold
                for (int i = c + 1; i < C; i++)
                {
                    t = 0;
#pragma unfold
                    for (int j = c; j < i; j++)
                        t -= cov_inv[i * C + j] * cov_inv[c * C + j];
                    cov_inv[c * C + i] = t;
                }
            }
            __syncthreads();

            if (!outOfBounds)
            {
                // (3) Compute the precision matrix via inverse L (L^-T D^-1 L^-1)
#pragma unfold
                for (int j = 0; j <= c; j++)
                {
                    t = (c == j ? 1.0f : cov_inv[j * C + c]) / matrix_D[c];
#pragma unfold
                    for (int i = c + 1; i < C; i++)
                        t += cov_inv[c * C + i] * cov_inv[j * C + i] / matrix_D[i];
                    cov_inv[c * C + j] = t;
                }
            }
            __syncthreads();
            if (!outOfBounds)
            {
#pragma unfold
                for (int j = c + 1; j < C; j++)
                    cov_inv[c * C + j] = cov_inv[j * C + c];
            }
        }

        template <int C, int K, int M>
        void GaussianMixtures<C, K, M>::maximizationStep(bool doMovingAverage)
        {
            const int CC(C * C);
            const int MK(M * K);
            // TODO: dynamically balance the number of blocks and the block size to keep the shared memory within limits.
            const int blocks(min(N, blockSizeForN)); // make sure there are less than blockSizeForN blocks
            int shared_size;

            // calculate mixture_weights
            shared_size = sizeof(float) * blockSizeForN * M;
            maximization_aggregateWeights<K, M><<<blocks, blockSizeForN, shared_size, mainStream>>>(N, labels, probs_weights, aggr_temp, imputed_weight);
            shared_size = sizeof(float) * blocks * M;
            aggregateN<K, M><<<1, blocks, shared_size, mainStream>>>(aggr_temp, mixture_weights);

            // calculate means
            shared_size = sizeof(float) * blockSizeForN * M * C;
            maximization_aggregateMeans<C, K, M><<<blocks, blockSizeForN, shared_size, mainStream>>>(N, features, labels, probs_weights, mixture_weights, aggr_temp, imputed_weight);
            shared_size = sizeof(float) * blocks * M * C;
            aggregateN<K, M* C><<<1, blocks, shared_size, mainStream>>>(aggr_temp, means);

            // fix bad components by looking for too small or repeating weights
            cudaMemcpyAsync(cpu_mixture_weights_temp, mixture_weights, sizeof(float) * MK, cudaMemcpyDeviceToHost, mainStream);
            cudaErrorCheck(mainStream);
            cudaStreamSynchronize(mainStream); // need to wait for this memory copy to finish.
            const float weightEps(max(1.0f, 0.001f * N / MK));
            for (int m = 0; m < M; m++)
            {
                for (int k = 0; k < K; k++)
                {
                    auto mk = m * K + k;
                    auto w = cpu_mixture_weights_temp[mk];
                    auto really_bad = isnan(w);
                    auto faulty = really_bad || w < weightEps;
                    auto rand_feature = rand() % N;
                    for (int i = m * K; i < mk && !faulty; i++)
                        faulty = abs(cpu_mixture_weights_temp[i] - w) < weightEps;
                    if (faulty)
                    {
                        // std::cout << "Faulty! " << m << ":" << k << std::endl;
                        if (really_bad)
                            cpu_mixture_weights_temp[mk] = weightEps;
                        // else
                        //     cpu_mixture_weights_temp[mk] /= 2;
                        cudaMemcpyAsync(means + mk * C, features + rand_feature * C, sizeof(float) * C, cudaMemcpyDeviceToDevice, mainStream);
                        cudaErrorCheck(mainStream);
                    }
                }
            }
            cudaMemcpyAsync(mixture_weights, cpu_mixture_weights_temp, sizeof(float) * MK, cudaMemcpyHostToDevice, mainStream);
            cudaErrorCheck(mainStream);

            // calculate covariance matrices
            shared_size = sizeof(float) * blockSizeForN * M * CC;
            maximization_aggregateCovs<C, K, M><<<blocks, blockSizeForN, shared_size, mainStream>>>(N, features, labels, probs_weights, mixture_weights, means, aggr_temp, imputed_weight);
            shared_size = sizeof(float) * blocks * M * CC;
            aggregateN<K, M* CC><<<1, blocks, shared_size, mainStream>>>(aggr_temp, covs);
            cudaErrorCheck(mainStream);

            // update current and longterm model parameters if necessary.
            if (doMovingAverage)
            {
                int MK2 = 1;
                while (MK2 < MK)
                    MK2 <<= 1;
                maximization_doMovingAverage<C><<<1, MK2, 0, mainStream>>>(MK, alpha,
                    mixture_weights, mixture_weights_longterm,
                    means, means_longterm,
                    covs, covs_longterm);
                cudaErrorCheck(mainStream);
            }

            // calculate inverse covariance matrices and determinants.
            const int matricesPerBlock(BLOCK_SIZE / C);
            covariance_solve<C, MK><<<distribute(MK, matricesPerBlock), BLOCK_SIZE, BLOCK_SIZE * sizeof(float), mainStream>>>(covs, covs_inv, covs_det);
            cudaErrorCheck(mainStream);
        }

        template <int C, int M>
        __global__ void initModels_minmax1(
            int N,
            const float* features /* N * C */,
            const int8_t* labels,
            float* aggr_temp /* used memory, up to: blockSizeForN * M * C * 2  */)
        {
            extern __shared__ float sdata[];

            // init local storage
            int tid = threadIdx.x;
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            int m;

#pragma unroll
            for (int m = 0; m < M; m++)
            {
#pragma unroll
                for (int c = 0; c < C; c++)
                    sdata[2 * (tid * M + m) * C + c] = 9999999999.9f;

#pragma unroll
                for (int c = 0; c < C; c++)
                    sdata[(2 * (tid * M + m) + 1) * C + c] = -9999999999.9f;
            }

            // sequential part: aggregate a few items in the global memory into the shared memory
            for (int i = index; i < N; i += stride)
            {
                m = labels[i];
                if (m >= M || m < 0)
                    continue;

#pragma unroll
                for (int c = 0; c < C; c++)
                    sdata[2 * (tid * M + m) * C + c] = min(sdata[2 * (tid * M + m) * C + c], features[i * C + c]);

#pragma unroll
                for (int c = 0; c < C; c++)
                    sdata[(2 * (tid * M + m) + 1) * C + c] = max(sdata[(2 * (tid * M + m) + 1) * C + c], features[i * C + c]);
            }
            __syncthreads();

            // parallel reduction
            for (unsigned int t = blockDim.x >> 1; t > 0; t >>= 1)
            {
                if (tid < t)
                {
#pragma unroll
                    for (int m = 0; m < M; m++)
                    {
#pragma unroll
                        for (int c = 0; c < C; c++)
                            sdata[2 * (tid * M + m) * C + c] = min(sdata[2 * (tid * M + m) * C + c], sdata[2 * ((tid + t) * M + m) * C + c]);

#pragma unroll
                        for (int c = 0; c < C; c++)
                            sdata[(2 * (tid * M + m) + 1) * C + c] = max(sdata[(2 * (tid * M + m) + 1) * C + c], sdata[(2 * ((tid + t) * M + m) + 1) * C + c]);
                    }
                }
                __syncthreads();
            }

            for (int i = tid; i < C * M * 2; i += blockDim.x)
                aggr_temp[blockIdx.x * C * M * 2 + i] = sdata[i];
        }

        template <int C, int M>
        __global__ void initModels_minmax2(float* aggr_temp)
        {
            extern __shared__ float sdata[];

            // init local storage
            int tid = threadIdx.x;

#pragma unroll
            for (int i = 0; i < M * C * 2; i++)
                sdata[2 * tid * C * M + i] = aggr_temp[2 * tid * C * M + i];
            __syncthreads();

            // parallel reduction
            for (unsigned int t = blockDim.x >> 1; t > 0; t >>= 1)
            {
                if (tid < t)
                {
#pragma unroll
                    for (int m = 0; m < M; m++)
                    {
#pragma unroll
                        for (int c = 0; c < C; c++)
                            sdata[2 * (tid * M + m) * C + c] = min(sdata[2 * (tid * M + m) * C + c], sdata[2 * ((tid + t) * M + m) * C + c]);

#pragma unroll
                        for (int c = 0; c < C; c++)
                            sdata[(2 * (tid * M + m) + 1) * C + c] = max(sdata[(2 * (tid * M + m) + 1) * C + c], sdata[(2 * ((tid + t) * M + m) + 1) * C + c]);
                    }
                }
                __syncthreads();
            }

            for (int i = tid; i < C * M * 2; i += blockDim.x)
                aggr_temp[i] = sdata[i];
        }

        template <int C, int K, int M>
        __global__ void initModelsKernel(
            float* features_minmax /* M * C * 2 */,
            float* mixture_weights /* M * K */,
            float* means /* M * K * C */,
            float* covs_inv /* N * K * C * C */,
            float* covs_det /* N * K */)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            float fmin[C];
            float fmax[C];
            float cov_inv[C];
            float cov_det;
            float temp;
            for (int i = index; i < M; i += stride)
            {
                cov_det = 1;
#pragma unroll
                for (int c = 0; c < C; c++)
                {
                    fmin[c] = features_minmax[2 * i * C + c];
                    fmax[c] = features_minmax[(2 * i + 1) * C + c];
                    temp = (fmax[c] - fmin[c]);
                    temp = max(EPS, temp * temp / 3); // assume minmax span is 3 sigma.
                    cov_inv[c] = 1.0f / temp;
                    cov_det *= cov_det;
                }

#pragma unroll
                for (int k = 0; k < K; k++)
                {
                    mixture_weights[i * K + k] = 1;
                    covs_det[i * K + k] = cov_det;

#pragma unroll
                    for (int ci = 0; ci < C; ci++)
                    {
                        means[(i * K + k) * C + ci] = 0.5f * (fmax[ci] - fmin[ci]) * cospif((float)k / (float)(K - 1) + 1.5 * (float)ci / (float)C) + 0.5 * (fmax[ci] + fmin[ci]);
#pragma unroll
                        for (int cj = 0; cj < C; cj++)
                            covs_inv[((i * K + k) * C + ci) * C + cj] = ci == cj ? cov_inv[ci] : 0;
                    }
                }
            }
        }

        template <int C, int K, int M>
        void GaussianMixtures<C, K, M>::initModels()
        {

            const int blocks(min(N, blockSizeForN)); // make sure there are less than blockSizeForN blocks
            int shared_size;
            shared_size = sizeof(float) * blockSizeForN * M * C * 2;
            initModels_minmax1<C, M><<<blocks, blockSizeForN, shared_size, mainStream>>>(N, features, labels, aggr_temp);
            cudaErrorCheck(mainStream);
            shared_size = sizeof(float) * blocks * M * C * 2;
            initModels_minmax2<C, M><<<1, blocks, shared_size, mainStream>>>(aggr_temp);
            cudaErrorCheck(mainStream);
            initModelsKernel<C, K, M><<<1, BLOCK_SIZE, 0, mainStream>>>(aggr_temp, mixture_weights, means, covs_inv, covs_det);
            cudaErrorCheck(mainStream);
        }

        template <int C, int K, int M>
        __global__ void inferKernel(
            int N,
            const float* features, const int8_t* labels,
            const float* means, const float* covs_inv, const float* covs_det, float* out_llhoods)
        {
            const int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= N)
                return;

            const int gtm = labels ? labels[i] : -1; // ground truth label
            const int imm = gtm & 0x7F;              // imputed label
            float x[C];                              // data point
            float w;                                 // temporary value for the exponent
            const int KC = K * C;
            const int KCC = KC * C;
            const int CC = C * C;
            const float normal_coeff = 0.91893853 * C; // log ( (2*pi)^(Dims/2) )
            float modelEnergy;

#pragma unroll
            for (int m = 0; m < M; m++)
            {
                modelEnergy = 1e12f;
#pragma unroll
                for (int k = 0; k < K; k++)
                { // go over all Gaussian components of a single selected GMM m.
#pragma unroll
                    for (int c = 0; c < C; c++)
                        x[c] = features[i * C + c] - means[m * KC + k * C + c];

                    w = 0;

                    if (m != gtm) // ground truth match has the minimal energy
                    {
                        if (gtm >= 0 && gtm < M) // ground truth fail: add 3 sigma
                            w += 3 * C;
#pragma unroll
                        for (int ci = 0; ci < C; ci++)
#pragma unroll
                            for (int cj = 0; cj < C; cj++)
                                w += x[ci] * x[cj] * covs_inv[m * KCC + k * CC + ci * C + cj];

                        // imputed extra:
                        if (m == imm)
                            w *= 0.8f;
                        else if (imm < M)
                            w *= 1.1f;

                        w = max(w, 0.0f); // avoid numerical error
                    }

                    modelEnergy = min(w + __logf(max(EPS, covs_det[m * K + k])), modelEnergy);
                }
                out_llhoods[i * M + m] = -max(normal_coeff + 0.5f * modelEnergy, 0.0f);
            }
        }

        template <int C, int K, int M>
        void GaussianMixtures<C, K, M>::infer()
        {
            inferKernel<C, K, M><<<blocksForN, blockSizeForN, 0, mainStream>>>(N, features, labels, means, covs_inv, covs_det, llhoods);
        }

    } // namespace gmm
} // namespace salient