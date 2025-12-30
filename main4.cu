#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

using namespace std;

#define CUDA_CHECK(call) do {                                  \
    cudaError_t err = call;                                    \
    if (err != cudaSuccess) {                                  \
        cerr << "CUDA error: " << cudaGetErrorString(err)      \
             << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        exit(1);                                               \
    }                                                          \
} while(0)

static vector<int> generate_array(size_t n, int lo=-1000000, int hi=1000000) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(lo, hi);
    vector<int> a(n);
    for (size_t i=0;i<n;i++) a[i]=dist(gen);
    return a;
}

// Простая сортировка внутри блока для своего чанка (odd-even sort)
// Для демонстрации: работает на небольших чанках (например 1024/2048)
__global__ void block_sort_kernel(int* data, int n, int chunk) {
    int bid = blockIdx.x;
    int start = bid * chunk;
    int end = min(start + chunk, n);
    int len = end - start;

    // сортируем data[start..end) внутри блока
    // odd-even sort: O(len^2) но чанк небольшой
    for (int phase = 0; phase < len; phase++) {
        int idx = threadIdx.x;
        int offset = phase % 2;
        for (int i = idx * 2 + offset; i + 1 < len; i += blockDim.x * 2) {
            int a = start + i;
            int b = start + i + 1;
            if (data[a] > data[b]) {
                int tmp = data[a];
                data[a] = data[b];
                data[b] = tmp;
            }
        }
        __syncthreads();
    }
}

// Слияние двух отсортированных отрезков [left, mid) и [mid, right)
// Пишем в out
__global__ void merge_pass_kernel(const int* in, int* out, int n, int width) {
    int pair_id = blockIdx.x; // каждая "пара" отрезков - отдельный блок
    int left = pair_id * 2 * width;
    int mid  = min(left + width, n);
    int right= min(left + 2 * width, n);

    int i = left;
    int j = mid;
    int k = left;

    // Один блок последовательно сливает свой диапазон (просто и надежно для задания)
    // Можно ускорять, но тут цель - корректность и структура.
    if (threadIdx.x == 0) {
        while (i < mid && j < right) {
            if (in[i] <= in[j]) out[k++] = in[i++];
            else out[k++] = in[j++];
        }
        while (i < mid) out[k++] = in[i++];
        while (j < right) out[k++] = in[j++];
    }
}

static double run_gpu_merge_sort(vector<int> h, int chunk) {
    int n = (int)h.size();
    int *d_a=nullptr, *d_b=nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_a, h.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // 1) block sort
    int blocks = (n + chunk - 1) / chunk;
    block_sort_kernel<<<blocks, 256>>>(d_a, n, chunk);
    CUDA_CHECK(cudaGetLastError());

    // 2) iterative merge passes
    int width = chunk;
    bool flip = false;
    while (width < n) {
        int pairs = (n + 2*width - 1) / (2*width);
        if (!flip) merge_pass_kernel<<<pairs, 256>>>(d_a, d_b, n, width);
        else       merge_pass_kernel<<<pairs, 256>>>(d_b, d_a, n, width);
        CUDA_CHECK(cudaGetLastError());
        flip = !flip;
        width *= 2;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // optional: verify by copying back
    vector<int> out(n);
    if (!flip) {
        CUDA_CHECK(cudaMemcpy(out.data(), d_a, n*sizeof(int), cudaMemcpyDeviceToHost));
    } else {
        CUDA_CHECK(cudaMemcpy(out.data(), d_b, n*sizeof(int), cudaMemcpyDeviceToHost));
    }

    // correctness check vs CPU sort
    auto ref = h;
    sort(ref.begin(), ref.end());
    if (out != ref) {
        cerr << "WARNING: GPU sort mismatch!\n";
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return (double)ms;
}

static double run_cpu_sort(vector<int> h) {
    auto t1 = chrono::high_resolution_clock::now();
    sort(h.begin(), h.end());
    auto t2 = chrono::high_resolution_clock::now();
    return chrono::duration_cast<chrono::microseconds>(t2 - t1).count() / 1000.0; // ms
}

int main() {
    vector<int> sizes = {10000, 100000};
    int chunk = 1024; // чанк на блок (можно 2048, но тогда внутри блока сортировка тяжелее)

    cout << "Chunk = " << chunk << "\n";

    for (int n : sizes) {
        auto arr = generate_array(n);

        double cpu_ms = run_cpu_sort(arr);
        double gpu_ms = run_gpu_merge_sort(arr, chunk);

        cout << "N=" << n << "\n";
        cout << "CPU std::sort : " << cpu_ms << " ms\n";
        cout << "GPU merge sort: " << gpu_ms << " ms\n";
        if (gpu_ms > 0) cout << "Speedup ~ " << cpu_ms / gpu_ms << "x\n";
        cout << "----\n";
    }

    // Выводы:
    // Для 10k GPU может не сильно выигрывать из-за накладных расходов,
    // для 100k чаще видно преимущество (зависит от GPU и параметров chunk).
    return 0;
}
