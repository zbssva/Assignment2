#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <stack>
#include <vector>
#include <limits>

// Проверка ошибок CUDA
#define CUDA_CHECK(call) do {                                  \
    cudaError_t err = (call);                                  \
    if (err != cudaSuccess) {                                  \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)  \
                  << " at " << __FILE__ << ":" << __LINE__      \
                  << std::endl;                                 \
        std::exit(1);                                          \
    }                                                          \
} while(0)

// Генерация массива
static std::vector<int> generate_array(size_t n, int lo=-1'000'000, int hi=1'000'000) {
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(lo, hi);
    std::vector<int> a(n);
    for (size_t i = 0; i < n; i++) a[i] = dist(rng);
    return a;
}

static bool is_sorted_cpu(const std::vector<int>& a) {
    for (size_t i = 1; i < a.size(); i++) {
        if (a[i-1] > a[i]) return false;
    }
    return true;
}

// CPU: сортировки

// CPU Merge Sort 
static void merge_cpu(std::vector<int>& a, std::vector<int>& tmp, int l, int m, int r) {
    int i = l, j = m, k = l;
    while (i < m && j < r) {
        if (a[i] <= a[j]) tmp[k++] = a[i++];
        else tmp[k++] = a[j++];
    }
    while (i < m) tmp[k++] = a[i++];
    while (j < r) tmp[k++] = a[j++];
    for (int t = l; t < r; t++) a[t] = tmp[t];
}

static void merge_sort_cpu_rec(std::vector<int>& a, std::vector<int>& tmp, int l, int r) {
    if (r - l <= 1) return;
    int m = l + (r - l) / 2;
    merge_sort_cpu_rec(a, tmp, l, m);
    merge_sort_cpu_rec(a, tmp, m, r);
    merge_cpu(a, tmp, l, m, r);
}

static void merge_sort_cpu(std::vector<int>& a) {
    std::vector<int> tmp(a.size());
    merge_sort_cpu_rec(a, tmp, 0, (int)a.size());
}

// CPU Quick Sort 
static int partition_cpu(std::vector<int>& a, int l, int r) {
    int pivot = a[l + (r - l) / 2];
    int i = l, j = r - 1;
    while (i <= j) {
        while (a[i] < pivot) i++;
        while (a[j] > pivot) j--;
        if (i <= j) std::swap(a[i++], a[j--]);
    }
    return i; // точка раздела
}

static void quick_sort_cpu_rec(std::vector<int>& a, int l, int r) {
    if (r - l <= 1) return;
    int mid = partition_cpu(a, l, r);
    quick_sort_cpu_rec(a, l, mid);
    quick_sort_cpu_rec(a, mid, r);
}

static void quick_sort_cpu(std::vector<int>& a) {
    quick_sort_cpu_rec(a, 0, (int)a.size());
}

// CPU Heap Sort 
static void heapify_cpu(std::vector<int>& a, int n, int i) {
    while (true) {
        int largest = i;
        int L = 2*i + 1;
        int R = 2*i + 2;
        if (L < n && a[L] > a[largest]) largest = L;
        if (R < n && a[R] > a[largest]) largest = R;
        if (largest == i) break;
        std::swap(a[i], a[largest]);
        i = largest;
    }
}

static void heap_sort_cpu(std::vector<int>& a) {
    int n = (int)a.size();
    for (int i = n/2 - 1; i >= 0; --i) heapify_cpu(a, n, i);
    for (int end = n - 1; end > 0; --end) {
        std::swap(a[0], a[end]);
        heapify_cpu(a, end, 0);
    }
}

//                 GPU: вспомогательные ядра

// Простая сортировка вставками в shared памяти внутри блока
// Нужна как "локальная" сортировка кусков для merge sort
__global__ void block_insertion_sort(int* d, int n, int chunk) {
    // chunk — размер куска (например 1024)
    int blockId = blockIdx.x;
    int start = blockId * chunk;
    int end = min(start + chunk, n);

    // Один блок сортирует свой кусок.
    // Для простоты — один поток делает insertion sort (учебно, но работает).
    // Можно усложнить до bitonic, но для практики достаточно.
    if (threadIdx.x == 0) {
        for (int i = start + 1; i < end; i++) {
            int key = d[i];
            int j = i - 1;
            while (j >= start && d[j] > key) {
                d[j + 1] = d[j];
                j--;
            }
            d[j + 1] = key;
        }
    }
}

// Merge одного прохода: сливаем пары отсортированных отрезков длины width
// src -> dst
__global__ void merge_pass_kernel(const int* src, int* dst, int n, int width) {
    int pairId = blockIdx.x * blockDim.x + threadIdx.x;
    int left = pairId * (2 * width);
    if (left >= n) return;

    int mid = min(left + width, n);
    int right = min(left + 2 * width, n);

    int i = left;
    int j = mid;
    int k = left;

    // обычное слияние двух отсортированных кусков
    while (i < mid && j < right) {
        if (src[i] <= src[j]) dst[k++] = src[i++];
        else dst[k++] = src[j++];
    }
    while (i < mid) dst[k++] = src[i++];
    while (j < right) dst[k++] = src[j++];
}

//          GPU Merge Sort (bottom-up)
static void merge_sort_gpu(int* d_arr, int n) {
    // 1) разбиваем на куски и сортируем каждый кусок
    const int CHUNK = 1024; // можно менять
    int blocks = (n + CHUNK - 1) / CHUNK;
    block_insertion_sort<<<blocks, 256>>>(d_arr, n, CHUNK);
    CUDA_CHECK(cudaGetLastError());

    // 2) делаем поэтапные проходы слияния: width = CHUNK, 2*CHUNK, 4*CHUNK ...
    int* d_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tmp, n * sizeof(int)));

    int width = CHUNK;
    bool flip = false;

    while (width < n) {
        const int* src = flip ? d_tmp : d_arr;
        int* dst = flip ? d_arr : d_tmp;

        int pairs = (n + 2 * width - 1) / (2 * width);
        int threads = 256;
        int grid = (pairs + threads - 1) / threads;

        merge_pass_kernel<<<grid, threads>>>(src, dst, n, width);
        CUDA_CHECK(cudaGetLastError());

        flip = !flip;
        width *= 2;
    }

    // если результат оказался в tmp — копируем в d_arr
    if (flip) {
        CUDA_CHECK(cudaMemcpy(d_arr, d_tmp, n * sizeof(int), cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaFree(d_tmp));
}

//          GPU Quick Sort (итеративный)
//   Идея:
//   - Берем подмассив [l, r)
//   - Берем pivot
//   - Параллельно помечаем элементы < pivot
//   - Делаем exclusive_scan по этим меткам -> получаем позиции
//   - Scatter в temp массив
//   - Получаем newMid, пушим подзадачи в стек

__global__ void mark_less_kernel(const int* in, int* flags, int n, int l, int r, int pivot) {
    int idx = l + (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= r) return;
    flags[idx] = (in[idx] < pivot) ? 1 : 0;
}

__global__ void scatter_partition_kernel(const int* in, int* out,
                                        const int* flags, const int* scan,
                                        int l, int r, int pivot,
                                        int lessCount) {
    int idx = l + (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= r) return;

    int x = in[idx];
    int isLess = flags[idx];
    int posLess = scan[idx];       
    int pos = 0;

    if (isLess) {
        pos = l + posLess;
    } else {
        // элементы >= pivot идут после lessCount
        // считаем сколько не-less до нас: (idx-l) - posLess
        int notLessBefore = (idx - l) - posLess;
        pos = l + lessCount + notLessBefore;
    }
    out[pos] = x;
}

__global__ void copy_range_kernel(const int* src, int* dst, int l, int r) {
    int idx = l + (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= r) return;
    dst[idx] = src[idx];
}

static void insertion_sort_small_gpu(int* d_arr, int l, int r) {
    // маленький подмассив сортируем прямо на GPU одним потоком (учебно)
    // (для quicksort, когда задача маленькая)
    // Реализуем через cudaMemcpy на хост? — нет, лучше остаёмся на GPU.
    // Сделаем просто thrust::sort нельзя (нужен include), но можно:
    // Здесь оставим простой вариант: копия на host — но это уже “не честно”.
    // Поэтому делаем микро-сортировку одним потоком в kernel:

    // локальное ядро внутри функции
    auto kernel	ti = [] __global__ (int* a, int l, int r) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            for (int i = l + 1; i < r; i++) {
                int key = a[i];
                int j = i - 1;
                while (j >= l && a[j] > key) {
                    a[j + 1] = a[j];
                    j--;
                }
                a[j + 1] = key;
            }
        }
    };

    // запускаем 1 блок
    kernel	ti<<<1, 1>>>(d_arr, l, r);
    CUDA_CHECK(cudaGetLastError());
}

static void quick_sort_gpu(int* d_arr, int n) {
    int* d_tmp = nullptr;
    int* d_flags = nullptr;
    int* d_scan = nullptr;

    CUDA_CHECK(cudaMalloc(&d_tmp,   n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_flags, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scan,  n * sizeof(int)));

    std::stack<std::pair<int,int>> st;
    st.push({0, n});

    const int SMALL = 2048; // порог: маленькие сортируем вставками

    while (!st.empty()) {
        auto [l, r] = st.top();
        st.pop();

        int len = r - l;
        if (len <= 1) continue;

        if (len <= SMALL) {
            insertion_sort_small_gpu(d_arr, l, r);
            continue;
        }

        // Берём pivot: просто центральный элемент (копируем 1 int на host)
        int pivot = 0;
        CUDA_CHECK(cudaMemcpy(&pivot, d_arr + (l + len/2), sizeof(int), cudaMemcpyDeviceToHost));

        // 1) flags[idx] = (a[idx] < pivot)
        int threads = 256;
        int grid = (len + threads - 1) / threads;
        mark_less_kernel<<<grid, threads>>>(d_arr, d_flags, n, l, r, pivot);
        CUDA_CHECK(cudaGetLastError());

        // 2) exclusive_scan по flags на диапазоне [l, r)
        // scan[idx] = сколько "less" до idx
        // thrust работает с device_ptr
        thrust::device_ptr<int> fptr(d_flags);
        thrust::device_ptr<int> sptr(d_scan);
        thrust::exclusive_scan(fptr + l, fptr + r, sptr + l);

        // 3) lessCount = scan[r-1] + flags[r-1] (считаем на CPU, копируя 2 int)
        int lastScan = 0, lastFlag = 0;
        CUDA_CHECK(cudaMemcpy(&lastScan, d_scan + (r - 1), sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&lastFlag, d_flags + (r - 1), sizeof(int), cudaMemcpyDeviceToHost));
        int lessCount = lastScan + lastFlag;

        // 4) scatter: переносим элементы в d_tmp по новым позициям
        scatter_partition_kernel<<<grid, threads>>>(
            d_arr, d_tmp, d_flags, d_scan, l, r, pivot, lessCount
        );
        CUDA_CHECK(cudaGetLastError());

        // 5) копируем обратно [l, r)
        copy_range_kernel<<<grid, threads>>>(d_tmp, d_arr, l, r);
        CUDA_CHECK(cudaGetLastError());

        // 6) пушим две части как quicksort:
        // [l, l+lessCount) и [l+lessCount, r)
        // (pivot может оказаться не “на своем финальном месте” из-за >= pivot,
        //  но идея разбиения сохраняется; дальше рекурсия досортирует)
        int mid = l + lessCount;
        if (mid - l > 1) st.push({l, mid});
        if (r - mid > 1) st.push({mid, r});
    }

    CUDA_CHECK(cudaFree(d_tmp));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_scan));
}

//          GPU Heap Sort (корректный)
//   - build heap "уровнями" (параллельнее)
//   - далее извлекаем max: swap(0,end), heapify(0,end)
//     heapify делаем на GPU (один поток)

__device__ void heapify_down_device(int* a, int heapSize, int i) {
    while (true) {
        int largest = i;
        int L = 2*i + 1;
        int R = 2*i + 2;
        if (L < heapSize && a[L] > a[largest]) largest = L;
        if (R < heapSize && a[R] > a[largest]) largest = R;
        if (largest == i) break;
        int tmp = a[i];
        a[i] = a[largest];
        a[largest] = tmp;
        i = largest;
    }
}

__global__ void heapify_level_kernel(int* a, int heapSize, int startNode, int count) {
    // каждый поток heapify для одного узла на уровне
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    int node = startNode + t;
    heapify_down_device(a, heapSize, node);
}

__global__ void swap_root_with_end_kernel(int* a, int end) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int tmp = a[0];
        a[0] = a[end];
        a[end] = tmp;
    }
}

__global__ void heapify_root_kernel(int* a, int heapSize) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        heapify_down_device(a, heapSize, 0);
    }
}

static void heap_sort_gpu(int* d_arr, int n) {
    // 1) build heap снизу вверх "уровнями"
    // последний внутренний узел: n/2 - 1
    // по уровням: можно примерно считать так:
    // идём от нижних узлов к корню, но пачками (это и есть распараллеливание)
    int lastInternal = n/2 - 1;
    while (lastInternal >= 0) {
        // Возьмём "пачку" узлов и обработаем параллельно
        // Для простоты — обрабатываем по 4096 узлов за запуск
        int batch = std::min(4096, lastInternal + 1);
        int startNode = lastInternal - batch + 1;

        int threads = 256;
        int grid = (batch + threads - 1) / threads;
        heapify_level_kernel<<<grid, threads>>>(d_arr, n, startNode, batch);
        CUDA_CHECK(cudaGetLastError());

        lastInternal -= batch;
    }

    // 2) извлекаем максимум: end от n-1 до 1
    for (int end = n - 1; end > 0; --end) {
        swap_root_with_end_kernel<<<1, 1>>>(d_arr, end);
        CUDA_CHECK(cudaGetLastError());
        heapify_root_kernel<<<1, 1>>>(d_arr, end);
        CUDA_CHECK(cudaGetLastError());
    }
}

//                 Замеры времени

static double time_cpu_ms(void(*sort_fn)(std::vector<int>&), std::vector<int> a) {
    auto t1 = std::chrono::high_resolution_clock::now();
    sort_fn(a);
    auto t2 = std::chrono::high_resolution_clock::now();
    if (!is_sorted_cpu(a)) {
        std::cerr << "CPU result NOT sorted!\n";
        std::exit(1);
    }
    return std::chrono::duration<double, std::milli>(t2 - t1).count();
}

static float time_gpu_ms(void(*gpu_fn)(int*, int), const std::vector<int>& a) {
    int n = (int)a.size();
    int* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, a.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    gpu_fn(d, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // проверка (копируем назад и проверяем на CPU)
    std::vector<int> out(n);
    CUDA_CHECK(cudaMemcpy(out.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost));
    if (!is_sorted_cpu(out)) {
        std::cerr << "GPU result NOT sorted!\n";
        std::exit(1);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d));
    return ms;
}

// Обертки под сигнатуру time_cpu_ms
static void cpu_merge_wrap(std::vector<int>& a){ merge_sort_cpu(a); }
static void cpu_quick_wrap(std::vector<int>& a){ quick_sort_cpu(a); }
static void cpu_heap_wrap (std::vector<int>& a){ heap_sort_cpu(a); }

//                      MAIN
int main() {
    std::cout << "Lab 3 CUDA Sorts (CPU vs GPU)\n";

    std::vector<int> sizes = {10'000, 100'000, 1'000'000};

    for (int n : sizes) {
        std::cout << "\n=============================\n";
        std::cout << "N = " << n << "\n";

        auto base = generate_array(n);

        // -------- CPU --------
        double cpu_merge = time_cpu_ms(cpu_merge_wrap, base);
        double cpu_quick = time_cpu_ms(cpu_quick_wrap, base);
        double cpu_heap  = time_cpu_ms(cpu_heap_wrap,  base);

        // -------- GPU --------
        float gpu_merge = time_gpu_ms(merge_sort_gpu, base);
        float gpu_quick = time_gpu_ms(quick_sort_gpu, base);
        float gpu_heap  = time_gpu_ms(heap_sort_gpu,  base);

        std::cout << "CPU Merge: " << cpu_merge << " ms\n";
        std::cout << "CPU Quick: " << cpu_quick << " ms\n";
        std::cout << "CPU Heap : " << cpu_heap  << " ms\n";

        std::cout << "GPU Merge: " << gpu_merge << " ms\n";
        std::cout << "GPU Quick: " << gpu_quick << " ms\n";
        std::cout << "GPU Heap : " << gpu_heap  << " ms\n";

        std::cout << "\nSpeedup (CPU/GPU):\n";
        std::cout << "Merge: " << (cpu_merge / gpu_merge) << "x\n";
        std::cout << "Quick: " << (cpu_quick / gpu_quick) << "x\n";
        std::cout << "Heap : " << (cpu_heap  / gpu_heap ) << "x\n";

        std::cout << "=============================\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
