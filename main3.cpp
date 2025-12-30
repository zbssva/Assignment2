#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <functional>

using namespace std;

static vector<int> generate_array(size_t n, int lo = -1000000, int hi = 1000000) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(lo, hi);

    vector<int> a(n);
    for (size_t i = 0; i < n; i++) a[i] = dist(gen);
    return a;
}

static void selection_sort_seq(vector<int>& a) {
    const int n = (int)a.size();
    for (int i = 0; i < n - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n; j++) {
            if (a[j] < a[min_idx]) min_idx = j;
        }
        swap(a[i], a[min_idx]);
    }
}

// Параллелим поиск min_idx на каждом i
static void selection_sort_omp(vector<int>& a) {
    const int n = (int)a.size();
    for (int i = 0; i < n - 1; i++) {
        int global_min_idx = i;

        #pragma omp parallel
        {
            int local_min_idx = global_min_idx;

            #pragma omp for nowait
            for (int j = i + 1; j < n; j++) {
                if (a[j] < a[local_min_idx]) local_min_idx = j;
            }

            #pragma omp critical
            {
                if (a[local_min_idx] < a[global_min_idx]) {
                    global_min_idx = local_min_idx;
                }
            }
        }

        swap(a[i], a[global_min_idx]);
    }
}

static long long time_us(function<void()> fn) {
    auto t1 = chrono::high_resolution_clock::now();
    fn();
    auto t2 = chrono::high_resolution_clock::now();
    return chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
}

int main() {
    vector<size_t> sizes = {1000, 10000};

    for (size_t N : sizes) {
        auto base = generate_array(N);

        auto a1 = base;
        auto a2 = base;

        long long t_seq = time_us([&](){ selection_sort_seq(a1); });
        long long t_omp = time_us([&](){ selection_sort_omp(a2); });

        bool ok = (a1 == a2);
        cout << "N=" << N << "\n";
        cout << "Seq: " << t_seq << " us\n";
        cout << "OMP: " << t_omp << " us\n";
        cout << "Equal: " << (ok ? "YES" : "NO") << "\n";
        if (t_omp > 0) cout << "Speedup ~ " << (double)t_seq/(double)t_omp << "x\n";
        cout << "----\n";
    }

    // Выводы:
    // Selection sort имеет O(n^2), а распараллеливание только поиска минимума
    // дает ограниченный выигрыш, особенно на малых N.
    // На 10000 может быть чуть лучше, но часто overhead тоже заметен.
    return 0;
}
