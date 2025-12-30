#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <chrono>
#include <omp.h>

using namespace std;

static vector<int> generate_array(size_t n, int lo = -1000000, int hi = 1000000) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(lo, hi);

    vector<int> a(n);
    for (size_t i = 0; i < n; i++) a[i] = dist(gen);
    return a;
}

int main() {
    std::cout << "START\n";

    const size_t N = 10000;
    vector<int> a = generate_array(N);

    // --- Sequential ---
    auto t1 = chrono::high_resolution_clock::now();
    int mn_seq = numeric_limits<int>::max();
    int mx_seq = numeric_limits<int>::min();
    for (size_t i = 0; i < N; i++) {
        mn_seq = min(mn_seq, a[i]);
        mx_seq = max(mx_seq, a[i]);
    }
    auto t2 = chrono::high_resolution_clock::now();
    auto seq_us = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();

    // --- Parallel OpenMP ---
    auto t3 = chrono::high_resolution_clock::now();
    int mn_par = numeric_limits<int>::max();
    int mx_par = numeric_limits<int>::min();

    #pragma omp parallel
    {
        int local_min = numeric_limits<int>::max();
        int local_max = numeric_limits<int>::min();

        #pragma omp for nowait
        for (int i = 0; i < (int)N; i++) {
            local_min = min(local_min, a[i]);
            local_max = max(local_max, a[i]);
        }

        #pragma omp critical
        {
            mn_par = min(mn_par, local_min);
            mx_par = max(mx_par, local_max);
        }
    }

    auto t4 = chrono::high_resolution_clock::now();
    auto par_us = chrono::duration_cast<chrono::microseconds>(t4 - t3).count();

    cout << "N = " << N << "\n";
    cout << "Sequential: min=" << mn_seq << " max=" << mx_seq << " time=" << seq_us << " us\n";
    cout << "OpenMP    : min=" << mn_par << " max=" << mx_par << " time=" << par_us << " us\n";

    if (mn_seq != mn_par || mx_seq != mx_par) {
        cout << "WARNING: results mismatch!\n";
    }

    double speedup = (par_us > 0) ? (double)seq_us / (double)par_us : 0.0;
    cout << "Speedup ~ " << speedup << "x\n";
    std::cout << "END\n";

    // Выводы:
    // На N=10000 ускорение может быть небольшим,
    // потому что сама операция min/max очень быстрая, а overhead OpenMP заметен.
    // На больших N ускорение обычно растет.
    return 0;
}
