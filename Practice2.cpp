// Practical Work #2
// Parallel sorting (Bubble/Selection/Insertion) on CPU using OpenMP
//
// Компиляция:
// g++ -O2 -fopenmp task2.cpp -o task2
// (на Windows MinGW: g++ -O2 -fopenmp task2.cpp -o task2.exe)
//
// Запуск:
// ./task2
//
// В этом файле:
// 1) Последовательные версии: bubble, selection, insertion
// 2) Параллельные версии:
//    - "Bubble" -> odd-even transposition sort (это безопасная параллельная версия "пузырька")
//    - Selection -> параллелим поиск минимума
//    - Insertion -> почти не параллелится (зависимость шагов), оставим корректно, но без ощутимого ускорения
// 3) Бенчмарк на N = 1000, 10000, 100000 с chrono

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <omp.h>

using namespace std;

// ---------- Генерация массива ----------
static vector<int> generate_array(size_t n, int lo = -1000000, int hi = 1000000) {
    // random_device и mt19937 — стандартный способ делать "хороший" рандом
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(lo, hi);

    vector<int> a(n);
    for (size_t i = 0; i < n; i++) {
        a[i] = dist(gen);
    }
    return a;
}

// ---------- Проверка, что массив отсортирован ----------
static bool is_sorted_non_decreasing(const vector<int>& a) {
    for (size_t i = 1; i < a.size(); i++) {
        if (a[i - 1] > a[i]) return false;
    }
    return true;
}

// =======================================================
// 1) ПОСЛЕДОВАТЕЛЬНЫЕ СОРТИРОВКИ
// =======================================================

// ---- Bubble Sort (классический пузырёк) ----
static void bubble_sort_seq(vector<int>& a) {
    // Идея: много раз пробегаем массив, сравниваем соседей и меняем местами
    size_t n = a.size();
    for (size_t pass = 0; pass < n; pass++) {
        // После каждого прохода самый большой элемент "всплывает" вправо
        bool swapped = false;
        for (size_t j = 0; j + 1 < n - pass; j++) {
            if (a[j] > a[j + 1]) {
                swap(a[j], a[j + 1]);
                swapped = true;
            }
        }
        // Если за проход не было обменов — массив уже отсортирован
        if (!swapped) break;
    }
}

// ---- Selection Sort (сортировка выбором) ----
static void selection_sort_seq(vector<int>& a) {
    // Идея: для каждой позиции i ищем минимум на отрезке [i..n-1], ставим в i
    size_t n = a.size();
    for (size_t i = 0; i < n; i++) {
        size_t min_idx = i;
        for (size_t j = i + 1; j < n; j++) {
            if (a[j] < a[min_idx]) min_idx = j;
        }
        swap(a[i], a[min_idx]);
    }
}

// ---- Insertion Sort (сортировка вставками) ----
static void insertion_sort_seq(vector<int>& a) {
    // Идея: слева поддерживаем "уже отсортированную часть"
    // Берём следующий элемент и вставляем его в нужное место слева
    size_t n = a.size();
    for (size_t i = 1; i < n; i++) {
        int key = a[i];         // элемент, который будем вставлять
        long long j = (long long)i - 1;

        // сдвигаем вправо всё, что больше key
        while (j >= 0 && a[(size_t)j] > key) {
            a[(size_t)j + 1] = a[(size_t)j];
            j--;
        }
        // ставим key в освободившееся место
        a[(size_t)j + 1] = key;
    }
}

// =======================================================
// 2) ПАРАЛЛЕЛЬНЫЕ СОРТИРОВКИ (OpenMP)
// =======================================================

// ---- "Параллельный пузырёк" через Odd-Even Transposition Sort ----
// Почему так?
// Классический пузырёк одновременно делает swap соседей, и если это распараллелить "как есть",
// потоки начнут мешать друг другу (гонки данных).
//
// Odd-even сортировка работает фазами:
// - odd-фаза: сравниваем пары (1,2), (3,4), (5,6) ...
// - even-фаза: сравниваем пары (0,1), (2,3), (4,5) ...
// В каждой фазе пары НЕ пересекаются, значит можно безопасно распараллелить.
static void bubble_sort_omp_oddeven(vector<int>& a) {
    size_t n = a.size();

    // Делаем максимум n фаз — этого достаточно, чтобы гарантированно отсортировать
    for (size_t phase = 0; phase < n; phase++) {

        // Выбираем старт:
        // чётная фаза -> start=0 (even pairs)
        // нечётная фаза -> start=1 (odd pairs)
        size_t start = phase % 2;

        // Внутри фазы каждая итерация работает со своей парой (i, i+1), пары не пересекаются.
        // Поэтому можно безопасно #pragma omp parallel for
        #pragma omp parallel for
        for (long long i = (long long)start; i + 1 < (long long)n; i += 2) {
            if (a[(size_t)i] > a[(size_t)i + 1]) {
                int tmp = a[(size_t)i];
                a[(size_t)i] = a[(size_t)i + 1];
                a[(size_t)i + 1] = tmp;
            }
        }
    }
}

// ---- Selection Sort (параллелим поиск минимума) ----
// На каждом i нам нужно найти min на отрезке [i..n-1].
// Это можно делать параллельно, но аккуратно:
// каждый поток ищет свой локальный минимум, потом выбираем общий.
static void selection_sort_omp(vector<int>& a) {
    size_t n = a.size();

    for (size_t i = 0; i < n; i++) {

        int global_min_val = a[i];
        size_t global_min_idx = i;

        // Параллельный регион: каждый поток будет иметь свой локальный минимум
        #pragma omp parallel
        {
            int local_min_val = global_min_val;
            size_t local_min_idx = global_min_idx;

            // Раздаём j по потокам
            #pragma omp for nowait
            for (long long j = (long long)i + 1; j < (long long)n; j++) {
                if (a[(size_t)j] < local_min_val) {
                    local_min_val = a[(size_t)j];
                    local_min_idx = (size_t)j;
                }
            }

            // Теперь нужно "свести" локальные минимумы в один общий.
            // Делать это будем через critical (это простая, понятная версия).
            #pragma omp critical
            {
                if (local_min_val < global_min_val) {
                    global_min_val = local_min_val;
                    global_min_idx = local_min_idx;
                }
            }
        }

        // Ставим найденный минимум на позицию i
        swap(a[i], a[global_min_idx]);
    }
}

// ---- Insertion Sort (параллельная версия) ----
// Вставки по смыслу идут последовательно: шаг i зависит от уже отсортированных [0..i-1].
// Если наивно распараллелить внешний цикл i — получится НЕПРАВИЛЬНО.
//
// Поэтому здесь честный вывод: ускорение почти невозможно таким способом.
// Мы оставим алгоритм как последовательный (для корректности),
// но покажем структуру и объяснение.
static void insertion_sort_omp(vector<int>& a) {
    // Здесь просто вызываем последовательную версию.
    // Это "параллельная" часть в смысле исследования:
    // ты увидишь, что для вставки OpenMP почти не даёт выигрыш.
    insertion_sort_seq(a);
}

// =======================================================
// 3) БЕНЧМАРК (измерение времени)
// =======================================================

template <typename Func>
static long long measure_ms(Func f, vector<int>& a) {
    // Засекаем время ДО
    auto t1 = chrono::high_resolution_clock::now();

    // Запускаем сортировку
    f(a);

    // Засекаем время ПОСЛЕ
    auto t2 = chrono::high_resolution_clock::now();

    // Считаем разницу в миллисекундах
    return chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
}

static void run_one_size(size_t N) {
    cout << "\n==============================\n";
    cout << "N = " << N << "\n";
    cout << "Threads (OpenMP max) = " << omp_get_max_threads() << "\n";

    // Генерируем исходный массив
    vector<int> base = generate_array(N);

    // ---------------- Bubble ----------------
    {
        vector<int> a1 = base;
        vector<int> a2 = base;

        long long t_seq = measure_ms(bubble_sort_seq, a1);
        long long t_omp = measure_ms(bubble_sort_omp_oddeven, a2);

        cout << "\n[Bubble Sort]\n";
        cout << "Sequential: " << t_seq << " ms"
             << " | sorted=" << (is_sorted_non_decreasing(a1) ? "YES" : "NO") << "\n";
        cout << "OpenMP(odd-even): " << t_omp << " ms"
             << " | sorted=" << (is_sorted_non_decreasing(a2) ? "YES" : "NO") << "\n";
    }

    // ---------------- Selection ----------------
    {
        vector<int> a1 = base;
        vector<int> a2 = base;

        long long t_seq = measure_ms(selection_sort_seq, a1);
        long long t_omp = measure_ms(selection_sort_omp, a2);

        cout << "\n[Selection Sort]\n";
        cout << "Sequential: " << t_seq << " ms"
             << " | sorted=" << (is_sorted_non_decreasing(a1) ? "YES" : "NO") << "\n";
        cout << "OpenMP(parallel min): " << t_omp << " ms"
             << " | sorted=" << (is_sorted_non_decreasing(a2) ? "YES" : "NO") << "\n";
    }

    // ---------------- Insertion ----------------
    {
        vector<int> a1 = base;
        vector<int> a2 = base;

        long long t_seq = measure_ms(insertion_sort_seq, a1);
        long long t_omp = measure_ms(insertion_sort_omp, a2);

        cout << "\n[Insertion Sort]\n";
        cout << "Sequential: " << t_seq << " ms"
             << " | sorted=" << (is_sorted_non_decreasing(a1) ? "YES" : "NO") << "\n";
        cout << "OpenMP(note: почти без ускорения): " << t_omp << " ms"
             << " | sorted=" << (is_sorted_non_decreasing(a2) ? "YES" : "NO") << "\n";
    }

    cout << "\nВывод по N=" << N << ": эти сортировки O(n^2), поэтому рост времени очень быстрый.\n";
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cout << "Practical Work #2 — OpenMP Sorting Benchmark\n";
    cout << "OpenMP is enabled. If not, check compilation flags.\n";

    // Если хочешь задать конкретное число потоков — раскомментируй:
    // omp_set_num_threads(4);

    const bool RUN_BIG_N = true; // поменяй на false, если 100000 слишком долго

    run_one_size(1000);
    run_one_size(10000);
    if (RUN_BIG_N) run_one_size(100000);

    cout << "\nDone.\n";
    return 0;
}
