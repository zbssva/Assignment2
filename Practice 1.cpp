#include <iostream>      // для cout/cin
#include <vector>        // для vector (динамический массив C++)
#include <random>        // для генерации случайных чисел
#include <chrono>        // для измерения времени
#include <limits>        // для numeric_limits (очень большие/маленькие значения)
#include <omp.h>         // для OpenMP (параллельность)

using namespace std;

static vector<int> generate_vector(size_t N, int lo = 1, int hi = 100) {
    // random_device — источник случайности (может зависеть от системы)
    random_device rd;

    // mt19937 — генератор случайных чисел (хороший стандартный)
    mt19937 gen(rd());

    // dist(lo, hi) — распределение: числа будут от lo до hi включительно
    uniform_int_distribution<int> dist(lo, hi);

    // создаём вектор на N элементов
    vector<int> a(N);

    // заполняем каждый элемент случайным числом
    for (size_t i = 0; i < N; i++) {
        a[i] = dist(gen);
    }

    // возвращаем заполненный массив
    return a;
}

// ЧАСТЬ 2: СТРУКТУРЫ ДАННЫХ
// Node — один узел для списка/стека/очереди
struct Node {
    int value;    // значение
    Node* next;   // указатель на следующий узел

    // конструктор: создаём узел со значением v
    Node(int v) : value(v), next(nullptr) {}
};

// -------------------- ОДНОСВЯЗНЫЙ СПИСОК --------------------
struct SinglyLinkedList {
    Node* head = nullptr; // head — начало списка (первый элемент)

    // добавление в начало списка
    void push_front(int v) {
        // 1) создаём новый узел в куче (heap) через new
        Node* n = new Node(v);

        // 2) "следующий" нового узла становится текущей головой
        n->next = head;

        // 3) теперь голова — это новый узел
        head = n;
    }

    // поиск значения (возвращает true/false)
    bool find(int v) const {
        // начинаем с головы
        Node* cur = head;

        // идём по списку, пока не закончится (cur == nullptr)
        while (cur) {
            // если нашли нужное значение — сразу возвращаем true
            if (cur->value == v) return true;

            // иначе идём дальше
            cur = cur->next;
        }

        // если дошли до конца и не нашли
        return false;
    }

    // удаление первого найденного элемента со значением v
    bool remove(int v) {
        Node* cur = head;      // текущий элемент
        Node* prev = nullptr;  // предыдущий элемент (нужен, чтобы "перешить" ссылки)

        // идём по списку
        while (cur) {
            // если нашли нужный узел
            if (cur->value == v) {

                // если удаляем НЕ голову (prev существует)
                if (prev) {
                    // предыдущий начинает ссылаться на следующий, обходя cur
                    prev->next = cur->next;
                } else {
                    // если удаляем голову — просто сдвигаем head
                    head = cur->next;
                }

                // освобождаем память под удаляемый узел
                delete cur;

                return true; // успешно удалили
            }

            // двигаемся дальше: prev становится cur, cur становится следующим
            prev = cur;
            cur = cur->next;
        }

        // не нашли такой элемент
        return false;
    }

    // очистка всего списка (освобождение памяти)
    void clear() {
        Node* cur = head;

        while (cur) {
            // запоминаем следующий узел
            Node* nxt = cur->next;

            // удаляем текущий
            delete cur;

            // переходим дальше
            cur = nxt;
        }

        // список пустой
        head = nullptr;
    }

    // деструктор: когда список выходит из области видимости — чистим память
    ~SinglyLinkedList() {
        clear();
    }
};

//СТЕК (LIFO) 
// LIFO: последний вошёл — первый вышел
struct Stack {
    Node* top = nullptr; // top — вершина стека

    // push — положить элемент на вершину
    void push(int v) {
        Node* n = new Node(v); // создаём новый узел
        n->next = top;         // новый узел ссылается на старый top
        top = n;               // top теперь новый
    }

    // проверить пустой ли стек
    bool isEmpty() const {
        return top == nullptr;
    }

    // pop — достать элемент с вершины
    // out — куда записать значение
    bool pop(int &out) {
        // если пусто — ничего не достать
        if (isEmpty()) return false;

        // сохраняем узел вершины
        Node* n = top;

        // забираем значение
        out = n->value;

        // сдвигаем вершину вниз (следующий становится top)
        top = n->next;

        // освобождаем память
        delete n;

        return true;
    }

    // очистка стека
    void clear() {
        int tmp;
        while (pop(tmp)) {
            // просто вытаскиваем пока не пусто
        }
    }

    // деструктор: чистим память
    ~Stack() {
        clear();
    }
};

// ОЧЕРЕДЬ (FIFO) 
// FIFO: первый вошёл — первый вышел
struct Queue {
    Node* front = nullptr; // начало очереди (отсюда удаляем)
    Node* back  = nullptr; // конец очереди (сюда добавляем)

    // проверка на пустоту
    bool isEmpty() const {
        return front == nullptr;
    }

    // добавить в конец очереди
    void push_back(int v) {
        Node* n = new Node(v); // создаём новый узел

        // если очередь пустая — и front и back будут на этот узел
        if (isEmpty()) {
            front = back = n;
        } else {
            // иначе: старый back теперь указывает на новый
            back->next = n;
            // и back сдвигается на новый узел
            back = n;
        }
    }

    // удалить из начала очереди
    bool pop_front(int &out) {
        // если пусто — нечего удалять
        if (isEmpty()) return false;

        // берём первый узел
        Node* n = front;

        // сохраняем значение
        out = n->value;

        // сдвигаем front на следующий узел
        front = front->next;

        // если front стал nullptr — значит очередь стала пустой
        // тогда back тоже надо обнулить
        if (!front) back = nullptr;

        // удаляем старый первый узел
        delete n;

        return true;
    }

    // очистка очереди
    void clear() {
        int tmp;
        while (pop_front(tmp)) {
            // вытаскиваем пока не пусто
        }
    }

    // деструктор
    ~Queue() {
        clear();
    }
};

// ЧАСТЬ 3: среднее значение в динамическом массиве

// среднее последовательно
double average_seq(const int* arr, int N) {
    long long sum = 0;         // long long, чтобы сумма не переполнилась
    for (int i = 0; i < N; i++) {
        sum += arr[i];         // складываем все элементы
    }
    return (double)sum / N;    // делим на N и получаем среднее
}

// среднее параллельно (OpenMP reduction)
double average_par(const int* arr, int N) {
    long long sum = 0;

    // parallel for — делим цикл по потокам
    // reduction(+:sum) — каждый поток считает свою сумму,
    // потом OpenMP аккуратно складывает всё в одну sum
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) {
        sum += arr[i];
    }

    return (double)sum / N;
}

// MAIN 
int main() {
    // Пишем заголовок
    cout << "=== PRACTICE WORK #1 (ALL IN ONE) ===\n";

    // Печатаем, сколько максимально потоков OpenMP может использовать
    cout << "OpenMP max threads: " << omp_get_max_threads() << "\n\n";

    // ЧАСТЬ 1: Массив (vector) + min/max + OpenMP + время
    cout << "----- PART 1: ARRAY (min/max) -----\n";

    // N1 — размер массива (можно менять)
    const size_t N1 = 1000000;

    // создаём и заполняем массив случайными числами 1..100
    vector<int> a = generate_vector(N1, 1, 100);

    // ---------- Последовательный min/max ----------
    // t1 — время начала
    auto t1 = chrono::high_resolution_clock::now();

    // mn_seq — начально ставим очень большим
    int mn_seq = numeric_limits<int>::max();

    // mx_seq — начально ставим очень маленьким
    int mx_seq = numeric_limits<int>::min();

    // проходим по массиву
    for (size_t i = 0; i < N1; i++) {
        // обновляем минимум
        mn_seq = min(mn_seq, a[i]);
        // обновляем максимум
        mx_seq = max(mx_seq, a[i]);
    }

    // t2 — время конца
    auto t2 = chrono::high_resolution_clock::now();

    // считаем разницу во времени в миллисекундах
    auto seq_ms = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();

    // ---------- Параллельный min/max ----------
    auto t3 = chrono::high_resolution_clock::now();

    // начальные значения такие же
    int mn_par = numeric_limits<int>::max();
    int mx_par = numeric_limits<int>::min();

    // IMPORTANT:
    // reduction(min:mn_par) — правильное объединение минимумов
    // reduction(max:mx_par) — правильное объединение максимумов
    #pragma omp parallel for reduction(min:mn_par) reduction(max:mx_par)
    for (long long i = 0; i < (long long)N1; i++) {
        mn_par = min(mn_par, a[i]);
        mx_par = max(mx_par, a[i]);
    }

    auto t4 = chrono::high_resolution_clock::now();
    auto par_ms = chrono::duration_cast<chrono::milliseconds>(t4 - t3).count();

    // выводим результаты
    cout << "Sequential: min=" << mn_seq << " max=" << mx_seq << " time=" << seq_ms << " ms\n";
    cout << "Parallel:   min=" << mn_par << " max=" << mx_par << " time=" << par_ms << " ms\n\n";

    // ЧАСТЬ 2: Список + Стек + Очередь + параллельное добавление
    cout << "----- PART 2: DATA STRUCTURES -----\n";

    // ---- Демонстрация списка ----
    SinglyLinkedList list;

    // добавим элементы в начало: (30 -> 20 -> 10) по логике push_front
    list.push_front(10);
    list.push_front(20);
    list.push_front(30);

    // ищем 20
    cout << "List: find 20 -> " << (list.find(20) ? "yes" : "no") << "\n";

    // удаляем 20
    cout << "List: remove 20 -> " << (list.remove(20) ? "removed" : "not found") << "\n";

    // проверяем снова
    cout << "List: find 20 -> " << (list.find(20) ? "yes" : "no") << "\n";

    // ---- Демонстрация стека ----
    Stack st;

    // кладём 1,2,3 — вершина будет 3
    st.push(1);
    st.push(2);
    st.push(3);

    // вынимаем один элемент
    int x;
    st.pop(x);

    // ожидаем 3, потому что LIFO
    cout << "Stack: pop -> " << x << " (LIFO)\n";

    // ---- Демонстрация очереди ----
    Queue q_demo;

    // добавляем 100,200,300 — первым выйдет 100
    q_demo.push_back(100);
    q_demo.push_back(200);
    q_demo.push_back(300);

    // удаляем из начала
    q_demo.pop_front(x);

    // ожидаем 100, потому что FIFO
    cout << "Queue: pop_front -> " << x << " (FIFO)\n";

    // ---- Сравнение скорости добавления в очередь: seq vs omp ----
    // поэтому для параллельного добавления используем critical 

    const int M = 200000; // количество элементов для теста

    // делаем массив значений 0..M-1
    vector<int> data(M);
    for (int i = 0; i < M; i++) data[i] = i;

    // ---------- Последовательное добавление ----------
    Queue q_seq;

    auto s1 = chrono::high_resolution_clock::now();
    for (int i = 0; i < M; i++) {
        q_seq.push_back(data[i]);
    }
    auto s2 = chrono::high_resolution_clock::now();
    auto qseq_ms = chrono::duration_cast<chrono::milliseconds>(s2 - s1).count();

    // ---------- Параллельное добавление ----------
    Queue q_par;

    auto p1 = chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int i = 0; i < M; i++) {

        // critical — означает: в этот блок заходит только один поток за раз
        // иначе очередь "сломается" из-за одновременного изменения указателей
        #pragma omp critical
        {
            q_par.push_back(data[i]);
        }
    }

    auto p2 = chrono::high_resolution_clock::now();
    auto qpar_ms = chrono::duration_cast<chrono::milliseconds>(p2 - p1).count();

    cout << "Queue push_back sequential time: " << qseq_ms << " ms\n";
    cout << "Queue push_back parallel time (critical): " << qpar_ms << " ms\n\n";

    // ЧАСТЬ 3: Динамический массив (new/delete) + среднее + OpenMP
    cout << "----- PART 3: DYNAMIC ARRAY + AVERAGE -----\n";

    const int N3 = 5000000; // размер динамического массива

    // выделяем память под N3 int в куче
    int* arr = new int[N3];

    // генератор для заполнения
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(1, 100);

    // заполняем массив случайными числами
    for (int i = 0; i < N3; i++) {
        arr[i] = dist(gen);
    }

    // среднее последовательно
    auto a1t = chrono::high_resolution_clock::now();
    double avg1 = average_seq(arr, N3);
    auto a2t = chrono::high_resolution_clock::now();
    auto avg_seq_ms = chrono::duration_cast<chrono::milliseconds>(a2t - a1t).count();

    // среднее параллельно (reduction)
    auto a3t = chrono::high_resolution_clock::now();
    double avg2 = average_par(arr, N3);
    auto a4t = chrono::high_resolution_clock::now();
    auto avg_par_ms = chrono::duration_cast<chrono::milliseconds>(a4t - a3t).count();

    // вывод результатов
    cout << "Average (seq): " << avg1 << " time=" << avg_seq_ms << " ms\n";
    cout << "Average (par): " << avg2 << " time=" << avg_par_ms << " ms\n";

    // освобождаем память
    delete[] arr;
    cout << "\nMemory freed (delete[]).\n";

    cout << "\n=== DONE ===\n";
    return 0;
}
