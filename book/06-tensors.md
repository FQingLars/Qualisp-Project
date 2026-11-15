# Глава 6. Тензоры

Тензоры — N-мерные массивы, центральный тип данных Qualisp-ML. Аналог `np.ndarray` / `torch.Tensor`, но с линейной семантикой владения (гл. 4–5).

## 6.1 Структура тензора

Из `src/tensor/tensor.hpp`:

```cpp
class Tensor {
public:
    enum DType : uint8_t { F32 = 0, F64, I32, I64, U8, F16 };
    DType dtype;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    size_t numel;
    void *data;             // данные: с заголовком в одном выделении
    Device *device;         // абстракция устройства; реализация одна — CPU

    bool requires_grad;
    Tensor *grad;
    GradNode *grad_node;    // узел ленты (журнал доступен как (TAPE-FORM), гл. 7)
    bool viewed;            // read-only окно: клон — всегда глубокая копия
    Tensor *m, *v;          // Adam state (хранится прямо на тензоре)
    int64_t step;
};
```

Поддерживаемые dtype: **F16, F32, F64, I32, I64, U8**. Размеры: `F16=2, F32=4, F64=8, I32=4, I64=8, U8=1` байт (`Tensor::dtype_size`).

Тензор — **одно выделение памяти**: `[заголовок | данные]` с единственным разрушителем. Тензорная запись принадлежит объекту и ретайрится его деструктором; отдельного владельца данных нет. Промежуточные тензоры живут в scratch-scope; буфер идёт через `TensorBufferPool` — exact-size аллокация (см. `Tensor::create`: `t->device->alloc_pooled(numel * dtype_size(dt))` с `arena.track_external`), кэш по (shape, dtype), при переполнении лимита (~1 ГиБ) буфер уходит системе. Прямые `malloc`/`free` для данных тензоров разрешены — пул остаётся путём по умолчанию.

`Device` — абстрактный интерфейс (выделение, пул, `launch_*`), и реализация у него одна: CPU (гл. 11). Тензор всегда создаётся на `default_device()`.

## 6.2 Создание

```lisp
;; Литерал — вложенные списки F32 (по умолчанию)
(defvar a (tensor ((1.0 2.0) (3.0 4.0))))     ;; 2x2 (F32)

;; Фабрики
(zeros (list 2 3))        ;; нули            np.zeros((2,3))
(ones (list 4 2))         ;; единицы         np.ones((4,2))
(randn (list 100 1))      ;; N(0,1)          np.random.randn(100,1)
(randint 1 6)             ;; случайное целое из [1,6)

;; Из списка
(list->tensor '(1.0 2.0 3.0))             ;; вектор [3]

;; Явный выбор dtype
(tensor-f16 (randn (list 4 4)))           ;; [4,4] F16 (примитив TENSOR-F16)
(tensor-f32 (randn (list 4 4)))           ;; [4,4] F32 (примитив TENSOR-F32)
(tensor-u8  ...)                          ;; U8
(tensor-dtype h)                          ;; → F16 / F32 / ...
```

> 🐍 **Python-аналогия.**
> ```python
> a = np.array([[1., 2.], [3., 4.]])
> np.zeros((2, 3)); np.ones((4, 2)); np.random.randn(100, 1)
> a.astype(np.float16)
> ```

## 6.3 Информация о тензоре

```lisp
(tensor-shape a)      ;; → (2 2)        a.shape
(tensor-dtype a)      ;; → F32          a.dtype
(tref t 0 0)          ;; элемент по индексам (TREF)        a[0,0]
(tensor->list t)      ;; → список значений       a.ravel().tolist()
(tensor-item t 0)     ;; элемент по индексу      t.reshape(-1)[0]
```

`tensor-item` — примитив `prim_tensor_item`, отдельный от `tensor->list` (для быстрого доступа к одному значению). Второй аргумент — индекс элемента: читается именно этот элемент, а не всегда нулевой. Выход за границу и отрицательный индекс — ошибка.

### Контракт индексации `tref`

Два способа адресации, оба с проверкой границ:

- **Один индекс** — плоский номер элемента (как `NTH` и как весь код проекта читает `(tref m 0)` на матрице): `(tref m 1)` на `(tensor ((1.0 2.0)))` → `2.0`.
- **N индексов** — по координатам, ровно по рангу: `(tref m 0 1)` на матрице `[2,2]`, три индекса на тензоре ранга 3; число индексов, отличное от ранга (кроме одного), — ошибка.

Индекс проверяется и по типу, и по границам: не-fixnum — ошибка, выход за границы — ошибка, а не значение из чужой памяти.

```lisp
(tref (tensor ((1.0 2.0))) 0)        ;; → 1.0   — плоский номер
(tref (tensor ((1.0 2.0)) (3.0 4.0)) 1 1)  ;; → 4.0 — по координатам
(try (tref (randn (list 4)) 4) (e t))  ;; ошибка: tref: index 4 out of range for 4 elements
```

### Запись элемента: `TSET!`

`(TSET! tensor index value)` — запись элемента **на месте** по плоскому индексу (как у `TENSOR-ITEM`), возвращает **тензор** — поэтому записи цепочкой в одну строку.

```lisp
(defvar t (tensor ((0.0 0.0 0.0 0.0))))
(TSET! t 2 3.5)        ;; → тензор (для цепочек)
(tensor-item t 2)      ;; → 3.5
```

Числовые тензоры — массивы, и правила записи строгие:

| Кто | Правило |
|---|---|
| float-тензоры (F16/F32/F64) | принимают любое число; fixnum пишется как float — `(TSET! t 1 7)` даёт `7.0` |
| целые (I32/I64/U8) | **только fixnum и строго в диапазоне типа**: без wrap и усечения — 255 в U8 допустимо, 256 и −1 — ошибка; flonum в целочисленный тензор — ошибка |
| индекс | неотрицательный fixnum в границах: за границей, отрицательный, нецелый — ошибка |
| значение | строка, `nil` и прочее нечисло — ошибка |

`TSET!` работает и на **borrowed-тензорах** — пишет прямо в память хоста, это главный сценарий edge «буфер → лисп → буфер» (гл. 16), — и **из JIT-кода** (гл. 20), в том числе в циклах:

```lisp
(defvar f (JIT (lambda (n)
                 (let ((t (tensor ((0.0 0.0 0.0 0.0)))) (i 0))
                   (progn (while (< i n)
                            (progn (TSET! t i (* i 10.0)) (setq i (+ i 1)))
                            nil)
                          t)))))
(tensor->list (f 3))   ;; → (0.0 10.0 20.0 0.0)
```

> 🐍 Это `t.flat[i] = v` (или `t[i] = v`) в NumPy — но с проверкой типа и диапазона на каждой записи.

### Умная печать

`display`/`print` показывают голову+хвост (до 20 элементов целиком), шейп и dtype:

```text
tensor([1, 2, ..., 29, 30], shape=[30, 1], dtype=f32)
```

Усечение — display-only: машинные формы (`GRAPH-DATA`, QSRD-блобы) хранят полные значения, так что круговой путь print → read → print точен.

## 6.4 Арифметика и матричные операции

| Qualisp | Что делает | NumPy |
|---|---|---|
| `(T+ a b)` | поэлементное сложение с broadcasting | `a + b` |
| `(T- a b)` | поэлементная разность | `a - b` |
| `(T* a b)` | поэлементное умножение | `a * b` |
| `(MATMUL a b)` | матричное умножение (2D × 2D) | `a @ b` |
| `(RESHAPE t (list 4 1))` | изменение формы (глубокая копия) | `t.reshape(4,1)` |
| `(TRANSPOSE t)` | транспонирование (глубокая копия) | `t.T` |
| `(CONCAT a b ...)` | конкатенация по первой оси (axis 0) | `np.concatenate` |
| `(STACK a b ...)` | стопка с новой осью в начале | `np.stack` |
| `(T+_IP a b)` / `(T*= a b)` | in-place варианты | `a += b` |
| `(T-RELU t)` | relu in-place | — |
| `(T-FMA a b c)` | fused multiply-add: `a*b+c` (одна SIMD-операция) | — |
| `(T-FMA-RELU a b c)` | `max(0, a*b+c)` | — |
| `(T-ROWS t from len)` | read-only окно по строкам | `a[from:from+len]` |
| `(T-SLICE t from len)` | read-only плоское окно | `a.ravel()[from:from+len]` |
| `(T-GATHER-ROWS t ids)` | копия строк по индексам | `np.take(t, ids, axis=0)` |
| `(T-SCATTER-ADD dst ids src)` | прибавить строки по индексам на месте | `np.add.at(dst, ids, src)` |
| `(TRANSPOSE! t)` / `(RESHAPE! t shape)` | дифференцируемые версии | `t.T` / `t.reshape` |
| `(SUM! t axes keepdims)` / `(MEAN! t axes keepdims)` | дифференцируемые редукции | `.sum()` / `.mean()` |

Все тензорные операции — именно с префиксом `T*` (или `MATMUL`/`RESHAPE`/`CONCAT`/…). Скалярная арифметика (`+`, `*`, `SQRT`, `EXP`, `LOG`) работает только с числами — тензоры не подставляются неявно. Поэлементные тензорные аналоги математики:

```lisp
(TSIN t)   (TCOS t)   (T/TANH t)   ;; тригонометрия по элементам
```

```lisp
(defvar x (randn (list 3 2)))
(defvar y (randn (list 2 4)))
(print (tensor-shape (matmul x y)))    ;; → (3 4)
```

> 🐍 `T+`/`T*` — `a + b`/`a * b`; `MATMUL` — `a @ b`.

Арность тензорных примитивов проверяется, а не роняет процесс: `T+`/`T-`/`T*`/`MATMUL`, `CONV2D`/`MAXPOOL2D`, `BATCHNORM!`/`LAYERNORM!` дают внятную ошибку («expected 4 arguments, got fewer», «argument 4 must not be negative», «argument 4 must be a number»).

### Writable views нет: `reshape`/`transpose` — копия, окна — только чтение

`RESHAPE` и `TRANSPOSE` всегда возвращают новый тензор с глубокой копией данных. У результата ровно один владелец — тот, кто его получил; заголовок тензора не выделяется вторым способом (тензор — одно выделение, §6.1), так что «буфер, на который смотрят два записываемых владельца» в системе не возникает.

```lisp
(defvar a (tensor ((1.0 2.0) (3.0 4.0))))
(defvar b (transpose a))
;; b — независимая копия; изменение a не влияет на b
```

Это отличает Qualisp от NumPy, где `a.T` — представление над теми же байтами, и от PyTorch, где `view()` даёт тензор, разделяющий память.

**Окна чтения.** Отдельно от копирующих операций есть две оконные операции: они дают **read-only алиас** над живым тензором.

```lisp
(T-ROWS t from len)    ;; окно по строкам: [V] / [V,D] → [len] / [len,D]
(T-SLICE t from len)   ;; непрерывное окно из плоского индекса: всегда [len]
```

- `T-ROWS` индексирует **первую ось**: на тензоре `[3,2]` `(T-ROWS t 1 2)` даёт форму `[2,2]` — строки 1 и 2.
- `T-SLICE` берёт `len` элементов подряд с **плоского** номера `from`: на `[2,3]` `(T-SLICE t 2 3)` — элементы 3, 4, 5, форма `[3]`.
- Копий при создании нет; `TYPE-OF` окна — `TENSOR`.

Окно — алиас над **любым живым владельцем**, stable или scratch. Владелец держится счётчиком: stable-ячейка — `stable_ref`, а скретч-ячейка при создании окна повышается в stable. Поэтому окно читается и после выхода владельца из области, и после переписи имени, а `TSET!` в источник виден через окно.

```lisp
(defvar t (tensor ((1.0 2.0) (3.0 4.0))))
(let ((w (T-ROWS t 0 1)))
  (TSET! t 0 9.0)
  (tensor->list w))            ;; → (9.0 2.0) — окно видит запись
```

Запись **через окно** отказывает: `TSET!`, `T+=`, `T*=`, `T-RELU`, `T-SCATTER-ADD` и шаги оптимизаторов (`SGD-STEP`, `ADAM-STEP`, `ADAM-SPARSE-STEP`) дают ошибку `destination is a read-only view`. Оптимизатор не должен молча менять веса через read-only окно.

Окно материализуется в обычную копию в трёх случаях: при escape в stable (`defvar`, `setq` глобала — и такая копия **writable**), при взятии окна от окна (промежуточная ячейка — скретч), и при клонировании окна. Поэтому `(defvar v (T-ROWS t 0 1))` на верхнем уровне даёт независимую копию: писать в неё можно, источник не меняется.

```lisp
(defvar v (T-ROWS t 0 1))      ;; копия
(TSET! v 0 9.0)                ;; меняет только v
(tensor-item t 0)              ;; → 1.0, источник цел
```

> 🐍 В NumPy срез — записываемое представление: `b = a[0]; b[0] = 9` меняет `a`. В Qualisp окно только для чтения, а escape в переменную даёт независимую копию.

Алиасинг значений возникает ещё и **при присваивании**: для стабильных тензоров `(setq a b)` **делит** запись (шарит), не клонируя. Оба имени указывают на один тензор, и запись через `TSET!` видна под обоими. Это явное событие, а не побочный эффект `reshape`/`transpose`.

### Выборка строк: gather и scatter

```lisp
(T-GATHER-ROWS table ids)   ;; копия строк по индексам: [V]/[V,D] → [N]/[N,D]
(T-SCATTER-ADD dst ids src) ;; dst[ids[i],:] += src[i,:]
```

- `ids` — f32-тензор формы `[N]` или список целых; проверки типа и границ названы по операции: `T-GATHER-ROWS: id 5 out of range [0,2)`.
- `T-SCATTER-ADD` — мутирующий и **без ленты**: это backward-примитив gather. Дубликаты индексов складываются; несовпадение размеров — `T-SCATTER-ADD: source has 2 elements, expected 1`. Возвращает `dst`.
- Пустой батч допустим.

Дифференцируемые обёртки живут в stdlib `dl` и описаны в гл. 7: `t-gather!`, `t-rows!`, `t-slice!` (forward — примитив/окно, backward — `T-SCATTER-ADD` в нулевой градиент владельца), `EMB-LOOKUP` (синоним `t-gather!`).

### Конечность значений: `(FINITE? t)`

Единственный надёжный способ проверить, что в тензоре нет NaN или Inf, — примитив битовой проверки:

```lisp
(if (FINITE? w) (print "weights ok") (print "weights contain NaN or Inf"))
```

`FINITE?` смотрит на биты (`Tensor::has_nonfinite`), а не сравнивает числа. Это не педантизм: release-сборка идёт с `-ffast-math`, и при сравнении плавающих NaN ведёт себя непредсказуемо (подробности — в гл. 7). Поэтому проверку конечности в тестах и в рабочем коде делают именно так, а не через `=`/`equal`.

## 6.5 Broadcasting

Как в NumPy: операции растягивают измерения размера 1 (скаляр или вектор-строка к матрице). Реализовано в `Tensor::add/sub/mul/div` через `Tensor::broadcast_shape` (numpy-style, right-aligned, dim=1 broadcasts).

```lisp
(T+ (tensor ((1.0 2.0) (3.0 4.0))) (tensor ((10.0))))   ;; прибавит 10 ко всем
(T+ (randn (list 4 3)) (randn (list 3)))                ;; строка-вектор ко всем строкам
```

Для F16/F32 mixed dtype промежуточные значения приводятся через `promote_dtype` (`src/tensor/tensor.cpp:143`):
- `F16 + F16 → F16`
- `F64 + X → F64`
- всё остальное → `F32`

## 6.6 Редукции

```lisp
(TSUM t)                    ;; сумма всех элементов         t.sum()
(TSUM t (list 0))           ;; сумма по оси 0               t.sum(axis=0)
(TMEAN t)                   ;; среднее                      t.mean()
(ARGMAX t)                  ;; индексы максимумов           t.argmax()
(WEIGHTED-LOOKUP probs v1 v2) ;; взвешенная сумма: ∑ pᵢ·vᵢ (скаляр)
```

`WEIGHTED-LOOKUP` — обычная операция (`prim_weighted_lookup`); для HLO-графа есть отдельный `HLO-WEIGHTED-LOOKUP` (`prim_hlo_weighted_lookup`).

`ARGMAX` возвращает F32 — dtype результата зафиксирован, чтобы нативный путь и интерпретаторный фоллбэк считали одинаково.

## 6.7 Активации и элементы нейросетей

Autograd-операции (с `!`) пишут на ленту; те же операции без `!` — чистые вычисления (гл. 7).

```lisp
(RELU t)          ;; без ! — чистое вычисление
(DROPOUT t 0.5)   ;; без ! — тоже доступен

(SOFTMAX! t)   (SIGMOID! t)   (TANH! t)     ;; с ! — пишут на ленту
(CONV2D! input kernel 1 0)                   ;; input/kernel: [N,C,H,W], stride=1, padding=0
(MAXPOOL2D! input 2 2 2)                     ;; (x kh kw stride): окно 2×2, stride 2
(BATCHNORM! x gamma beta 1e-5)      ;; BN(x, γ, β, ε)
(LAYERNORM! x gamma beta 1e-5)      ;; LN(x, γ, β, ε)
(BATCHNORM! h)  (LAYERNORM! h)               ;; нормализации
(CROSS-ENTROPY! logits labels)
(FFT signal)                                 ;; быстрое преобразование Фурье
```

Softmax считает `exp(x − max)`: NaN при логитах больше 44 не бывает. `SOFTMAX!`, `LAYERNORM!`, `BATCHNORM!` и `SIGMOID!` — не отдельные C++-примитивы, а композиции на Lisp в `dl` (гл. 8): `TMAX`-сдвиг, `EXP!`, `SUM!`, деление. Рядом есть tape-free версии для инференса — `SOFTMAX`, `LAYERNORM`, `T-EXP`, `T-SQRT`, `T-LOG`.

> 🐍 Это `torch.nn.functional.relu / softmax / sigmoid / tanh / conv2d / max_pool2d / dropout / batch_norm / layer_norm / cross_entropy` — но как обычные функции языка, без классов-обёрток.

## 6.8 Производительность: SIMD, BLAS и ядра

- Поэлементные операции (`T+`, `T*`, `T-FMA`, `TSIN`, `RELU`, …) векторизованы через Device-API (`src/gpu/device.cpp`):
  - **AVX2/FMA** на x86_64 — 8×float за инструкцию (256 бит).
  - **NEON** на ARM — 4×float за инструкцию (128 бит).
  - Для F16 — отдельные `_Float16` ядра (`launch_elementwise_add_half` и т.п.); на x86_64 с `-march=native` компилятор генерирует F16C.
- `MATMUL` идёт через OpenBLAS (`cblas_sgemm`, `#ifdef QUALISP_HAS_BLAS`) — производительность уровня NumPy/PyTorch на CPU.
- **Многоядерность живёт внутри операций.** Поэлементные и редукционные циклы идут под OpenMP, матричные произведения — под OpenBLAS: одна тензорная операция занимает все доступные ядра, и это и есть основной путь параллелизма (аргументацию и асинхронный слой для работы вне одной операции — гл. 15.12). Замер ядерного пути: произведения 2048×2048 — 0.99 с в одном потоке против 0.29 с на всех ядрах процесса (занятость процесса 11.94× при 16 ядрах).
- Для F16 в matmul — отдельное ядро `launch_matmul_half` (half хранение, аккумуляция внутри ядра).
- Данные тензоров живут в **TensorBufferPool** (exact-size, гл. 5): буфер выделяется ровно под размер тензора, без округлений; при выходе из scope возвращается в free list по точному размеру и переиспользуется. Никакой фрагментации и аллокаций в горячем цикле.

### Как устроено ядро

- **Редукции**: slab-ядра на raw pointers (`sum_ps`/`max_ps`, OpenMP по хвостовому контуру, одометр без аллокаций на смешанных осях). Замер на sum-4096 — 4.7 мс.
- **Transpose/permute**: 64×64 2D-тайлы держат оба потока cache-resident. Замер на transpose-2048 — 4.8 мс.
- **Softmax**: векторизованный forward (`simd::softmax_row_ps`) + O(D) backward вместо O(D²) якобиана (диагональное тождество `g_i = s_i·(g_i − ⟨g,s⟩)`). Замер на softmax-1024 — 0.40 мс.
- **conv2d**: im2col одним memcpy (одноразовое нулевое окаймление входа), per-image GEMM прямо в слайс результата. Замер на conv2d-n8 — 9.8 мс.

## 6.9 dtype: F16, F32 и другие

Qualisp поддерживает несколько типов данных тензоров. В `src/tensor/tensor.hpp`:

```cpp
enum DType : uint8_t { F32 = 0, F64, I32, I64, U8, F16 };
```

- **F32** — default для литералов (`(tensor ...)` создаёт F32).
- **F16** — IEEE 754 binary16 (`_Float16` через `<cmath>`/`<cstring>` в `src/tensor/half.hpp`); есть kernels `add/mul/relu/matmul/sum` через `launch_*_half` API.
- **F64** — double (promote_dtype для F64+X).
- **I32, I64** — целочисленные тензоры (для argmax, embedding и т.п.).
- **U8** — байты (для квантизации, embedding-таблиц, изображений).

Преобразования: `tensor-dtype` возвращает символ dtype (`F16`/`F32`/...), `tensor-f16`/`tensor-f32`/`tensor-u8` создают тензор нужного dtype. Смешанные F16/F32 промежуточные результаты приводятся через `Tensor::promote_dtype`.

Сериализация: поддерживается NPY `<f2` (raw-байтовый доступ через `half_bits`/`half_from_bits` в `half.hpp`).

## 6.10 Сохранение/загрузка: NPY

Взаимодействие с NumPy — через `LOAD-NPY`/`SAVE-NPY` (примитивы в `src/data/`):

```lisp
(save-npy t "weights.npy")    ;; сохранить в .npy
(load-npy "weights.npy")      ;; загрузить обратно
```

F16 сериализуется как `<f2` (raw 2 байта на элемент). Эти примитивы позволяют интегрироваться с Python-экосистемой без потерь.

## 6.11 Полная картина тензорных операций

```text
            ┌────────────────────────────────────────┐
            │  Tensor: dtype, shape, strides, numel │
            └────────────────────────────────────────┘
               │      │       │        │          │
    create/save  copy   pointwise reduce     matmul/conv/pool
         │       │       │        │          │
         ▼       ▼       ▼        ▼          ▼
    TensorBufferPool   SIMD kernels    cblas_sgemm / F16 ядра
    (exact-size)       (AVX2/FMA/NEON)  (BLAS / half)
```

Слои операций:

| Слой | Операции |
|---|---|
| Создание | `TENSOR`, `ZEROS`, `ONES`, `RANDN`, `RANDINT`, `EYE`, `LIST->TENSOR`, `TENSOR-F16/F32/U8` |
| Копия/форма | `RESHAPE`, `TRANSPOSE`, `CONCAT`, `STACK`, `CLONE` |
| Окна (read-only) | `T-ROWS`, `T-SLICE`, строковые `STRING-VIEW`/`STRING-REF` |
| Поэлементно | `T+`, `T-`, `T*`, `T/`, `T-ABS`, `TSIN`, `TCOS`, `T/TANH`, `T-EXP`, `T-SQRT`, `T-LOG`, `T-POW`, `T-FMA`, `T-FMA-RELU`, `RELU` |
| In-place | `T+=`, `T*=`, `T-RELU`, `TSET!` |
| Редукции | `TSUM`, `TMAX`, `TMIN`, `TMEAN`, `TARGSORT`, `ARGMAX` |
| Матричные | `MATMUL`, `CONV2D`, `MAXPOOL2D`, `BATCHNORM`, `LAYERNORM`, `FFT` |
| Выборка | `T-GATHER-ROWS`, `T-SCATTER-ADD`, `LOOKUP`, `WEIGHTED-LOOKUP` |
| Дифференцируемые (`!` и stdlib) | `T+!`, `T-!`, `T*!`, `T/!`, `MATMUL!`, `RELU!`, `EXP!`, `SUM!`, `MEAN!`, `TRANSPOSE!`, `RESHAPE!`, `T-GATHER!`, `T-ROWS!`, `T-SLICE!`, `SIGMOID!`, `TANH!`, `SOFTMAX!`, `LAYERNORM!`, `BATCHNORM!`, `DROPOUT!`, `CROSS-ENTROPY!`, `MSE!` |

## 6.12 Известные ограничения

- Поле `Tensor::size_bytes` **не реализовано** (есть только `numel * dtype_size(dtype)` через инлайн-вычисление).
- Поле `Tensor::is_stable` **не реализовано** — пока всё управляется через `Scope::track_external`.
- **Окна read-only.** `T-ROWS`/`T-SLICE` дают алиас только для чтения: писать в окно нельзя, а окно над trainable-параметром не даёт обучать его строки. Для обучения строк таблицы есть `EMB-LOOKUP`/`T-GATHER!` + `ADAM-SPARSE-STEP` (гл. 7, 8).
- **`LIST->TENSOR` и форма.** Список списков — это строки: `(list (list 1.0 2.0))` (одна строка) даёт форму `[2]`, две строки — `[2,2]`. Форму лучше задавать явно через `RESHAPE`.
- В стенсиль HLO попадает только подмножество dtype — **F32 и I32**; F16 уходит в интерпретаторный фоллбэк (в стенсиле f16 портил данные без всякой ошибки).
- `TANH` для тензоров — `T/TANH` (примитив `prim_tensor_tanh`), не `TANH` (тот зарезервирован для autograd-варианта `TANH!`).
- `launch_*` в Device-API покрывают float и half; double и int проходят через `get_float`/`set_float` — с потерей точности для целых значений выше 2²⁴.
