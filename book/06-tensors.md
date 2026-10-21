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
    void *data;             // тензорный буфер в TensorBufferPool
    Device *device;         // абстракция устройства; реализация одна — CPU

    bool requires_grad;
    Tensor *grad;
    GradNode *grad_node;    // связь с autograd-лентой
    Tensor *m, *v;          // Adam state (хранится прямо на тензоре)
    int64_t step;
};
```

Поддерживаемые dtype: **F16, F32, F64, I32, I64, U8**. Размеры: `F16=2, F32=4, F64=8, I32=4, I64=8, U8=1` байт (`Tensor::dtype_size`).

Промежуточные тензоры живут в scratch-scope, в `TensorBufferPool` — exact-size аллокация (см. `Tensor::create`: `t->device->alloc_pooled(numel * dtype_size(dt))` с `arena.track_external`).

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

### Нет views: `reshape` и `transpose` — глубокая копия

У тензоров в Qualisp **нет** views и alias'ов: `RESHAPE` и `TRANSPOSE` всегда возвращают новый тензор с глубокой копией данных. У результата ровно один владелец — тот, кто его получил; второй путь выделения заголовка тензора удалён, так что «буфер, на который смотрят два владельца», в системе не возникает.

```lisp
(defvar a (tensor ((1.0 2.0) (3.0 4.0))))
(defvar b (transpose a))
;; b — независимая копия; изменение a не влияет на b
```

Это отличает Qualisp от NumPy, где `t.T` — представление над теми же байтами, и от PyTorch, где `view()` даёт тензор, разделяющий память.

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

Softmax считает `exp(x − max)`: NaN при логитах больше 44 не бывает.

> 🐍 Это `torch.nn.functional.relu / softmax / sigmoid / tanh / conv2d / max_pool2d / dropout / batch_norm / layer_norm / cross_entropy` — но как обычные функции языка, без классов-обёрток.

## 6.8 Производительность: SIMD и BLAS

- Поэлементные операции (`T+`, `T*`, `T-FMA`, `TSIN`, `RELU`, …) векторизованы через Device-API (`src/gpu/device.cpp`):
  - **AVX2/FMA** на x86_64 — 8×float за инструкцию (256 бит).
  - **NEON** на ARM — 4×float за инструкцию (128 бит).
  - Для F16 — отдельные `_Float16` ядра (`launch_elementwise_add_half` и т.п.); на x86_64 с `-march=native` компилятор генерирует F16C.
- `MATMUL` идёт через OpenBLAS (`cblas_sgemm`, `#ifdef QUALISP_HAS_BLAS`) — производительность уровня NumPy/PyTorch на CPU.
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

## 6.12 Известные ограничения

- Поле `Tensor::size_bytes` **не реализовано** (есть только `numel * dtype_size(dtype)` через инлайн-вычисление).
- Поле `Tensor::is_stable` **не реализовано** — пока всё управляется через `Scope::track_external`.
- **Срезов тензора нет**: примитива `T-SLICE` не существует, достучаться до части тензора можно только через переиндексацию средствами модулей.
- В стенсиль HLO попадает только подмножество dtype — **F32 и I32**; F16 уходит в интерпретаторный фоллбэк (в стенсиле f16 портил данные без всякой ошибки).
- `TANH` для тензоров — `T/TANH` (примитив `prim_tensor_tanh`), не `TANH` (тот зарезервирован для autograd-варианта `TANH!`).
- `launch_*` в Device-API покрывают float и half; double и int проходят через `get_float`/`set_float` — с потерей точности для целых значений выше 2²⁴.
