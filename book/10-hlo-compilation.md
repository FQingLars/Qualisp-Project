# Глава 10. HLO-компиляция: от интерпретации к машинному коду

HLO-пайплайн — killer feature Qualisp: ваш тензорный код трассируется в граф, оптимизируется и компилируется в **нативный машинный код** собственным copy-and-patch стенсил-эмиттером — без LLVM, без промежуточного IR и без дискового кэша. Это собственная реализация идей XLA/torch.compile.

> 📌 **Пайплайн v2.7.0** (срез H3): единый эмиттер `compile_hlo_stencil` — один проход по графу, байтовые AVX2-стенсилы с «дырками» патчатся в W^X-страницу процесса. Прежний путь «LLVM IR → объектник → lldELF → .so → dlopen» полностью удалён вместе с дисковым кэшем.

## 10.1 Идея

```text
Код Qualisp → трассировка в HLO-граф (START-TRACE … HLO-COMPILE)
           → оптимизации (CSE, DCE, fusion, broadcast-shape, COMPOSITE-expansion)
           → compile_hlo_stencil: один проход по графу
           → байтовые AVX2-стенсилы с «дырками» → патч → W^X-страница процесса
           → нативное исполнение
```

С **v2.7.0** у HLO один эмиттер — `compile_hlo_stencil`. Он идёт по графу один раз и выписывает готовые x86-64 (AVX2) последовательности байт, оставляя «дырки» — imm-формы, rel32-мишени, movabs-адреса хелперов, — которые тут же патчатся, и копирует результат в W^X-страницу процесса. Сигнатура кернела: `void fn(float** params, float* out)`; узел `DOT` превращается в `call cblas_sgemm` (адрес прошивается movabs-патчем). Промежуточного IR нет вовсе — тот же принцип «инварианта одной формы», что и у JIT замыканий (гл. 20), только стенсилы тензорные.

- **CSE** — устранение общих подвыражений (`HloGraph::eliminate_common_subexpressions`).
- **DCE** — удаление мёртвого кода (`HloGraph::eliminate_dead_code`).
- **Fusion** — цепочки поэлементных операций (ADD→MUL→RELU) сливаются в **один кирнел** (`HloGraph::fuse_elementwise_ops`).
- **COMPOSITE-expansion** — `defhloop`-узлы разворачиваются в mini-граф (`composite_expanded` флаг защищает от двойного разворачивания).
- **Broadcast-shape** — `add`/`mul` вычисляют numpy-style broadcast shape (right-aligned, dim=1 broadcasts), а не форму левого операнда.
- **BLAS** — узлы `DOT` (матричное умножение) вызывают `cblas_sgemm`: адрес функции прошивается в стенсил movabs-патчем.
- **Кэш** — in-process, по fingerprint (версия эмиттера + shape-ключ): повторные компиляции одного графа в рамках процесса не перекомпилируются. С v2.3.6 fingerprint маршрута включает branch-подграфы и метки — графы с одинаковым плоским скелетом, но разными ветвями, не делят запись кэша. Дисковый кэш удалён вместе с LLVM-путём (v2.7.0).
- **W^X** — страницы выделяет `platform::page_alloc` / `page_protect_exec`; страницы живут до конца процесса и после установки не изменяются.

> 🐍 **Python-аналогия.** `torch.compile(model)` / `jax.jit(f)` — те же три этапа: трассировка, оптимизация, кодоген. Разница: в Qualisp это встроено в язык, кодоген — собственные стенсилы без IR (как у CPython 3.13 JIT, только для тензорных кернелов и с AVX2-векторизацией), и всё работает на S-выражениях.

## 10.2 Граф и его узлы

Из `src/hlo/graph.hpp`:

```cpp
enum class HloOp : uint8_t {
    PARAMETER, CONSTANT, DOT, ADD, MUL, RELU, FUSION, TUPLE,
    OUTPUT, LOOKUP, ARGMAX, SOFTMAX, COMPOSITE, CODEGEN,
    WEIGHTED_LOOKUP,
    ROUTE          // v2.3.5: нейросимвольная развилка внутри графа
};

struct HloNode {
    int id;
    HloOp op;
    std::vector<size_t> shape;
    std::vector<HloNode*> inputs;
    Tensor *constant_value;   // v2.8.6: клон в хранилище трейса, а не указатель
                              // на тензор области записи (граф переживает область)
    std::vector<HloOp> fused_ops;     // FUSION: последовательность операций
    std::vector<float> lookup_values; // LOOKUP / WEIGHTED_LOOKUP
    std::string composite_name;       // COMPOSITE: имя defhloop
    std::string codegen_kind;         // CODEGEN: kind ядра
    bool composite_expanded;
};
```

`HloGraph` владеет всеми узлами, поддерживает `entry()`, `dump()`, `clone()` (глубокое копирование для fallback-интерпретации), `fingerprint()` (для кэша), `optimize()` (CSE+DCE+fusion+DCE). У `ROUTE`-ноды (v2.3.5) внутри живут branch-подграфы, OR-overlap-метки и per-sample taken; clone/merge_subgraph переведены на pointer-identity maps, чтобы кросс-графовые ссылки на ветви переносились корректно, а DCE держит branch-ноды живыми и CSE их не подменяет и не сливает ROUTE-ноды между собой.

Результат компиляции — программа с тегом `HLOPROG`: если граф прошёл через стенсил-эмиттер, она несёт нативный указатель с флагом `is_stencil`; если граф вне компилируемого подмножества (см. §10.12) — fallback-программу, которую исполняет интерпретатор графа.

## 10.3 Ручная трассировка: базовый workflow

Примитивы (`src/eval/interpreter.cpp::register_primitives`):

```lisp
;; 1. Включить трассировку
(START-TRACE)

;; 2. Пометить входы как параметры графа (порядок = порядок аргументов HLO-RUN)
(defvar dx (GRAPH-PARAM x))
(defvar dw (GRAPH-PARAM w))

;; 3. Построить вычисление обычными операциями (без !)
(defvar out (T+ (MATMUL dx dw) b))

;; 4. Посмотреть граф (опционально)
(DUMP-GRAPH)

;; 5. Скомпилировать (возвращает HLOPROG — тег SExpr)
(defvar hlo (HLO-COMPILE out))

;; 6. Выполнить нативно: параметры передаются в порядке определения GRAPH-PARAM
(defvar result (HLO-RUN hlo x w b))
```

Интроспекция скомпилированной программы (v2.7.0):

```lisp
(HLO-STENCIL-P hlo)   ;; → T: кернел построен стенсил-эмиттером (нативная страница)
(HLO-HAS-RANGE hlo)   ;; → T: у поэлементного графа есть range-вход (§10.6)
```

> 🐍 Аналог:
> ```python
> @torch.compile
> def f(x, w, b): return x @ w + b
> f(x, w, b)
> ```
> В Qualisp трассировка явная: `GRAPH-PARAM` ≈ декларация сигнатуры графа, `HLO-RUN` ≈ вызов скомпилированной функции. Конвенция: **порядок вызовов `GRAPH-PARAM` = порядок аргументов `HLO-RUN`** (см. `TEMP_ISSUES.md` #13).

## 10.4 Два режима: трассировка и интерпретация

- Внутри `(START-TRACE)` … `(STOP-TRACE)` операции строят граф вместо вычисления.
- Вне трассировки те же операции считаются интерпретатором (с SIMD/BLAS).
- `(HLO-RUN hlo ...)` выполняет скомпилированный граф; если граф не прошёл стенсил-компиляцию (вне подмножества) — автоматический fallback на интерпретацию клонированного графа (`HloGraph::clone()` глубокое копирование — см. `TEMP_ISSUES.md` #23). С v2.3.7 `HLO-RUN` защищён arg-count guard: на нехватке аргументов — ошибка (fallback считает PARAMETER-ноды, native сверяет `hp->num_params`; лишние аргументы разрешены), а не SIGSEGV.

Готовые HLO-примитивы (для построения графов без тензорных значений):

```lisp
(HLO-SOFTMAX ...)             ;; softmax-узел
(HLO-ARGMAX ...)              ;; argmax-узел
(HLO-LOOKUP ...)              ;; embedding lookup
(HLO-WEIGHTED-LOOKUP ...)     ;; взвешенная сумма по embedding'ам
(HLO-TRAIN-SGD ...)           ;; обучающий шаг прямо в скомпилированном графе
```

С v2.7.1 (срез H4b) эти узлы больше не отправляются в фоллбэк: `SOFTMAX`, `ARGMAX`, `LOOKUP`, `WEIGHTED_LOOKUP` и CODEGEN-виды (`tanh`, `fabs`, `fneg`, `clamp01`, `copy`) компилируются в **нативные стенсилы** той же W^X-страницы — проверено: граф с `HLO-SOFTMAX` даёт `(HLO-STENCIL-P h)` → `T` и верные вероятности. Фоллбэк остаётся для графов с `ROUTE`, `COMPOSITE`/`TUPLE` и несовместимыми формами узлов.

## 10.5 `defun` vs `defuse` vs `defuse!`

Три способа объявить функцию различаются тем, должен ли компилятор слить её тело в fused-кирнел:

| Форма | Fusion? | Возврат обязателен? | Если не слилось |
|---|---|---|---|
| `defun` | никогда | нет (можно процедуру) | — |
| `defuse` | пытается | да | молча становится обычной функцией |
| `defuse!` | принудительно | да | **ошибка компиляции** |

```lisp
(defun plain-proc (x) (print x))          ;; процедура: ради побочного эффекта

(defuse soft-fn (x) (T* x x))             ;; попробует фьюзить

(defuse! hard-fn (x) (T+ x (T* x x)))     ;; обязан: x + x*x одним кирнелом
```

> 🐍 `defuse!` ≈ `@torch.compile(fullgraph=True)` — «либо скомпилируй целиком, либо падай»; `defuse` ≈ обычный `@torch.compile` с тихим fallback.

При трассировке вызов `defuse`-функции вставляет в граф **COMPOSITE-узел** — тело функции целиком становится одним кирнелом (через `prim_hlo_compile → COMPOSITE-expansion`).

## 10.6 Range-путь HLO (многопоточный)

Для поэлементных графов над параметрами стенсил-эмиттер строит **второй вход в той же странице** — range-кернел (срез H2, v2.6.2):

```cpp
void fn_range(float** params, float* out, long start, long end)
```

Исполнение — `hlo_range_exec`: OpenMP-параллелизм, чанки кратны 8 (ширина AVX2-вектора), guarded хвост покрывает остаток. Range-путь включается автоматически при `out_numel >= 1 МиБ` — на меньших объёмах накладные расходы на многопоточный вызов не окупаются.

Наличие range-входа проверяется интроспекцией: `(HLO-HAS-RANGE h)` → `T`.

## 10.7 ROUTE: нейросимвольная маршрутизация в графе (v2.3.5–2.3.6)

Развилка `NS-IF` перестала быть внешним слоем диспетчеризации — она стала узлом вычислений.

### `HloOp::ROUTE` (коммит `262f62d`)

- **Forward**: decision logits (inputs[0]) выбирают одну из принадлежащих узлу branch-подграфов на каждый вызов; ветви, OR-overlap-метки и per-sample taken живут внутри узла; `resolve_correct` зеркалит `src/ns/route.hpp` (взятая ветка побеждает).
- **NS-IF под трассировкой строит ROUTE-ноду**: каждая ветвь тела трассируется в собственный мини-граф (COMPOSITE-паттерн), константные результаты ветвей замораживаются как CONSTANT. Tape-путь и обучение через `NS-GRAD!` не изменились.
- **Интерпретаторный fallback исполняет ROUTE** (`hlo_fallback_node`): per-sample argmax выбирает ветвь (taken[0] для смешанного батча — та же семантика forward, что у tape-роутера), вложенный ROUTE рекурсирует. Графы с ROUTE компилируются сразу в fallback-программу (диспетчеризация ROUTE стенсилами — в плане, `TODO.md`).

### `HLO-ROUTE-GRAD!` — backward внутри узла (коммит `ff6685a`)

- Softmax-weighted STE сидится в decision_logits, всё состояние — внутри ROUTE-ноды (plain value vectors, без указателей). Forward fallback-исполнитель кэширует значения logits.
- Примитив **`HLO-ROUTE-GRAD!`** обходит исполненные ROUTE-ноды в pre-order и сажает сид первой разошедшейся ноде на сэмпл (label-transparent ноды пропускаются, полностью правильные маршруты сида не получают). Возвращает `[batch, N]`-градиенты — по одному тензору на ROUTE-ноду; один вызов = один backward.
- **Обучение без ленты вовсе**: `dW = featsᵀ·g`, `W -= lr·dW` — тест `hlo-route-grad-train-loop` сходится 4/4 за ≤400 шагов.

```lisp
;; обучаемый ROUTE-роутер внутри скомпилированного графа
(defvar grads (hlo-route-grad! prog labels))   ;; [batch, N] на каждую ROUTE-ноду
```

## 10.8 Граф как данные (v2.3.7)

Гомоиконность второго уровня доведена до HLO-графа: граф строится, читается и оптимизируется из языка.

```lisp
;; граф в S-выражении (данные, не указатели):
;; ноды с id/op/inputs/атрибутами, ROUTE с ветвями-подграфами и
;; OR-overlap-метками, CONSTANTS с полными значениями тензоров
(defvar data (graph-data prog))

;; обратное построение — без трассировки, включая ROUTE-развилки и константы
(defvar prog2 (graph-from-data data))

;; оптимизационные проходы вызываются из Lisp
(defvar prog3 (graph-run-passes prog2 '("cse" "dce" "fusion")))
```

- Круговой рейс (print → read → print) точен; QSRD v2 хранит блобы структурно (гл. 18.8).
- COMPOSITE-ноды ездят через `GRAPH-DATA`/`GRAPH-FROM-DATA` и через `GRAPH-RUN-PASSES`; fallback-исполнитель материализует подграф COMPOSITE при запуске.
- Именованные константы переживают проходы (id-переходы корректны).
- **Запись идёт в топологическом порядке, полез обёртки FUSION (v2.8.6).** Порядок `g->nodes()` после оптимизации **не** топологический: проход fusion дописывает объединённую ноду в конец списка, тогда как ранние ноды уже на неё ссылаются (DOT 5 → FUSION 11), и `graph_build_node` спотыкался о ненущенный вход («unknown input id 11»). Эмиссия идёт post-order DFS (зависимости раньше), scope ограничен так, что ссылки ветви на объемлющий граф разрешаются в проходе родителя. Payload FUSION больше не обёрнут лишним cons'ом — `((ADD RELU))` читалось как один символ и роняло `graph_op_of` с `std::bad_alloc`.
- Тесты: `tests/hlo_graph_data.qlsp` (в т.ч. `gd-fusion-round-trip`), `tests/image.qlsp`, `tests/production_training.qlsp`; закрывает пункт TODO «(print-graph) печатает граф в SExpr-виде и обратно».

> 🐍 Это `torch.fx.Graph`/`graph_module.print_readable()` + pickle графа — только данные нативные S-выражения, и всё оптимизируется из самого языка.

## 10.9 Компиляция обученной модели: полный пример

Из `examples/linear_regression.qlsp`:

```lisp
;; ... обучили w и b автоградиентом (гл. 7) ...

;; Компилируем forward для инференса
(START-TRACE)
(defvar dx (GRAPH-PARAM x-test))
(defvar dw (GRAPH-PARAM w))
(defvar db (GRAPH-PARAM b))
(defvar dp (T+ (MATMUL dx dw) db))

(defvar hlo (HLO-COMPILE dp))
(defvar hlo-result (HLO-RUN hlo x-test w b))   ;; нативный инференс
```

Разбор целиком — в главе 17.

## 10.10 Когда использовать HLO

- **Инференс обученной модели** — главный сценарий: граф фиксирован, fusion даёт максимум.
- **Горячие циклы обучения** — через `HLO-TRAIN-SGD` и `defuse!`-функции.
- Прототипирование — оставайтесь в интерпретаторе; компилируйте, когда алгоритм устоялся.

Практика: сначала `(defuse soft ...)`, и повышайте до `defuse!`, когда убедились, что тело трассируемое (только тензорные операции, без `print`, `if` по значениям тензоров и т.п.).

## 10.11 Известные ограничения (TODO)

- Произвольное переписывание графа из языка (`graph-node`/`graph-rewrite`) — в плане (`TODO.md` §1.2); база уже есть: `GRAPH-DATA`/`GRAPH-FROM-DATA` (2.3.7) и `GRAPH-RUN-PASSES` (CSE/DCE/Fusion из Lisp).
- Диспетчеризация ROUTE стенсилами — в плане; графы с ROUTE исполняются fallback-программой.
- Стенсилы пока только для AVX2 x86-64 (SysV): иных целевых платформ у эмиттера нет.
- `composite_expanded` флаг защищает от двойного разворачивания; identity-COMPOSITE (тело = сам параметр) коллапсируется в PARAM.
- Fingerprint in-process-кэша использует `std::hash` — теоретические коллизии, <10⁻⁹ вероятность (`TEMP_ISSUES.md` #22); с 2.3.6 fingerprint включает branch-подграфы и метки.
- Тесты: `tests/hlo_stencils.qlsp`, `tests/hlo_stencil_range.qlsp`, `tests/hlo_stencil_ops.qlsp` (нативные кернелы H4b, включая батч-регрессию WEIGHTED_LOOKUP), прежние `hlo_*.qlsp`.