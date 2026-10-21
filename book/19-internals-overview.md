# Глава 19. Устройство компилятора

Краткая экскурсия во внутренности Qualisp: как исходник превращается в программу, какие
подсистемы за этим стоят и как они делят ответственность. Полное описание — в
`ARCHITECTURE.md` дерева разработки.

## 19.1 Пайплайн

```text
Исходник → CharStream → Readtable (reader) → SExpr-дерево в ConsCellPool (гл. 5)
  → MacroExpander (backquote, defmacro, compiler-macros, MACROLET)
  → TypeChecker (постепенная типизация)
  → Interpreter::eval (special forms, применение функций, примитивы)
    → Tensor-операции → GradientTape (запись autograd)
    → GradientTape::backward (reverse-topo AD, blocked nodes для NS-IF)
    → горячее замыкание → jit::compile_closure → copy-and-patch → RX-страница (гл. 20)
  → HloProgram (форма графа — мастер) → program_lowered → HloGraph (одноразовый кэш)
    → compile_hlo_stencil → AVX2-стенсилы → W^X-страница (гл. 10)
    → NS-router (NS-IF/NS-GRAD! → per-sample routing + guilt detector; ROUTE-нода в HLO)
                     ↓
                LSP Server ←→ Editor
```

**Инвариант одной формы.** S-выражение — единственная авторитетная форма программы.
Оба «компилятора» — JIT замыканий и HLO-стенсил-эмиттер — потребляют S-выражения и
строят нативные страницы одним проходом, без промежуточного IR и без LLVM. Нативный код —
выбрасываемый кэш: `DISCARD-JIT` и инвалидация по эпохе макросов пересобирают его из
master-копий без изменения поведения.

То же правило действует и внутри HLO: мастер-копия программы — **форма графа**
(`HloProgram::graph_form`), а `HloGraph` — производный **одноразовый кэш понижения**
(`program_lowered`), который можно выбросить и перестроить; результат совпадает.

Отдельного AOT-компилятора нет: всё нативное собирается на лету из S-выражений.

## 19.2 Структура исходников

```
src/
├── core/          Scope (HibLin), SExpr, Symbol, ConsCellPool, TensorBufferPool, StableMemory,
│                  simd.hpp (slab-ядра), trace.{hpp,cpp} (хронология владения), image.{hpp,cpp} (образы, гл. 21)
├── reader/        CharStream, Readtable
├── eval/          Interpreter (80+ примитивов), специальные формы, символьный движок
├── types/         Type, TypeChecker (постепенная типизация)
├── macros/        MacroExpander, backquote
├── mop/           ClassObj, ClassRegistry, Instance (CLOS)
├── tensor/        Tensor, GradientTape (F16/F32/F64/I32/I64/U8)
├── gpu/           Device (абстракция устройства; реализация одна — CPU)
├── hlo/           HloGraph, HloNode, HloOp (вкл. ROUTE), оптимизации, fingerprint
├── jit/           stencils.hpp, copypatch.{hpp,cpp} — JIT замыканий (copy-and-patch, RX-страницы)
├── codegen/       HloCodeGen: единый стенсил-эмиттер compile_hlo_stencil (W^X-страницы, гл. 10)
├── symbolic/      S1-S8 (match, unify, plist, defrule, amb, compiler-macros, defhloop)
├── ns/            RouteHop, Router, GuiltDetector (S9)
├── data/          foundation, qserde (.qsrd-сериализация)
├── ffi/           FFI-LOAD/CALL/IMPORT (dlopen + dlsym)
├── visual/        Dashboard (HTTP, TensorBoard-подобный)
├── bindings/      bind.hpp, stb_image.cpp (нативные хелперы биндингов)
├── qvalent/       qvalent CLI (пакетный менеджер)
├── lsp/           server.cpp (только LSP, без движка)
├── stdlib/        *.qlsp → embedded_modules[] (через xxd -i)
└── main.cpp       REPL/интерпретатор (JIT встроен — отдельного компилятора нет)
```

Модули stdlib вшиты в бинарник (`cmake/stdlib_embed.cmake`), поэтому изменения stdlib
требуют пересборки.

## 19.3 Reader

Символьная диспетчеризация (`src/reader/readtable.cpp`):

| Символ | Действие |
|---|---|
| `(` | разбор списка (правильного или dotted) |
| `)` | error |
| `'` | разворачивается в `(QUOTE ...)` |
| `` ` `` / `,` / `,@` | `(BACKQUOTE ...)`, `(COMMA ...)`, `(COMMA-AT ...)` |
| `#` | dispatch: `#'` → FUNCTION, `#(` → вектор, `#\` → литера |
| `"` | строка с escape-последовательностями |
| `;` | комментарий до конца строки |

Токены: символы интернируются в верхнем регистре через `SymbolTable::instance()`
(регистронезависимость); сначала пробуется парсинг числа (int64 → double), иначе — символ;
`nil` → синглтон nil, `t` → символ T. Правило читателя — единственное: `INTERN` приводит
имя к тому же виду, каким его прочитал бы ридер, так что `(intern "print")` — тот же
символ, что `PRINT`.

> 🐍 Аналог — фаза tokenizer/parser в CPython, но результат — не `ast.AST`, а cons-ячейки: код и данные одного представления.

## 19.4 S-выражения

Тегированное объединение (`SExpr`, `src/core/sexpr.hpp`):

```cpp
class SExpr {
    Tag tag_;       // NIL, CONS, SYMBOL, FIXNUM, FLONUM, STRING,
                    // PRIMITIVE, CLOSURE, INSTANCE, TENSOR, HLOPROG, JITPROG
    union {         // tagged payload
        struct { SExpr *car_, *cdr_; } cons_;
        const Symbol *sym_;
        int64_t fixnum_;
        double flonum_;
        struct { const char *data_; size_t len_; } string_;
        Tensor *tensor_;
        struct { SExpr *params_, *body_, *env_; } closure_;
        // HLOPROG: указатель на W^X-страницу (is_stencil) либо fallback-программа
        // JITPROG: нативная RX-страница + master-замыкание + эпоха + счётчик вызовов
        struct { void *fn_ptr; bool is_stencil; SExpr *closure_; uint64_t epoch_, calls_; } native_;
    };
};
```

Все cons-ячейки выделяются в `ConsCellPool`; объекты — в `Scope` с деструктор-трекингом.

Замыкание: параметры + тело + захваченное окружение. Глобальные `defun`/`defmacro`/`defrule`
сохраняются в `root_arena_` (stable) — код-как-данные живёт всю программу (гомоиконность
рантайма).

## 19.5 Окружение и вычисление

```cpp
struct EnvFrame {
    const EnvFrame *parent;   // лексический родитель (nullptr для global)
    SExpr *bindings;          // alist: ((sym . val) (sym2 . val2) ...)
};
```

`bindings` — это SExpr-alist, а не отдельная структура: окружение выражено тем же
представлением, что и всё остальное. Переработка окружения (self-hosting) — направление
развития, а не текущее состояние.

Вычисление: атом → сам; символ → поиск переменной; cons → special form или вызов функции.
Замыкания захватывают цепочку окружений в плоский alist.

**Live-глобалы.** Замыкание не захватывает глобальный кадр: глобалы резолвятся в момент
вызова, как в Common Lisp, — переопределённый `defun` виден живому замыканию. Захватываются
только локальные кадры (`let`, `let*`, тела циклов), каждый со своей ссылкой на пару.
Побочный эффект, о котором стоит знать прямо: **лисп-обёртка с именем примитива затеняет
сам примитив** (символы регистронезависимы) — `(defun sample (logits) (SAMPLE logits))`
вызовет сам себя. В stdlib тела таких обёрток переведены на другие имена (`sleep` →
`WAIT-MS`, `string-contains` → `STRING-INDEX`).

## 19.6 Special forms

Из `sym_*` в `src/eval/interpreter.cpp`:

`QUOTE, IF, PROGN, LAMBDA(\), LET, LET*, DEFUN, DEFUSE, DEFUSE!, DEFMACRO, MACROLET, DEFVAR, SETQ, COND, WHILE, AND, OR, BACKQUOTE, IMPORT, THE, GENSYM, MACROEXPAND, MACROEXPAND-1, FUNCALL, APPLY, DEFCLASS, MAKE-INSTANCE, SLOT-VALUE, DEFSTRUCT, DEFGENERIC, DEFMETHOD, MATCH, UNIFY, SUBST, AMB-LET, REQUIRE, AMB-COLLECT, ASSERT-FACT, RETRACT-FACT, DEFRULE, QUERY, RUN-RULES, DEFINE-COMPILER-MACRO, DEFHLOOP, NS-IF, NS-GRAD!, TENSOR` — и **80+ примитивов** (арифметика, тензоры, autograd, HLO, FFI, IO, data).

## 19.7 Тензорная подсистема

```text
Tensor: dtype (F16/F32/F64/I32/I64/U8), shape, strides, numel,
        data (TensorBufferPool — exact-size, 0% перерасхода),
        requires_grad, grad, grad_node (лента), m/v/step (Adam)
```

- Поэлементные операции — SIMD: AVX2 (8×f32) на x86_64, NEON (4×f32) на ARM. Ядра
  работают по сырым указателям (`src/core/simd.hpp`: `sum_ps`/`max_ps`/`scale_ps`/`exp_ps`
  с полиномом AVX2), диспетчеризация идёт по layout, одометр работает без аллокаций.
- `MATMUL` — `cblas_sgemm` (OpenBLAS): транспонирования выполняются внутри GEMM
  (флаги `atrans`/`btrans`), backward-логи и шаги SGD/ADAM работают на сырых указателях.
- `reshape`/`transpose` — глубокие копии (view нет, гл. 4).
- `TensorBufferPool`: exact-size free list + кэш по (shape, dtype) — повторяющиеся
  размеры (батчи, активации) переиспользуются за O(1). Пул ограничен по объёму: при
  переполнении лишние буферы уходят системе, а не удерживают RSS за мёртвыми данными.
- F16: `launch_*_half` через `_Float16` (F16C на x86_64); matmul аккумулирует во float.
- `while`: каждая итерация живёт в собственном scope (`checkpoint()`/`restore()`),
  результат тела — в паре ping-pong scratch-скоупов, в root продвигается один раз на
  выходе. Tape-графы, не закрытые `GRAD!`, сбрасываются на границе формы — незакрытые
  графы не копятся в памяти формы.
- Утечки памяти в ядре ловит soak-харнесс (`tests/memory_soak.qlsp` + `bench/leak_watch.py`,
  VmRSS по `/proc`).

## 19.8 Autograd

Узел ленты (`src/tensor/gradient_tape.hpp`):

```cpp
struct GradNode { Tensor *result;
                  std::vector<Tensor*> inputs;
                  std::function<void(Scope&)> backward_fn; };

class GradientTape {  // thread_local singleton
    static GradientTape& instance();
    void push(GradNode* node);
    void backward(Tensor* loss, Scope& arena, blocked_set_t* blocked = nullptr);
    void clear();
};
```

`backward` идёт по dataflow-графу в reverse-topo (потребители раньше производителей) через
`result → producer`-карту, накапливая `+=` градиенты в `inputs`. `GradNode::inputs` хранит
`Tensor*` (не SExpr) — autograd не завязан на интерпретатор.

Нейросимвольные `NS-IF`/`NS-GRAD!` (см. гл. 14) добавляют «вину»: STE-градиент для
роутера, блокировка ошибочных ветвей через `blocked_set_t`.

### Контракт ленты

1. **Узлы принадлежат tape.** Аллоцируются в heap, удаляются в `clear()`. Узел не
   переживает свою ленту и не должен использоваться после её очистки.

2. **`defvar` промоутит промежуточный тензор.** `defvar`/`setq` клонирует тензор в
   stable-scope и перенаправляет ребро `result` tape-ноды на stable-копию. Градиент течёт
   в тензор, который видит пользователь.

3. **`reset_scratch` «запечатывает» незакрытые графы.** Если форма верхнего уровня
   завершилась без `GRAD!`, и какие-то tape-узлы ссылаются на scratch-тензоры, эти тензоры
   один раз клонируются в stable, и рёбра переписываются. Обучающие циклы ничего не платят
   (tape пуст после каждого `GRAD!`).

4. **Биндинги не алиасят литералы.** `let`/`bind_and_eval` копируют скалярные значения
   (fixnum/flonum/symbol) в свежие ячейки — `setq` больше не мутирует исходные литералы
   в коде.

5. **Adam-состояние живёт на параметре.** `m`/`v`/`step` аллоцируются в stable-scope на
   самом параметре: это персистентное состояние оптимизатора, обязано переживать
   checkpoint/restore итераций `while` (иначе — висячие указатели и NaN).

6. **HLO/NS-прокси — строковые хендлы.** Биндинги копируют скаляры, а fixnum-прокси при
   копировании теряет идентичность (ключ `trace_nodes_`). Поэтому прокси — строки
   `"#hlo-node:<id>"`: идентичность переживает и копирование биндинга, и promote-клон.
   `HLO-COMPILE` на чужом хендле — внятная ошибка вместо падения.

7. **Guard времени жизни градиентов.** При входе в `backward()` на каждый нодовый тензор
   регистрируется guard (`Scope::track_dtor` + `Tensor::clear_grad`): при смерти
   scratch-региона `->grad` честно обнуляется — следующий backward не пишет в
   переиспользованный пул.

8. **Градиент не переживает границу формы.** Градиент аллоцирован в scratch-арене формы и
   к её концу уже мёртв, поэтому значение после границы честно нулевое, а на stderr
   печатается предупреждение о непрочитанных градиентах. Прочтением градиента считается
   `GRAD-OF`, `SGD-STEP` и `ADAM-STEP` — они снимают параметр с учёта.

9. **Шаги оптимизатора детектируют нечисловые значения.** `SGD-STEP` и `ADAM-STEP`
   замечают NaN/Inf и не уводят веса в `inf` молча.

10. **Сравнение плавающих в release ненадёжно.** Release собирается с `-ffast-math`, а он
    включает `-ffinite-math-only`: компилятор вправе считать, что NaN не бывает, и
    сворачивает `a == b` в `!(a<b) && !(a>b)`, что для NaN истинно. Поэтому конечность
    проверяется примитивом `(FINITE? t)` (битовая проверка), а числовые пороги в тестах
    ставятся выше baseline вырожденного решения.

### Смежные механизмы

- **Tail-call optimization.** Self-хвостовой вызов пере-биндит кадр (параметры копируются
  в heap-стор, итерация освобождается через `restore`) вместо роста стека. Гейт: если
  итерация оставила живые tape-узлы, TCO для неё выключается — обычная рекурсия держит кадр.
  `let`/`let*` разворачиваются в транполине до 8 уровней, глубже — обычный кадр.
- **Кэш раскрытия макросов.** `MacroExpander` кэширует раскрытие на форму-источник (хит по
  содержимому), инвалидация — на `defmacro`/`define-compiler-macro`; `macrolet`-протяжения
  вне кэша.
- **DEBUG-TRACE.** Кольцевой буфер на 4096 записей (`src/core/trace.{hpp,cpp}`) с
  категориями `scope`/`clone`/`expand`/`macro`; примитивы `DEBUG-TRACE`, `TRACE-STATS`,
  `TRACE-DUMP`, `TRACE-CLEAR`, переменная окружения `QUALISP_TRACE`. Трейс отвечает на
  вопрос «как сюда пришли», тогда как `(mem-report)` — «что стало» (гл. 5). Выключенный
  трейс стоит один relaxed load, сам буфер не течёт.

## 19.9 HLO-пайплайн

### Узлы и операции

Граф из узлов `HloOp` (см. `src/hlo/graph.hpp`): `PARAMETER, CONSTANT, DOT, ADD, MUL, RELU,
FUSION, TUPLE, OUTPUT, LOOKUP, ARGMAX, SOFTMAX, COMPOSITE, CODEGEN, WEIGHTED_LOOKUP, ROUTE`.
ROUTE-узел описывает только программу: ветви-подграфы внутри узла и OR-overlap-метки; то,
что оставляет прогон, живёт на программе (§19.9). Разрешение вины — OR-aware, взятая
ветвь выигрывает: `resolve_correct` у узла и у tape-роутера считают одинаково.

### Форма графа — мастер, `HloGraph` — кэш понижения

`HloProgram` (`src/codegen/hlo_codegen.hpp`) держит две разные вещи:

| поле | что это | кто читает |
|------|---------|-----------|
| `graph_form` (`SExpr*`) | **мастер-копия программы**, живёт в `storage` и умирает с программой | `GRAPH-DATA`, образ на диске, понижение |
| `lowered` (`HloGraph*`) | производный одноразовый артефакт | эмиттер, фоллбэк, `GRAPH-RUN-PASSES`, `HLO-ROUTE-GRAD!` |

`program_lowered(hp, arena, interp)` строит кэш из формы при первом обращении и заканчивает
развёртыванием COMPOSITE. Источником правды форма не перестаёт быть: кэш можно выбросить и
получить то же самое. Обратной записи нет — держать два представления в согласии нечего,
потому что нечего рассинхронизировать.

### Проходы на форме

CSE, DCE и FUSION работают на форме как `SExpr → SExpr`. Идентичность узла — это его `id`,
а не указатель, поэтому замена потребителей есть переписывание списка чисел, а слитый
дубликат выбрасывается прямо при эмиссии формы.

Форма узла: `(OP id (shape) (inputs) payload dtype)` — шестой элемент несёт dtype (символом;
старые формы без dtype читаются как F32). Константы в форме — **атомы-тензоры**, а не
списки float: сохраняется dtype, и round-trip точен для любого dtype. Имена операций берутся
из одной таблицы `hlo_op_names` в обе стороны, так что добавление операции — одна строка.

**FUSION** сливает цепочки ADD/MUL/RELU (не только содержащие RELU) и кодирует **строгий
левый сгиб**: головная бинарная операция берёт два внешних входа, следующая — сгибающийся
результат и ровно один новый вход, RELU — только сгибающийся результат. Точка соединения
двумя поглощаемыми входами левым сгибом не выражается, поэтому такие узлы сливаются лишь
по отдельности.

**dtype** живёт и в `HloNode`, и в форме. `fits_stencil` пропускает в стенсил только F32 и
I32 — **f16 уходит в интерпретаторный фоллбэк** (в стенсиле f16 портил данные без ошибки).
Значения в фоллбэке пишутся через `get_float`/`set_float`, поэтому тихой порчи dtype нет;
`ARGMAX` возвращает F32.

### Память фоллбэка

- Буферы узлов освобождаются **по последнему потребителю**: выделение ленивое, значение
  узла считается один раз.
- Свёртка FUSION пишет в буфер узла (накопление in place) вместо выделения тензора на
  операцию: прирост пика RSS фоллбэк-графа со свёрткой из 7 операций на 2048×2048 f32 —
  ~50 МБ вместо ~390 МБ.
- Состояние прогона ROUTE живёт **на программе** (`RouteRunState`), а не в узле: `HloNode`
  описывает только программу, понижение можно перестроить, backward не теряется (состояние
  адресовано по id).
- Константы трейса владеют своими байтами: константа клонируется в хранилище, живущее
  столько же, сколько сессия трассировки; `seal_graph_constants` копирует из хранилища,
  переживающего граф.
- Под трассировкой рекурсия **не хвостовая** — иначе откат scratch освобождал узлы графа,
  и `HLO-COMPILE` резолвил вход в средний узел графа.

### Стенсильный эмиттер

Кодоген (`src/codegen/hlo_codegen.cpp` → `hlo_stencil.cpp`): **единый стенсил-эмиттер**
`compile_hlo_stencil` — один проход по графу выписывает байтовые AVX2-стенсилы с «дырками»
(imm-формы, rel32, movabs-адреса хелперов), патчит их и копирует в W^X-страницу процесса.
Кернел — C-ABI-функция `void fn(float** params, float* out)`; узел `DOT` → `call cblas_sgemm`
(адрес прошивается movabs-патчем). Подмножество: PARAMETER, скалярный CONSTANT, DOT,
ADD/MUL/RELU, CODEGEN (`tanh`/`fabs`/`fneg`/`clamp01`/`copy`), SOFTMAX, ARGMAX, LOOKUP,
WEIGHTED_LOOKUP; фоллбэк — для ROUTE, COMPOSITE/TUPLE. Промежуточного IR нет — LLVM-путь
удалён.

Кэш — in-process по fingerprint (версия эмиттера + shape-ключ + dtype). Дискового кэша нет.
Fallback: граф вне компилируемого подмножества → интерпретация клонированного графа
(`HloGraph::clone()`); `HLO-RUN` защищён arg-count guard (ошибка вместо чтения за границей).

### Граф как данные

`GRAPH-DATA` / `GRAPH-FROM-DATA` — S-выражение с ROUTE-ветвями и константами (точный
круговой рейс print→read→print), `GRAPH-RUN-PASSES` — оптимизатор из Lisp: `(CSE DCE FUSION
OPT)`. `GRAPH-DATA` отдаёт копию формы и **не строит понижение вовсе**: инспекция не должна
зависеть от артефакта понижения. Range-кернел `fn_range(params, out, start, end)` строится в
той же странице (гл. 10.6).

### Copy-and-patch JIT (`src/jit/`)

Горячие замыкания компилируются тем же методом — без IR и без LLVM; детали — гл. 20 и
`src/jit/CODE.md`. Ключевые факты для экскурсии:

- **ABI страницы**: `SExpr *entry(SExpr **argv, size_t argc, Interpreter *interp)`; локали —
  слоты `[rbp - 8*(slot+1)]` по 8 байт (boxed `SExpr*` или raw int64/double).
- **Один проход по S-дереву**: инлайнятся `if`/`let`/`while`/`setq`/`progn`, арифметика и
  сравнения с двумя числовыми аргументами; всё остальное — `lookup_var` + `call_value`,
  поэтому рекурсия jit-to-jit работает.
- **Bail-блок**: несовпадение типа или неизвестная ситуация → возврат bail-токена →
  интерпретатор повторяет тело по master-копии (transaction-retry). Хук — `Interpreter::apply`
  (case `JITPROG`).
- **Эпоха и heal**: любое `defmacro`/`define-compiler-macro` → `jit::invalidate_all()` (все
  страницы сбрасываются); stale-обёртка пересобирается после 64 интерпретируемых вызовов,
  засевая решётку типов наблюдаемой сигнатурой (`JIT-SIG`).
- **Только SysV x86-64** — стенсилы выписаны руками под эту ABI.

## 19.10 Neuro-Symbolic Router

`src/ns/route.{hpp,cpp}`:

- `RouteHop` — одна символьная развилка: `decision_logits`, `batch`, `taken[]`, `correct[]`,
  `branch_covers[]`, диапазон tape-узлов.
- `RouteHop::resolve_correct(label, taken)` — case-insensitive; OR-aware: taken-ветвь
  первая; если она покрывает label → маршрут верный (виновен лист).
- `Router` — упорядоченные хопы от корня к листу с depth-трекингом.
- `GuiltDetector::detect(router, labels)` → `GuiltVerdict { kind, guilty_hop, per_sample_hop,
  per_sample_kind }`.
- `NS-IF` (`do_ns_if`) выбирает ветвь per-sample по argmax, записывает хоп, исполняет ветвь
  первого образца (для чистого батча; для смешанного — маршрут/guilt per-sample, но
  исполнение одно).
- `NS-GRAD!` (`do_ns_grad`) блокирует tape-узлы виновных ветвей, добавляет
  softmax-weighted STE в `decision_logits->grad`.
- Роутер работает и поверх макро-слотов (`ns-route`/`ns-route-sample`/`ns-route-index`):
  NS-RL-движок blackit живёт на уровне stdlib + примитивов `SAMPLE`/`EVAL-SANDBOXED` —
  гл. 14.10.

## 19.11 LSP

`qualisp-lsp` (`src/lsp/server.cpp`, отдельная цель CMake без линковки движка) — полноценный
Language Server, автономный (JSON-RPC over stdio, встроенный JSON-парсер с честным
анескейпингом `\n \t \r \b \f \uXXXX`):

- **Ядро**: двухпроходный сканер S-форм, повторяющий конвенции ридера (строки с
  экранированием, `;`-комментарии, диспетч-макросы `#' #(` `#\`, quote/quasiquote/comma);
  инкрементальная синхронизация (`change: 2`) через офсетный сплайсинг — без перечитывания
  документа.
- **Диагностики**: незакрытая форма (с позицией открывающей скобки), лишняя `)`, незакрытая
  строка — точные диапазоны.
- **Воркспейс-индекс** на initialize: скан `*.qlsp` (глубина ≤ 8, кап 500 файлов,
  пропуск `.git`/`build*`/`node_modules`/`.cache`); didSave/didClose перечитывают файл с
  диска.
- **Навигация**: cross-file definition (Location | Location[], кап 20), references (кап 200),
  documentHighlight, rename (WorkspaceEdit.changes по индексу, кап 50 URI), workspace/symbol
  (кап 100), documentSymbol (14 def-видов + `deftest` со строковым именем); сопоставление
  символов без учёта регистра (ридер апкасит).
- **Комфорт**: foldingRange (пары OPEN/CLOSE на разных строках), formatting (Lisp-отступы:
  тело defun/let/if/… = col+2, аргументы после головы headCol+len+1; многострочные строки и
  пустые строки не трогаются), ~140 автодополнений из `lsp_tables.inc` с
  hover-документацией.
- **Ограничения**: семантические диагностики (unbound vars, арность — нужен eval-контекст),
  semantic tokens, signatureHelp; колонки в байтах (не UTF-16).
- Регрессионный тест — `tests/lsp_test.sh` (JSON-RPC-драйвер over stdio, 35 проверок).
- Пакетный режим того же движка — `tools/qlsp-check.py`: диагностика читателя
  (незакрытые/лишние скобки) плюс структурный план `defun`-дерева (`--symbols`).
  Для C++ навигация идёт через `clangd` по `build/compile_commands.json`, который генерирует
  CMake (`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`).

## 19.12 Сборка и инструменты

```bash
# Native Linux (LLVM не нужен)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)

# Кросс-компиляция qvalent/qualisp-lsp под Windows из Linux
# (для qualisp корректный путь — нативная MSYS2-сборка)
cmake -B build-windows \
  -DCMAKE_TOOLCHAIN_FILE=cmake/x86_64-w64-mingw32.cmake \
  -DCMAKE_BUILD_TYPE=Release
```

Продукты сборки (из `CMakeLists.txt`):

- `qualisp` — REPL/интерпретатор + встроенный JIT (OpenBLAS, OpenMP; LLVM не нужен).
- `qualisp-lsp` — LSP-сервер (только `src/lsp/server.cpp`).
- `qvalent` — пакетный менеджер (только `src/qvalent/main.cpp`).

AOT-компилятора нет: всё нативное собирается на лету из S-выражений.

Модули stdlib вшиваются в бинарник на этапе сборки (`cmake/stdlib_embed.cmake`, через
`cmake/gen_stdlib_header.cmake` — без `xxd` для переносимости на MSYS2/MinGW) — поэтому
дистрибутив Qualisp — один файл ~2 МБ.

### Windows-сборка

Windows-сборка портируема (mingw-портируемость: коллизии enum, guard BLAS-линковки, native
DOT) — CI зелёный на Linux и Windows.

Корректные пути для Windows-бинарника:

1. **CI** (`.github/workflows/build.yml`, job `windows`): MSYS2 MinGW64 + OpenBLAS нативно под
   Win64 ABI. Артефакты CI публикуются как релизные бинарники.
2. **MSYS2 локально**: установить `mingw-w64-x86_64-gcc`/`mingw-w64-x86_64-openblas`, затем
   `cmake -G "MinGW Makefiles" -S . -B build && cmake --build build -j`.
3. **Кросс-компиляция `qualisp-lsp`/`qvalent` из Linux** через toolchain-файл (см. выше).

## 19.13 Ключевые константы

| Параметр | Значение |
|---|---|
| Блок scratch-Scope | 16 МБ (`DEFAULT_BLOCK_SIZE`) |
| ConsCellPool | предвыделенный, фиксированный размер ячейки; `acquire`/`free` — O(1) |
| Guard глубины | 10 000 eval-кадров; ловимая ошибка + `(EVAL-MAX-DEPTH n)`; RAII-декремент, eval жив после catch |
| Кольцо DEBUG-TRACE | 4096 записей; категории `scope`/`clone`/`expand`/`macro`; выключено по умолчанию |
| rc-контракт | `refs_` в стабильной ячейке (rc=1 при создании, `nil` бессмертен); `QUALISP_RC_POISON=1` — QA-хук против use-after-release |
| LSan-гейт | ASan c `detect_leaks=1` + `ulimit -s unlimited`, рецепт `tools/asan_gate.sh`; гейт собирает **все** падения за прогон и падает в конце, а не на первом файле |
| Владение | структура — RC (`stable_ref`/`stable_release`, каскад worklist, шарится); тензорные буферы — без RC: один владелец, точный размер, `TensorBufferPool` |
| TensorBufferPool | exact-size, кэш по (shape, dtype), при переполнении лимита буферы уходят системе |
| SIMD (AVX2/FMA) | 8×f32 = 256 бит |
| SIMD (NEON) | 4×f32 = 128 бит |
| Кэш HLO | in-process fingerprint (версия эмиттера + shape-ключ + dtype) |
| Порог heal JIT | 64 интерпретируемых вызова (`k_hot_calls`, гл. 20) |
| Модули stdlib вшито | 13 (core string dl ns ml io regex audio datetime pkg errors visual tablet) |
| `dl` | слои-как-данные; `linear`/`net-forward`/`train-step`/`train-epochs` |
| Автодополнения LSP | ~140 (`lsp_tables.inc`) |
| Стенсилы JIT | SysV x86-64 |
| GCC/Clang (минимум) | GCC 13+ / Clang 16+ (C++20); LLVM не нужен |
| qvalent локальный каталог | `./qualisp-packages/<name>/` |
| qvalent кэш | `$HOME/.cache/qualisp/<deployer>/<repo>/` |
| QSRD v2 magic | `"QSRD"` (u16 ver) |
| Tail-call | self-хвост пере-биндит кадр; гейт по живым tape-узлам; `let`/`let*` разворачиваются до 8 уровней |
| Контракт CI | `run-tests` считает падения в `*test-failed*`; процесс возвращает 1 при любом упавшем сюите |
| Тесты | сюит зелёный (Release, Linux); ASan-гейт с `detect_leaks=1` по всему корпусу; poison-гейт `QUALISP_RC_POISON=1` по всему корпусу; прод-датасеты `tests/data/*.csv` прогоняются продакшен-сценарием обучения |

## 19.14 Известные ограничения

- **GPU-бэкенда нет.** Устройство одно — CPU. `Device` остаётся абстрактным интерфейсом
  (alloc/free, пул, `launch_*`), чтобы бэкенд можно было добавить позже, но в дереве нет
  ни CUDA-, ни ROCm-кода. `launch_*` покрывают float и half (F16); double/int — через
  `get_float`/`set_float`.
- **HLO:**
  - f16 идёт в интерпретаторный фоллбэк (в стенсиле — только F32/I32);
  - часть буферов узлов при ROUTE освобождается небезопасно: тела ветвей читают значения
    главного графа «ambient», без ребра от ROUTE. Направление — собственное пространство
    буферов у ветвей, то есть переработка исполнителя, а не точечная правка;
  - `ABS` на тензоре отдаёт мусор (примитив скалярный и не проверяет тип);
  - `codegen_kind` — единственное оставшееся второе представление: для `:codegen` тело
    отбрасывается, а смысл ядра живёт в C++ if-chain эмиттера. Это сознательный долг.
- **Диспетчеризация ROUTE стенсилами** не поддерживается: ROUTE исполняется fallback-программой.
- **`let` под биндингом с 1- или 2-байтным dtype теряет свои биндинги** (F16, U8; на F32 не
  воспроизводится). Практический обход: такие тензоры связывать через `defvar`, вложенный
  `let` под ними не ставить.
- **Per-value RC для scratch-ячеек не завершён**: счётчик у scratch-ячеек есть, но полноценное
  владение ими на границе жизни scratch-значений — впереди (оптовый возврат scope не умеет
  освобождать поячеечно).
- **Хвостовой перебиндинг TCO делит под RC только уже стабильные параметры**; scratch-параметры
  и горячий ML-путь копируются. Нужна смена конвенции возврата — это изменение всей цепочки
  eval, поэтому вопрос остаётся открытым.
- **Сравнение плавающих в release ненадёжно** (`-ffast-math`): конечность — через
  `(FINITE? t)`, числовые пороги в тестах — выше baseline вырожденного решения.
- **Глубокая не-хвостовая рекурсия переполняет стек**; TCO снимает этот предел для
  self-хвостов.
- **Острова циклов через `set_cdr` не собираются** — GC нет, циклы по определению не
  разбираются.
- **`run-tests` освобождает реестр в конце**: повторный вызов в том же скрипте даёт `0/0
  passed`.
- **JIT-стенсилы** — только SysV x86-64; покрытое подмножество форм невелико;
  `MACROLET`-тела не компилируются; bail = повтор тела интерпретатором (transaction-retry,
  гл. 20.7). Есть разобранный, но не устранённый случай: ошибка во вложенном вызове внутри
  JIT-обёртки роняет тест под ASan (в release проходит) — причина в цепочке освобождения
  master-копии, нужен отдельный разбор.
- **EnvFrame** — SExpr-alist; переработка окружения (self-hosting) — направление развития.
- **Произвольное переписывание графа** (`graph-node`/`graph-rewrite`) не поддерживается; база
  есть: `GRAPH-DATA`/`GRAPH-FROM-DATA`/`GRAPH-RUN-PASSES`.
- **Семантические диагностики LSP** (unbound vars, арность), semantic tokens, signatureHelp —
  roadmap.
- **CI** — GitHub Actions (`build.yml`) на Linux и Windows; тесты — `.qlsp`-скрипты в `test/`
  (standalone) и `tests/` (`deftest`/`run-tests`); артефакты CI публикуются как релизные
  бинарники.

---

Подробности по подсистемам: гл. 10 (HLO-стенсилы), гл. 20 (JIT copy-and-patch), гл. 21
(образы), гл. 22 (Tablet). Тесты — сюиты в `tests/` (плюс фазовые в `test/`): `jit.qlsp`,
`image.qlsp`, `hlo_stencils.qlsp`, `hlo_stencil_ops.qlsp`, `hlo_graph_data.qlsp`,
`hlo_form_canary.qlsp`, `f16_dtype.qlsp`, `stdlib_losses.qlsp`, `tablet.qlsp`, `dl.qlsp`,
`memory_setq.qlsp`, `loop_lifetime.qlsp`, `memory_pools.qlsp`, `production_mem.qlsp`,
`expansion_cache.qlsp`, `debug_trace.qlsp`, `tail_call.qlsp`, `rc_contract.qlsp`,
`rc_escape.qlsp`, `arena_stress.qlsp`, `live_globals.qlsp`, `equal.qlsp`, `tref.qlsp`,
`print_arity.qlsp`, `math_ops.qlsp`, `tensor_print.qlsp`, `grad_lifetime.qlsp`,
`ns_router_hlo.qlsp`, `homoiconic.qlsp`, `io.qlsp`, а также продакшен-сюит
`production_training.qlsp` (CSV-датасеты из `tests/data/`: классификация и регрессия,
HLO-депл с проверкой против интерпретатора, плоская память по эпохам, round-trip образа и
весов). Проверочный контур памяти — гл. 5.5: poison-гейт, ASan-гейт и `(mem-report)`.
