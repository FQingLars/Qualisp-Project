# Глава 19. Устройство компилятора

Краткая экскурсия во внутренности Qualisp релиза 0.19.15 (реализация на Zig): как исходник превращается в программу, какие подсистемы за этим стоят и как они делят ответственность. Полные решения по языку — в `SPEC.md`, архитектурная сводка и дорожная карта — в `AGENTS.md`, публичные API подсистем — в `src/CODE.md` и `src/*/CODE.md` дерева разработки.

## 19.1 Пайплайн

```text
Исходник
  → lexer.zig            (текст → токены со спанами)
  → reader.zig           (токены → S-выражения в Store)
  → Session.expandBody   (раскрытие макросов, в т.ч. в телах defgraph/defuse)
  → checker.zig          (статическая проверка типов, declareFunctions)
  → Emitter.compile      (copy-and-patch: байты + relocations + source map)
  → CodePage (W^X)       (выбрасываемый кэш)
  → entry(frame, env, rt, out) → Value

  defgraph → HLO-граф (trace на нулевых placeholders) → CompiledGraph
           → ядра x86-64 / W^X-страница → forward / backward

  ns-if → src/ns: атомы → знаки → band → ROBDD-покрытие → router → compaction
                    → подграфы ветвей (только непустые)
```

**Инвариант одной формы.** S-выражение — единственная авторитетная форма программы. Всё нативное — выбрасываемый кэш, выведенный из S-выражения: JIT-страница пересобирается из мастер-формы `Func`, HLO-граф — производный одноразовый артефакт мастер-формы `defgraph`, скомпилированный `defuse` — кэш скалярного цикла. Промежуточного IR, байткода, LLVM и AOT-компилятора нет. Нет и интерпретатора/tree-walk: конвейер — `read → expand → typecheck → emit → run`.

Top-level форма компилируется и исполняется сразу; тело `defun` — при определении (`emitDefun` строит `FunctionTemplate` с готовым `entry`). По одному проходу по S-дереву, в порядке LNR: оператор, затем аргументы слева направо; top-level формы — слева направо.

## 19.2 Структура исходников

```text
build.zig, build.zig.zon
src/
├── main.zig          CLI/REPL/runner; --trace, raiseStackLimit
├── session.zig       Session: read→expand→check→emit→run; world/scratch;
│                     загрузка модулей, defgraph/defuse, QSRD
├── lib.zig           публичные экспорты
├── span.zig          Span/Diagnostic
├── symbol.zig        интернирование символов (верхний регистр)
├── value.zig         Value, атомы, Cons, Rope
├── store.zig         bump-хранилище cons/байтов, mark/restore
├── lexer.zig         source → токены со спанами
├── reader.zig        токены → S-выражения
├── printer.zig       Value → читаемый текст
├── program.zig       единый корень top-level форм
├── runtime.zig       глобалы, Func/Env, кадры вызовов, страницы
├── form.zig          cons/list, quasiquote
├── syntax.zig        :syntax-операции
├── types.zig         Type, парсер, assignable
├── checker.zig       статический typechecker ядра
├── overload.zig      ключи и разрешение перегрузки
├── macros.zig        defmacro, expander, gensym
├── eval.zig          evalValue: expand→check→emit→run
├── jit/              emitter, memory (W^X), helpers, cache, stencils, abi
├── prims/            list, math, string, io, type, error
├── object.zig        struct/trait логика
├── tensor/           Tensor + eager-операции + TensorBufferPool
├── hlo/              граф, trace, forward/backward, compile/pool/fused, kernels
├── ns/               atom, expr, bdd, coverage, router, compaction, form, plan
├── ffi/              C-типы, trampoline, загрузка библиотек
├── image/            бинарный формат QSRD
├── profile.zig       трассировка Chrome Trace JSON
├── stdlib.zig        11 встроенных модулей (через @embedFile)
├── stdlib/*.qlsp     core, string, collections, test, io, ml, dl, symbolic,
│                     argparse, tablet, neuro (+ probe.qlsp — только тесты)
├── tests.zig         регистрация 834 тестов
├── *_test.zig        24 файла с тестами
└── test_support.zig  общий harness
tools/qsrd/           читатель QSRD (inspect/verify/cat/params/symbols/extract)
tools/qprof/          агрегатор трасс self/inclusive по именам
```

Stdlib — Qualisp-код, вшитый в бинарник через `@embedFile`; `core` загружается и импортируется при старте, остальные модули — лениво при первом `import`. `probe.qlsp` — test-only fixture, вшивается только в test-сборки. В каждом подкаталоге `src/X/` лежит `CODE.md` с API и ограничениями подсистемы.

## 19.3 Reader

Лексер отделён от reader: `lexer.zig` превращает текст в токены со спанами, `reader.zig` — токены в S-выражения. Reader не типизирует и не исполняет: он возвращает `Value`/`Cons`, как требует инвариант одной формы. Ошибки — `diag` + error union, без panic.

- Пробелы, табы, переводы строк; `;` — комментарий до конца строки.
- Списки `( … )`, dotted pair `(a . b)`; `(`/`)`, `.`, `'`, `` ` ``, `,`, `,@` — отдельные токены.
- `'x → (quote x)`, `` `x → (quasiquote x)``, `,x → (unquote x)`, `,@x → (unquote-splicing x)`, `\ → lambda`.
- int64/f64 (`42`, `-7`, `3.14`, `1e-5`); `0xFF` и `1/2` числами не читаются.
- Строки `"…"` с escape `\n \t \r \\ \"`; символы `#\a`, `#\space`, `#\newline`, `#\tab` (Unicode scalar u21). Векторы, `#'`, `#| |#` в срез не входят.
- Символы регистронезависимы, интернируются в верхний регистр (`(quote a)` печатается `A`); `:name` — keyword, самовычисляемый.

> 🐍 Аналог — tokenizer/parser в CPython, но результат — не `ast.AST`, а cons-ячейки: код и данные одного представления.

## 19.4 S-выражения

Реальное `Value` из `src/value.zig`:

```zig
pub const Value = union(enum) {
    nil,
    t,
    int: i64,
    float: f64,
    char: u21,
    symbol: SymbolId,
    string: Rope,
    cons: *Cons,
    func: *Func,
    tensor: *tensor_mod.Tensor,
    future: *Future,
    structure: *StructInstance,
    result: *Result,
    ptr: ?*anyopaque,
};
```

Порядок тегов фиксирован инвариантом (`nil == 0`, `string == symbol + 1`): инлайн-проверки эмиттера используют `tag <= :symbol`. Cons-ячейки неизменяемы и несут спан (файл/строка/колонка); спан есть и у атома-литерала. `nil` — false, «None» и пустой список одновременно; `t` и интернированные символы бессмертны.

Структура — **атом**: тег + типизированные слоты, мутируется `set-slot`; гомоиконность относится к коду, а не к каждому рантайм-значению. `Func` — атом: код (JIT-страница или примитив) + захваченное окружение; при escape в stable окружение копируется целиком. `Tensor` — атом: dtype + concrete shape + плоский буфер. `Result`, `Future` и `ptr` — тоже значения.

Память значений — `Store`: bump-регион cons-ячеек и байтов с `mark()/restore()` и детерминированным сбросом scratch. Ни GC, ни reference counting, ни общего аренщика нет.

## 19.5 Окружение и вычисление

`Runtime` держит `world`/`scratch`, плотный массив глобалов `values: []Value` и отдельный `present: []bool` (JIT читает оба напрямую; `values[id]` не читается при `present[id] == false`). В `world` значения копируются глубоко (`cloneToWorld`); `owner == .world` делает повторную копию no-op.

Скоупы создают вызов функции, top-level форма и итерация `while`; не создают `let` (биндит в текущем) и `progn`. На выходе — откат указателя `Store` и освобождение тензорных буферов скоупа, возвращаемое значение копируется в родительский. `setq` перемещает значение в биндинг; `defvar` копирует в stable: `(defvar b a)` — копия, не алиас. Stable никогда не ссылается на scratch; замыкание захватывает локальные биндинги копией, глобалы — live. Явное удаление — только `clear`.

### Кадры вызова и соглашение вызова

Кадры — одна неподвижная bump-арена `frame_area` с указателем `frame_top`, строго LIFO. `takeFrame`/`giveFrame` — сложение/вычитание размера; `replaceFrame` заменяет кадр на месте для хвостового вызова с большим калом; в Debug сверяется указатель. Ёмкость в коде — `1024 * 1024` слотов (~24 МБ адресного пространства, память трогается по мере глубины); рост невозможен, поэтому исчерпание — типизированная ошибка `call depth limit exceeded` с позицией. Машинный стек поднят до 256 МБ (`raiseStackLimit`; `stack_size` в PE), иначе арена не успела бы диагностировать исчерпание.

Соглашение вызова JIT-страницы:

```c
entry(frame: *Value, env: ?*Env, rt: *Runtime, out: *Value) void
```

Кал сам пишет результат (24 байта) в `*out` и делает `ret`. `out` указывает в кадр вызывающего вызывающего и остаётся валидным, пока кал работает, — это снимает главный блокер TCO. Регистры страницы (callee-saved): `rbx` — слоты кадра, `r12` — env замыкания, `r13` — `*Runtime` (пролог кладёт `rt`, инлайн-код читает глобалы без TLS), `r14` — `out`; пролог — четыре push + `sub rsp,8`, эпилог восстанавливает.

- **Самовызов в хвостовой позиции** (`tail_self`): аргументы считаются во временные слоты, затем копируются в слоты параметров (поэтому `(f b a)` безопасен), `jmp tail_head`. Кадр, traceback и `out` сохраняются; условие — тот же кал по имени, точная арность, без `&optional`/`&rest`.
- **Хвостовой вызов другой функции** (`emitTailPrepare` → `ql_tail_prepare(frame, name, first, argc, capacity)`): кал разрешается в рантайме (глобал или метод по тегу получателя), потому что при эмиссии тела `Func` второй функции взаимной пары ещё нет; `placeArgs` переносит аргументы, верхняя запись traceback переименовывается, пролог снимается, `jmp entry`. Если кал больше кадра, берётся кадр из арены, старый возвращается, владелец — точка входа, освобождающая цепочку через `Runtime.tail_frame`. Промах → обычный `ql_call_global` с прежней диагностикой. Ёмкость кадра передаётся `mov r8d, imm32` и патчится после эмиссии тела.
- `Runtime.enterTailScope`/`leaveTailScope` окружают каждый вход в JIT-код: вложенная страница `eval` не может освободить чужой кадр.
- Проверено: взаимная рекурсия на 1M вызовов в постоянном числе кадров, `eval` внутри взятого кадра, бэктрейс без исчезнувшего вызывающего, `catch-error`.

Регистровые аргументы (этап D плана) измерены и отклонены: фиксированная цена вызова ~10.6 нс против 1.3–1.5 нс на аргумент; кал всё равно спиливает значение в слот. Вместо этого — дешёвая bump-арена кадров.

### Ошибки и source maps

`(:result T)` = `ok T | err Error`; `throw : :a → :never`; `catch-error : (:func () T) → (:result T)`. `Error` — структура `message :string`, `cause :(:option (:struct Error))`, `traceback :(:list CallFrame)`; кадр — `(name file line form)`. Ошибки компиляции/типа — не `Error`-значения, а отказ с диагностикой со спаном.

Runtime-ошибка: хелпер зовёт `bailAt(@returnAddress(), msg)`, сообщение сохраняется в `Runtime.error_message`, позиция — в `error_location`. Source map — `CodePage.map: []MapEntry{start,end,span}` плюс имя файла: запись на каждую cons-форму, вложенные формы дают вложенные записи, `lookup` берёт самую внутреннюю, ~24 байта на форму. Адрес возврата указывает **после** `call`, поэтому lookup идёт по `offset-1`. CLI печатает `file:line:col: message`. Ошибки не разматываются через нативный C-стек, только через JIT-кадры.

## 19.6 Special forms

Ядро чекера и эмиттера: `quote`, `if`, `progn`, `let`, `setq`, `while`, `lambda`/`\`, `defun`, `defvar`, `and`, `or`, `cond`, `the`, вызовы; `funcall`/`apply`, `read-from-string`/`write-to-string`; `defmacro`, `macroexpand-1`, `macroexpand`, `macroexpand-all`, `gensym`; `defstruct`, `slot-value`, `set-slot`, `deftrait`, `impl`; `import`, `import-from`; `defgraph`, `defuse`; `ns-if`; `tensor`; `throw`, `catch-error`; `eval`. Примитивы (`+ - * /`, сравнения, `car`/`cdr`/`cons`, `print`, `map`/`filter`/`reduce`, тензорные ядра, `ffi-load`/`ffi-defun`, файловые и QSRD-примитивы, `future`/`await`) живут в реестрах, а не в спецформах.

- `defun`/`lambda` требуют аннотаций параметров и возврата; `&optional` внутри тела — `(:option T)`, `&rest` — `(:list T)`; `let` — параллельные биндинги; `let*` отсутствует. Пользовательских перегрузок нет: повторный `defun` с тем же именем — ошибка. `:any` запрещён.
- Сигнатуры всех литеральных top-level `defun` публикуются **до** проверки первой формы (`declareFunctions`), поэтому прямая ссылка вперёд и взаимная рекурсия типизируются; определение исполняется по порядку, вызов до определения — runtime-ошибка.
- Тела `defgraph`/`defuse` раскрываются отдельно (`Session.expandBody`) и клонируются в world: иначе тело `defuse` висело бы на scratch. Тело макроса проверяется в живом окружении (чекер берётся из `Runtime.checker`), поэтому макрос может звать хелперы модуля.
- `eval : :syntax → :a` — проверяемый каст; `:a` выводится из контекста; форма заново проходит expand/typecheck/emit, несовпадение типа — runtime-ошибка. Единственная документированная граница статики.

### Модули

Модуль — файл с `(export name …)` в конце; приватные хелперы не экспортируются. `(import m)` открывает все экспорты, `(import-from m (a b))` — только перечисленные, каждое имя проверяется по `module_exports`; повторный импорт идемпотентен, конфликт — `ImportConflict`. Приватные id создаются через `SymbolTable.internPrivate`: имя лежит в `names`, но не в `map`, поэтому `MODULE::NAME` нельзя набрать в исходнике. `Session.expose` алиасит публичное имя на приватное. `load`/`import` обрабатываются только на top-level. `core` — предустановленный prelude, остальные модули грузятся лениво.

## 19.7 Тензорная подсистема

```zig
pub const DType = enum { f16, f32, f64, u8 };
pub const Owner = enum { scratch, world, pool };
pub const Tensor = struct {
    dtype: DType,
    shape: []usize,          // конкретная форма
    storage: Storage,        // union(DType) { f16/f32/f64: []T, u8: []u8 }
    owner: Owner = .scratch,
    requires_grad: bool = false,
};
```

Тензор — атом: dtype + concrete shape + плоский буфер. Литерал `(tensor ((1.0 2.0) (3.0 4.0)))` всегда `:f32`, прямоугольность проверяется; `zeros`/`ones`/`randn` требуют dtype-ключ и литеральную форму. `randn` — xorshift64* в `Runtime`, не криптостойкий. Eager-тензор живёт в scope-арене, `PARAM`/`defvar` — stable, `TensorBufferPool` — только HLO.

Быстрые пути: `elementwise`/`map` — `@Vector` 8×f32 / 4×f64 без f64-раундтрипа; `matmul` по `i-k-j` с `@Vector`; reductions блоками — 8 независимых f64-полос, фиксированный порядок сведения (результат детерминирован, но не «слева направо»). `u8`-деление запрещено статически; `:u8` в `tsum`/`tmax` даёт `:fixnum`.

`TensorBufferPool` — exact-size free lists по `(len, alignment)`, реализация `std.mem.Allocator`; буферы переиспользуются, финальный выход копируется в scratch. `Graph.peak_bytes` — оценка пика, жёсткий cap не enforced.

Устройства нет: `Owner` — `scratch`/`world`/`pool`, `device` в перечислении отсутствует, все ядра — C-ABI функции на x86-64 с AVX2. Writable views и `t-slice` отсутствуют; `reshape`, `transpose`, `t-rows`, `t-select-rows`, `t-column` копируют. Печать — `#<tensor dtype (dims)>` без содержимого.

## 19.8 Autograd

`(param t)` помечает float-тензор `requires_grad` (`:u8` запрещён); флаг переживает `cloneToWorld`. `defgraph` трассирует тело один раз на нулевых placeholder-тензорах: управление статическое, `tref`/`tset!` в графе не поддерживаются, константы (`tensor`, `zeros`/`ones`/`randn` в теле) — stable. Вызов компилируется в `(%GRAPH-RUN gid ret arg…)`.

`ql_graph_run` сохраняет последний запуск в `Runtime.last_run` (сбрасывается в конце top-level формы); forward с `frame.retain = 1` не освобождает активации — все лежат в per-run `TensorBufferPool`. `(grad! loss)` сеет единицы формы выхода и идёт по узлам с конца, читая forward-значения из `run.values` — **без replay**, O(n) по узлам и O(n) по активациям до `grad!`/конца формы. Градиенты накапливаются в scratch; для `param`-входов копируются в stable и кладутся в `Tensor.grad`; `(grad-of t)` возвращает stable-градиент или нули той же формы/dtype.

Backward-правила: `add`, `sub`, `mul`, `div` (reduce по broadcast), `neg`, `matmul`, `reshape`, `transpose`, `relu`, `exp`, `log`, `tanh`, `softmax`, `sum`, `mean`, `max`; `sum` размножает grad, `mean` делит на число элементов, `max` отдаёт grad всем максимумам. Backward компилируется в обратную последовательность вызовов `ql_bwd_*` (x86-64, W^X); интерпретатор — fallback при ошибке компиляции.

**Ограничения.**

- `grad!` работает только с последним запуском графа в текущей top-level форме; не проверяет, что аргумент — именно его выход.
- Нет `tref`/`tset!`/`pow`/`sqrt`/`sin`/`cos`/`atan`/`argmax` в графе; reductions в графе — только float; `defgraph` с `:fixnum` фактически недостижим.
- Нет вложенных `defgraph`, рекурсии и data-dependent control flow; нет автоматического fusion/CSE/DCE — fusion только через `defuse`.
- Параметры, не участвовавшие в последнем запуске, сохраняют старый градиент; `grad-of` никогда не считанного параметра — нули.
- `clear` имени снимает callable, но граф/JIT-страница живут до teardown.

## 19.9 HLO-пайплайн и copy-and-patch JIT

### Узлы и операции

`Op` из `src/hlo/graph.zig`: `constant, identity, add, sub, mul, div, neg, matmul, reshape, transpose, relu, exp, log, tanh, softmax, sum, mean, max, ns_if`. Узлы с арностью > 2 разворачиваются в бинарные. `ns_if` несёт подграфы ветвей (`Graph.subgraphs`, по одному параметру-payload на ветвь). Граф — производный одноразовый артефакт, мастер — S-выражение `defgraph`.

### Компиляция и пул

`Graph.compiled` — ленивый `CompiledGraph`; второй вызов — без ожидания, первый ждёт job (`mutex` + `Condition`). `N = min(cpu_count, 8)` воркеров компилируют независимые графы; `Session` создаёт пул лениво, если есть `io`, иначе синхронно. Каждый compiled-вызов создаёт свой `TensorBufferPool`: один граф можно вызывать параллельно из futures без mutex. Ядра C-ABI — в `src/hlo/kernels.zig`, backward — в `src/hlo/bwd_kernels.zig`, узел NS — `ql_hlo_ns_if`; f64/f32/u8 ядра, SIMD AVX2.

`defuse` — отдельный путь: тело — одно elementwise-выражение (`+ - * /`, унарный `-`, `relu`, числа, параметры; без `let`, вызовов и тензорных литералов), все тензорные параметры — `:f32` одной конкретной формы, скалярные — `:flonum`/`:fixnum`. Первый eager-вызов компилирует скалярный f32-цикл x86-64 без промежуточных тензоров: AVX2 по 8 f32 + scalar-tail, константы в `CompiledFused.constants`. Внутри `defgraph` вызов `defuse` инлайнится в `let` перед трассировкой, поэтому forward/backward идут обычными HLO-узлами.

### Copy-and-patch JIT (`src/jit/`)

Конвейер: `Program (root) → typecheck → Emitter.compile(form) → CodePage → CodePage.run() → Value`. Контракт:

```zig
emit(form, env, expected_type) -> Code {
    bytes,
    relocations,
    bail_sites,
    source_map,
}
```

Эмиттер копирует байтовые стенсилы, патчит адреса хелперов (imm64) и rel32-переходы; дерево в момент исполнения не читается. W^X: `mmap`/`mprotect` POSIX, `VirtualAlloc`/`VirtualProtect` Windows; страницы лежат в `StableMemory`, выбрасываются и пересобираются из мастер-формы `Func`. `jit_abi.call` (`src/jit/abi.zig`) — `.c` (= SysV) на Linux, `.x86_64_sysv` на Windows: Zig порождает SysV-функции внутри PE, поэтому байты JIT-страниц идентичны на обеих ОС. `JitCache` ключуется по узлу формы; `clear` снимает глобал, но страницы живут до teardown.

Быстрые пути: литералы, копия слота (16+8 байт), чтение глобала (`id < present.len` + байт `present`), запись глобала (только immediate `tag ≤ :symbol`, не-worker, существующий id), promotion в `while`, откат итерации (`Emitter.mayAllocate`), инлайн fixnum-арифметики и сравнений с guard'ом тега и bail-блоком на прежний helper. Смещения считаются через `@offsetOf`: переезд поля ломает сборку, не семантику.

Ограничения: deopt не реализован — страницы не специализированы, helper-путь и есть generic-путь; guard'ы ставят `RuntimeError`. Struct-инстансы и `Result` живут до `Runtime.deinit`. `eval` возвращает `:syntax`/`:atom` консервативно; конкретика через `the`.

## 19.10 Нейросимволический роутер

`src/ns/`:

| Модуль | Роль |
|---|---|
| `expr.zig` | знаковое дерево NS-выражений, канонизация, инварианты |
| `bdd.zig` | ROBDD, порядок переменных по первому появлению |
| `coverage.zig` | проверки покрытия: `I ∧ ¬covered` UNSAT, `I ∧ Gi ∧ Gj` UNSAT, `I ∧ Gi` SAT |
| `form.zig` | разбор формы `ns-if` |
| `router.zig` | атомы `[B,K]` → знаки → band → ветвь → группы строк; проверка overlap/uncovered на месте |
| `compaction.zig` | counting sort, `gatherRows`/`scatterRows` для всех 4 dtype |
| `plan.zig` | runtime-план |

Дискретное условие — NS-функция: непрерывная level-функция, знак которой равен истинности; `>`/`<`/`and`/`or` не перегружаются. `ns-grt`/`ns-geq` — `a-b`, `ns-lss`/`ns-leq` — `b-a`, `ns-and`/`ns-or` — левая свёртка бинарной R-функции, `ns-not` — `-p`, `ns-eq`/`ns-neq` — сахар полосы. n-арная формула `Σaᵢ − √(Σaᵢ²)` знаково неверна при n ≥ 3.

Два спуска: знаковый (для роутера и проверки покрытия, без `sqrt`/делений) и числовой (только если значение используется как данные). Диспетчеризация никогда не берёт знак float-значения R-комбинации. `band[i] = ∃j |atom_j[i]| < ε`; `|combined|` не вычисляется. Guards разворачиваются в булевы формулы над K атомами; непокрытие, пересечение и пустая ветка — ошибки компиляции (`NsIfUncovered`, `NsIfOverlap`, `NsIfEmptyBranch`).

Forward: atoms → band → branch → counting sort → gather/вызов подграфа/scatter → tape (`branch[]`, `order[]`, counts/offsets, `band[]`, atoms). Стоимость O(B·K) на роутер плюс сумма строк исполненных ветвей; ветки без строк не исполняются. Backward идёт только по исполненным веткам; в `defgraph` маршруты пересчитываются из сохранённых атомов и `eps`, активации ветки пересчитываются на её строках. Градиента по решению роутера нет.

Мост `neuro`: `(term 'ТЕРМ '(vars))` и `term->code` собирают тензорный код из символьного терма; проверка отделена от генерации (`term-problem` возвращает первое сообщение, не бросая).

## 19.11 Инструменты трассировки

Профайлер — инструмент, не фича языка: форм `profile-start`/`profile-stop` нет; включение только `qualisp --trace out.json script.qlsp`. Пишется Chrome Trace JSON: события `B`/`E`, микросекунды, `pid=1,tid=1`; трасса пишется и при ошибке (`defer` в `main`); каждый хук вне трассировки — одна проверка `active`. Инструментированы только невидимые из AST точки: границы вызовов функций, вызов HLO-графа, фаза `ns-if`. Компиляция JIT-страницы входит в inclusive-время первого вызова.

`tools/qprof` — агрегатор: `qprof <trace.json> [--top N]`, таблица `name / calls / total_us / self_us / self%`. Без аргументов — `usage: qprof <trace.json> [--top N]` (exit 2).

```text
$ qprof /tmp/fib.json
trace: /tmp/fib.json (16722 events)
name                          calls       total_us        self_us   self%
FIB                            8361       255806.0        21248.0  100.0%

total self time: 21248.0 us
```

## 19.12 Сборка и инструменты

```bash
zig build          # сборка в zig-out/bin/qualisp
zig build test     # 834 теста
zig build run      # запуск
zig fmt src        # форматирование
bash bench/run.sh  # замеры (эталон — Rust -O3)
gh workflow run CI # ручной CI: Linux и Windows
```

`build.zig` импортирует `build.zig.zon` для `manifest.version`; для Windows выставляет `exe.stack_size = 256 * 1024 * 1024`. `zig build test`: **834 теста** на Linux и 834 на Windows (3 POSIX-FFI теста пропускаются, 1 Windows-FFI проверяет kernel32/msvcrt и стековые аргументы). Регистрация — `src/tests.zig`, 24 файла `_test.zig`; крупнейшие: `jit_test` (170), `checker_test` (88), `session_test` (82), `ns_test` (66), `ns_expr_test` (58), `ns_lang_test` (44), `lexer_test` (43), `printer_test` (36). Системы сюит `deftest`/`run-tests` в реализации нет — тесты плоские.

Релизные артефакты: `qualisp-0.19.15-linux-x86_64`, `qualisp-0.19.15-windows-x86_64.exe`, `qprof-0.19.15-linux-x86_64`, `qprof-0.19.15-windows-x86_64.exe/.pdb`, `SHA256SUMS.txt`; `--version` печатает `qualisp 0.19.15`.

Платформы: Linux x86-64 — основная; Windows x86-64 (`gnu`/`msvc`) — с SysV-ABI для JIT-кода; отличаются только W^X (`VirtualAlloc`/`VirtualProtect`), файловые хэндлы-токены поверх `std.Io.File` и резерв стека в PE. FFI: POSIX — `dlopen`/`dlsym`, Windows — `LoadLibraryA`/`GetProcAddress` и Win64-трамплин (регистры по позиции аргумента, 32 байта shadow space, стек для позиций ≥ 4).

Замеры ReleaseFast: fib 42 мс, matmul 16 мс, reduce 6 мс, ns-if graph 58 мс (эталон — Rust `-O3`). Относительные изменения по этапам оптимизации: fib(30) −11 %, loop-id −9 %, взаимная рекурсия с разными кадрами −8 %.

QSRD — бинарный формат с magic `"QSRD"`, u16 major/minor, таблицей секций с crc32 и секциями `meta`, `symbols`, `forms`, `data`, `model`; `save-weights`/`load-weights` работают с секцией `model`. Читатель `tools/qsrd` на Rust: `inspect`, `verify`, `cat`, `params`, `symbols`, `extract`; есть конформанс-корпус.

## 19.13 Ключевые константы

| Параметр | Значение |
|---|---|
| Арена кадров вызова | `1024 * 1024` слотов (~24 МБ адресного пространства); LIFO, исчерпание — типизированная ошибка |
| Резерв машинного стека | 256 МБ (`raiseStackLimit`, `stack_size` в PE) |
| Воркеры компиляции графа | `min(cpu_count, 8)` |
| AVX2 | 8×f32 / 4×f64 |
| Reductions | 8 независимых f64-полос, фиксированный порядок сведения |
| Source map | ~24 байта на форму, `MapEntry{start,end,span}` |
| DType | `f16`, `f32`, `f64`, `u8` |
| Модули stdlib | 11 (`probe.qlsp` — только в test-сборках) |
| Тесты | 834 (`src/tests.zig`, 24 файла `_test.zig`) |
| QSRD | magic `"QSRD"`, u16 major/minor, crc32 на секцию |

## 19.14 Известные ограничения

- **GPU/Device нет.** Вычисления идут на CPU x86-64 через JIT и AVX2-ядра; других архитектур и бэкендов нет.
- Deopt не реализован: страницы не специализированы, helper-путь и есть generic-путь; guard'ы ставят `RuntimeError`, а не перекомпилируют страницу.
- `:any` запрещён; HM-вывод между функциями не делается; пользовательских перегрузок нет.
- Writable views и `t-slice` отсутствуют; `reshape`/`transpose`/`t-rows`/`t-select-rows` копируют.
- Autograd: `grad!` привязан к последнему запуску графа в форме; в графе нет `tref`/`tset!`, `pow`/`sqrt`/`sin`/`cos`/`atan`/`argmax`; reductions — только float; нет вложенных графов, рекурсии и data-dependent control flow; нет автоматических CSE/DCE/FUSION.
- Параметры, не участвовавшие в запуске, сохраняют старый градиент; `clear` не освобождает граф и JIT-страницу до teardown.
- Struct-инстансы и `Result` живут до `Runtime.deinit`; ошибки не разматываются через нативный C-стек.
- `eval` возвращает `:syntax`/`:atom` консервативно; конкретика через `the`.
- В релизе только `qualisp` и `qprof`: других инструментов в поставке нет. Отдельного модуля `async` нет — только примитивы `future`/`await` и тип `(:future T)`; `FFI` и `NS` — не импортируемые модули.
- Обновление `defvar`/`setq` глобала внутри `future` — runtime-ошибка; возвращать future из future нельзя.

## 19.15 Модель памяти (кратко)

Полное описание — гл. 5; здесь — сводка механики, на которой держится рантайм.

**Модель (SPEC/AGENTS):** три пула HibLin — `ConsCellPool` (стек скоупа: cons, rope-узлы, замыкания, структуры), `StableMemory` (`defvar`, веса, глобалы, JIT-страницы; освобождение — `clear`/teardown), `TensorBufferPool` (только HLO, exact-size free lists). GC, RC и аренщика общего назначения нет.

**Реализация в Zig:** `Store` — bump-регион фиксированных cons-ячеек и байтов с `mark()/restore()`; `Runtime` держит `world`/`scratch` и плотный массив глобалов; `cloneToWorld` — глубокая копия; кадры вызовов — отдельная bump-арена (§19.5). Исключения модели: бессмертные `nil`/`t`/символы; stable живёт до `clear`/teardown; циклы в мутабельных stable-структурах не собираются; JIT-страницы — выбрасываемый кэш; `eval` — единственная граница статики.

Stable никогда не ссылается на scratch: запись scratch-значения в stable — глубокая копия. Замыкание захватывает локальные биндинги копией, глобалы — live. `setq` перемещает значение в биндинг; `defvar` копирует. Возврат из скоупа — откат указателя `Store` и копия значения в родительский скоуп.

---

Подробности по подсистемам: гл. 6 (тензоры), гл. 7 (autograd), гл. 10 (HLO), гл. 20 (JIT copy-and-patch), гл. 21 (образы QSRD). Тесты — 834 плоских теста в 24 файлах `_test.zig`, зелёные по `zig build test` на Linux и Windows.
