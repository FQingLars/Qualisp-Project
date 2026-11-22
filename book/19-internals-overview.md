# Глава 19. Устройство компилятора

Краткая экскурсия во внутренности Qualisp релиза 0.19.16 (реализация на Zig): как исходник превращается в программу, какие подсистемы за этим стоят и как они делят ответственность. Полные решения по языку — в `SPEC.md`, архитектурная сводка и дорожная карта — в `AGENTS.md`, публичные API подсистем — в `src/CODE.md` и `src/*/CODE.md` дерева разработки; контракт LSP — в `docs/lsp.md`, замеры JIT — в `docs/monomorphization.md`.

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

Все аннотации — параметры, поля структур, результат функции (последняя форма тела), сигнатура метода трейта и тип `defvar` — разбирает один парсер `src/param_spec.zig` (`normalizeParams`, `splitHeader`, `splitMethod`, `splitDefvar`). Dotted pair — данные, позиционная запись в ядре не принимается; `defmacro` — единственное исключение и пользуется своей формой.

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
├── printer.zig       Value → читаемый текст; структуры — `#<tag …>`
├── program.zig       единый корень top-level форм
├── runtime.zig       глобалы, Func/Env, кадры вызовов, страницы; Error/CallFrame
├── form.zig          cons/list, quasiquote
├── syntax.zig        :syntax-операции
├── types.zig         Type, парсер, assignable
├── param_spec.zig    единый парсер парных аннотаций: normalizeParams,
│                     splitHeader, splitMethod, splitDefvar
├── checker.zig       статический typechecker ядра
├── overload.zig      ключи и разрешение перегрузки
├── macros.zig        defmacro, expander, gensym
├── eval.zig          evalValue: expand→check→emit→run
├── jit/              emitter (raw fixnum-слоты, emitRaw), memory (W^X),
│                     helpers, cache, stencils, abi
├── prims/            list, math, string, io, type, error
├── object.zig        struct/trait логика
├── tensor/           Tensor + eager-операции (параллельные ветки) + TensorBufferPool
├── kernel_pool.zig   ленивый пул вычислительных ядер; воркеры без Runtime
├── hlo/              граф (sin/cos/sqrt/atan, mask-ops), trace,
│                     forward/backward, compile/pool/fused, kernels
├── ns/               atom, expr, bdd, coverage, router, compaction, form, plan;
│                     параллельные band/branch/counting-sort/gather/scatter
├── ffi/              C-типы, trampoline, загрузка библиотек
├── image/            бинарный формат QSRD
├── profile.zig       трассировка Chrome Trace JSON
├── lsp_abi.zig       libqualisplsp: analysis-only C ABI для LSP
├── stdlib.zig        13 встроенных модулей (через @embedFile)
├── stdlib/*.qlsp     core, error, string, collections, test, ml, dl, symbolic,
│                     argparse, tablet, io, datetime, neuro
│                     (+ probe.qlsp, typepairs_probe.qlsp — только тесты)
├── tests.zig         регистрация 910 тестов
├── *_test.zig        32 файла с тестами
└── test_support.zig  общий harness
tools/qsrd/           читатель QSRD (inspect/verify/cat/params/symbols/extract)
tools/qprof/          агрегатор трасс self/inclusive по именам
tools/lsp/build.sh    сборка libqualisplsp + qualisp-lsp в zig-out/lsp/
tools/qualisp-lsp/    Rust-сервер: stdio JSON-RPC над C ABI
editors/vscode/       расширение VSCodium/VS Code (чистый JS), VSIX
experiments/          source-to-source полигон (defun*)
docs/lsp.md           контракт C ABI, фичи LSP, владение буферами
docs/monomorphization.md  замеры специализации-lite и M3a (unboxed locals)
```

Stdlib — Qualisp-код, вшитый в бинарник через `@embedFile`; prelude — `core` **и `error`** (загружаются и импортируются при старте), остальные 11 модулей — лениво при первом `import`. `probe.qlsp` и `typepairs_probe.qlsp` — test-only фикстуры, вшиваются только в test-сборки. В каждом подкаталоге `src/X/` лежит `CODE.md` с API и ограничениями подсистемы. LSP и IDE живут в дереве (`src/lsp_abi.zig`, `tools/qualisp-lsp/`, `editors/vscode/`), но их готовые бинарники в релизные ассеты не входят.

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

`(:result T)` = `ok T | err Error`; `throw : :a → :never`; `catch-error : (:func () T) → (:result T)`. Модуль `error` в prelude определяет две структуры парными полями:

```lisp
(defstruct call-frame
  ((name . :symbol) (file . :string) (line . :fixnum) (form . :string)))

(defstruct error
  ((message . :string)
   (cause . (:option :atom))
   (traceback . (:option (:list (:struct call-frame))))
   (kind . :symbol)))
```

`kind` — интернированный символ категории: `'runtime`, `'parse`, `'type`, `'io`, `'user` (`runtime-error`, …), сравнение через `eq`. Рантайм резолвит публичные теги `ERROR`/`CALL-FRAME` через `struct_aliases` на приватные теги модуля. `throw` кладёт брошенное значение в `cause`, `message` — его печатное представление, `kind = USER` для `throw` и `RUNTIME` для рантайм-ошибки. Кадры `call-frame` наполняются `name/file/line/form` из source map Q4a (`FramePlace`); без места — `file="<unknown>"`, `line=0`, `form=""`. `result-error : (:result T) → (:struct error)` — примитив, достающий структуру из неудачного результата (`ql_result_error`); `format-error : (:struct error) → :string` (`ql_format_error`) собирает отчёт:

```text
Error: <message> (<kind в нижнем регистре>)
  cause: <если причина отличается от сообщения>
  at <name>  <file>:<line>  in <form>
```

Кадры без источника печатаются без фиктивного `<unknown>:0`, внутренние теги (`ERROR::…`) в отчёт не попадают. Принтер структур печатает `#<tag …>` вместо `#<struct>`; непойманный `throw` в CLI даёт `file:line:col: message`. Ошибки компиляции/типа — не `Error`-значения, а отказ с диагностикой со спаном. Хвостовые вызовы переименовывают верхний кадр traceback'а.

Runtime-ошибка: хелпер зовёт `bailAt(@returnAddress(), msg)`, сообщение сохраняется в `Runtime.error_message`, позиция — в `error_location`. Source map — `CodePage.map: []MapEntry{start,end,span}` плюс имя файла: запись на каждую cons-форму, вложенные формы дают вложенные записи, `lookup` берёт самую внутреннюю, ~24 байта на форму. Адрес возврата указывает **после** `call`, поэтому lookup идёт по `offset-1`. CLI печатает `file:line:col: message`. Ошибки не разматываются через нативный C-стек, только через JIT-кадры.

## 19.6 Special forms

Ядро чекера и эмиттера: `quote`, `if`, `progn`, `let`, `setq`, `while`, `lambda`/`\`, `defun`, `defvar`, `and`, `or`, `cond`, `the`, вызовы; `funcall`/`apply`, `read-from-string`/`write-to-string`; `defmacro`, `macroexpand-1`, `macroexpand`, `macroexpand-all`, `gensym`; `defstruct`, `slot-value`, `set-slot`, `deftrait`, `impl`; `import`, `import-from`; `defgraph`, `defuse`; `ns-if`; `tensor`; `throw`, `catch-error`; `eval`. Примитивы (`+ - * /`, сравнения, `car`/`cdr`/`cons`, `print`, `map`/`filter`/`reduce`, тензорные ядра, `result-error`/`format-error`, `ffi-load`/`ffi-defun`, файловые и QSRD-примитивы, `future`/`await`) живут в реестрах, а не в спецформах.

- `defun` требует парных аннотаций параметров `(p . :T)` и результата — последней формы тела `(expr . :ret)`; у `lambda` аннотируются только параметры, результат выводится. `defvar name (value . :type)`; поля структур `(field . :T)`, метод трейта `(name ((self . :self)) . :ret)`. `&optional` внутри тела — `(:option T)`, `&rest` — `(:list T)`; `let` — параллельные биндинги; `let*` отсутствует. `defmacro` — исключение: `(defmacro name (form :syntax) :syntax …)`. Пользовательских перегрузок нет: повторный `defun` с тем же именем — ошибка. `:any` запрещён.
- Сигнатуры всех литеральных top-level `defun` публикуются **до** проверки первой формы (`declareFunctions`), поэтому прямая ссылка вперёд и взаимная рекурсия типизируются; определение исполняется по порядку, вызов до определения — runtime-ошибка.
- Тела `defgraph`/`defuse` раскрываются отдельно (`Session.expandBody`) и клонируются в world: иначе тело `defuse` висело бы на scratch. Тело макроса проверяется в живом окружении (чекер берётся из `Runtime.checker`), поэтому макрос может звать хелперы модуля.
- `eval : :syntax → :a` — проверяемый каст; `:a` выводится из контекста; форма заново проходит expand/typecheck/emit, несовпадение типа — runtime-ошибка. Единственная документированная граница статики.

### Модули

Модуль — файл с `(export name …)` в конце; приватные хелперы не экспортируются. `(import m)` открывает все экспорты, `(import-from m (a b))` — только перечисленные, каждое имя проверяется по `module_exports`; повторный импорт идемпотентен, конфликт — `ImportConflict`. Приватные id создаются через `SymbolTable.internPrivate`: имя лежит в `names`, но не в `map`, поэтому `MODULE::NAME` нельзя набрать в исходнике. `Session.expose` алиасит публичное имя на приватное. `load`/`import` обрабатываются только на top-level. `core` и `error` — предустановленный prelude (`Session.loadPrelude` грузит и импортирует оба), остальные 11 модулей грузятся лениво.

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

Быстрые пути: `elementwise`/`map` — `@Vector` 8×f32 / 4×f64 без f64-раундтрипа; `matmul` по `i-k-j` с `@Vector`; reductions блоками — 8 независимых f64-полос, фиксированный порядок сведения (результат детерминирован, но не «слева направо»). `u8`-деление запрещено статически; `:u8` в `tsum`/`tmax` даёт `:fixnum`. Вычислительные ядра параллелятся через `src/kernel_pool.zig`: ленивый пул (воркеры создаются на первом достаточно большом вызове, `max(1, min(cpu-1, 8))`), у воркеров нет `Runtime`, вызывающий выделяет непересекающиеся выходные диапазоны, общие `frame.pool`, `grads[]`, `rt.trace` и scope-арены остаются на одном потоке, вложенный параллелизм отключён. Маленькие входы остаются последовательными, `QUALISP_SERIAL=1` отключает пул. Параллельны `elementwise`/`map` (порог ~256K), `matmul` по строкам, `reshape`/`transpose` по строкам, `tsum`/`tmean` по 8-полосным partial'ам с фиксированным порядком сведения — результат побайтно совпадает с последовательным. Пока последовательные: `max`, `softmax`, `argmax`, `reduceToShape`.

`ones-like`, `tmax-mask`, `relu-mask`, `reshape-like` — forward-only строительные блоки символьного градиента (`tensor.zig`: `onesLike`, `maskMax`, `maskPositive`, `reshapeLike`), сохраняющие форму тензора; `reshape-like` требует совпадения числа элементов.

`TensorBufferPool` — exact-size free lists по `(len, alignment)`, реализация `std.mem.Allocator`; буферы переиспользуются, финальный выход копируется в scratch. `Graph.peak_bytes` — оценка пика, жёсткий cap не enforced.

Устройства нет: `Owner` — `scratch`/`world`/`pool`, `device` в перечислении отсутствует, все ядра — C-ABI функции на x86-64 с AVX2. Writable views и `t-slice` отсутствуют; `reshape`, `transpose`, `t-rows`, `t-select-rows`, `t-column` копируют. Печать — `#<tensor dtype (dims)>` без содержимого.

## 19.8 Autograd

`(param t)` помечает float-тензор `requires_grad` (`:u8` запрещён); флаг переживает `cloneToWorld`. `defgraph` трассирует тело один раз на нулевых placeholder-тензорах: управление статическое, `tref`/`tset!` в графе не поддерживаются, константы (`tensor`, `zeros`/`ones`/`randn` в теле) — stable. Вызов компилируется в `(%GRAPH-RUN gid ret arg…)`.

`ql_graph_run` сохраняет последний запуск в `Runtime.last_run` (сбрасывается в конце top-level формы); forward с `frame.retain = 1` не освобождает активации — все лежат в per-run `TensorBufferPool`. `(grad! loss)` сеет единицы формы выхода и идёт по узлам с конца, читая forward-значения из `run.values` — **без replay**, O(n) по узлам и O(n) по активациям до `grad!`/конца формы. Градиенты накапливаются в scratch; для `param`-входов копируются в stable и кладутся в `Tensor.grad`; `(grad-of t)` возвращает stable-градиент или нули той же формы/dtype.

Backward-правила: `add`, `sub`, `mul`, `div` (reduce по broadcast), `neg`, `matmul`, `reshape`, `transpose`, `relu`, `exp`, `log`, `tanh`, `sin`, `cos`, `sqrt`, `atan`, `softmax`, `sum`, `mean`, `max`; `sum` размножает grad, `mean` делит на число элементов, `max` отдаёт grad всем максимумам. `sin/cos/sqrt/atan` получили forward и backward во всех слоях HLO: узел `Op` (`graph.zig`), traced execution, compiled- и backward-ядра (`kernels.zig`, `bwd_kernels.zig`, `compile_backward.zig`); аналитические производные сверены с autograd по каждому оператору. Backward компилируется в обратную последовательность вызовов `ql_bwd_*` (x86-64, W^X); интерпретатор — fallback при ошибке компиляции.

Рядом с autograd работает символьный reverse-mode `(grad term var)` из `symbolic`: это градиент **скалярного лосса `(tsum term)`**, а не поэлементная производная `diff`. `grad = (simplify (grad-step term var 1.0))`; правила покрывают `+ - * / ^` (num exponent), `tsum`/`tmean`/`tmax`, `matmul` по обоим аргументам, `transpose`, `reshape`, `relu`, 1-D `softmax`, унарные `sin…atan` и операции-спутники `ones-like`/`tmax-mask`/`relu-mask`/`reshape-like`, которые сохраняются отдельными тензорными узлами и не склеиваются `simplify`. `term` транслирует все эти операции; `(^ base n)` над составным основанием связывает подтерм один раз через `let` (gensym `pow`), а не дублирует его. `diff` знает унарные производные `sin/cos/exp/log/sqrt/tanh/atan`, n-арные `+`/`*`/`-` и унарный минус, но на `relu` и `matmul` даёт внятный отказ с советом `grad`.

**Ограничения.**

- `grad!` работает только с последним запуском графа в текущей top-level форме; не проверяет, что аргумент — именно его выход.
- Нет `tref`/`tset!`/`pow`/`argmax` в графе; reductions в графе — только float; `defgraph` с `:fixnum` фактически недостижим. `sqrt`/`sin`/`cos`/`atan` в графе доступны.
- Нет вложенных `defgraph`, рекурсии и data-dependent control flow; нет автоматического fusion/CSE/DCE — fusion только через `defuse`.
- Параметры, не участвовавшие в последнем запуске, сохраняют старый градиент; `grad-of` никогда не считанного параметра — нули.
- `clear` имени снимает callable, но граф/JIT-страница живут до teardown.

## 19.9 HLO-пайплайн и copy-and-patch JIT

### Узлы и операции

`Op` из `src/hlo/graph.zig`: `constant, identity, add, sub, mul, div, neg, matmul, reshape, transpose, relu, exp, log, sin, cos, sqrt, atan, tanh, softmax, sum, mean, max, ones_like, tmax_mask, relu_mask, reshape_like, ns_if`. Узлы с арностью > 2 разворачиваются в бинарные. `ns_if` несёт подграфы ветвей (`Graph.subgraphs`, по одному параметру-payload на ветвь). Граф — производный одноразовый артефакт, мастер — S-выражение `defgraph`.

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

**M3a — unboxed fixnum-локальные.** Слоты кадра получили вид `value | raw_int`: параметры и `let`-биндинги с доказанным `:fixnum` живут в payload-области без тега. `emitRaw` эмитит арифметику, унимнус и сравнения над raw-слотами; боксинг происходит на границе — передача в вызов, возврат, захват замыканием. Функции с `lambda` в теле отключают raw (`allow_raw = false`), чтобы замыкание не читало слот как `Value`. Замер A/B ReleaseFast, 2M итераций (`docs/monomorphization.md` §4.7):

| Сценарий | До | После |
|---|---|---|
| `loop1` (один `+`) | 10.22 нс | 3.15 нс (3.2×) |
| `local_arith` (два `+`) | 14.94 нс | 4.49 нс (3.3×) |
| `arith` (глобалы) | 17.36 нс | 17.59 нс (без изм.) |
| `calls` (вызов в цикле) | 33.51 нс | 33.1 нс |
| `fib` | 42 мс | 42.5 мс (call-bound) |

Глобалы и вызовы остаются боксованными; тесты — `src/unboxed_test.zig` (7). **Специализация-lite** (карта типов на узлах чекера, типовое окружение эмиттера, unguarded fixnum-арифметика) измерена и откачена: 18.15 → 18.26 нс на arith-loop, 33.51 → 32.68 нс на calls-loop, fib 42 → 43 мс — tag-guard'ы не бутылочное горлышко, время уходит на 24-байтные `Value`-загрузки/сохранения и механизм вызова. JIT на данный момент мономорфным не является; реальный выигрыш дал только unboxing локальных.

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

Параллельный путь роутера лежит в `src/ns/{router,compaction}.zig` поверх того же `kernel_pool`: `fillBand`, выбор ветки, `countingSort` (гистограммы и курсоры по chunk'ам дают stable-порядок) и `gatherRows`/`scatterRows`. Ошибки `NsIfUncovered`/`NsIfOverlap` детектируются после параллельного прохода. Замер A/B ReleaseFast, 16 ядер, min из 11 (`AGENTS.md` P9p): ns-if eager 84.9 → 79.0 мс, graph 59.9 → 60.4 мс; остаток одной нити в ns-if — сами branch-лямбды (4 ветки), для них нужен per-worker `Runtime`/пул.

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

**LSP/IDE — инструмент из исходников.** `src/lsp_abi.zig` собирается как `libqualisplsp` (Zig, `addLibrary` + `link_libc`, rpath `$ORIGIN`) и экспортирует analysis-only C ABI: `ql_analyzer_new/free`, `set_source`, `check`, `diagnostics`, `document_symbols`, `definition`, `hover`, `completion`, `document_highlight`, `semantic_tokens`, `register_stdlib` и парные `ql_*_free`. Возврат `int`: `0` — ok, `1` — диагностики/не найдено, `-1` — внутренняя ошибка; позиции line 1-based, column/offset — байты (перевод UTF-16 ↔ bytes делает Rust). Буферы освобождаются парной функцией из той же библиотеки, строки — NUL-терминированные копии.

`tools/qualisp-lsp/` — Rust-сервер stdio JSON-RPC, зависимости только `libloading`/`serde`/`serde_json`, фрейминг вручную; библиотека ищется по `$QUALISP_LIB`, каталогу бинарника, `../lib/` и системным путям (`libqualisplsp.so` / `qualisplsp.dll`). Параллельные запросы сериализуются мьютексом: один анализатор на процесс. Анализатор не исполняет формы верхнего уровня документа — только `read → macroexpand → typecheck`; `emit`/`run` не вызываются, код исполняется лишь в телах макросов. Работают diagnostics, definition (по всем `.qlsp` воркспейса и встроенным модулям), hover (статический тип, исходная форма и комментарий над ней), documentSymbol, completion, documentHighlight, semanticTokens (7 типов), полная пересылка текста на `didChange`. Инкрементального парсера нет. Сборка: `bash tools/lsp/build.sh` → `zig-out/lsp/`; расширение VSCodium — чистый JS (`editors/vscode/`), мок-тест `node editors/vscode/test/smoke.js`, самодостаточный VSIX — `python3 editors/vscode/package-vsix.py`. 12 Rust-интеграционных тестов и 16 Zig ABI-тестов; готовый `qualisp-lsp`, `.so/.dll` и `.vsix` в релизные ассеты **не входят**.

## 19.12 Сборка и инструменты

```bash
zig build          # сборка в zig-out/bin/qualisp
zig build test     # 910 тестов
zig build run      # запуск
zig fmt src        # форматирование
bash bench/run.sh  # замеры (эталон — Rust -O3)
gh workflow run CI # ручной CI: Linux и Windows
```

`build.zig` импортирует `build.zig.zon` для `manifest.version` (`0.19.16`, `minimum_zig_version = 0.17.1`); для Windows выставляет `exe.stack_size = 256 * 1024 * 1024`. `zig build test`: **910 тестов** на Linux и 910 на Windows (3 POSIX-FFI теста пропускаются, 1 Windows-FFI проверяет kernel32/msvcrt и стековые аргументы). Состав счётчика: 909 именных `test "…"` плюс один анонимный блок `test { … }` в `src/tests.zig:1`. Регистрация — `src/tests.zig`, 32 файла `_test.zig` (добавлены `parallel_test`, `experiments_test`, `type_pairs_test`, `error_module_test`, `unboxed_test`, `neuro_dumb_test`, `stdlib_datetime_test`, `lsp_abi_test`); крупнейшие: `jit_test` (170), `checker_test` (88), `session_test` (82), `ns_test` (66), `ns_expr_test` (58), `ns_lang_test` (44), `stdlib_dl_test` (40), `printer_test` (36), `lexer_test` (34), `stdlib_symbolic_test` (29), `stdlib_neuro_test` (19), `lsp_abi_test` (16), `neuro_dumb_test` (15), `stdlib_datetime_test` (6), `error_module_test` (6), `unboxed_test` (7), `parallel_test` (3). Системы сюит `deftest`/`run-tests` в реализации нет — тесты плоские. Отдельно 12 Rust-интеграционных тестов LSP и мок-тест расширения `node editors/vscode/test/smoke.js`.

Релизные артефакты — ровно 6 файлов: `qualisp-0.19.16-linux-x86_64`, `qualisp-0.19.16-windows-x86_64.exe`, `qprof-0.19.16-linux-x86_64`, `qprof-0.19.16-windows-x86_64.exe`, `qprof-0.19.16-windows-x86_64.pdb`, `SHA256SUMS.txt`; `--version` печатает `qualisp 0.19.16`. LSP-сервер, `libqualisplsp` и VSIX в ассеты не входят — их собирают локально.

Замеры ReleaseFast этапа 0.19.16: unboxed-локальные дают 3.2–3.3× на локальных fixnum-циклах (§19.9); параллельный matmul — 15.9 → 7.6 мс (2.10×) на 16 ядрах (§19.7); общий fib остаётся call-bound (~42 мс).

Платформы: Linux x86-64 — основная; Windows x86-64 (`gnu`/`msvc`) — с SysV-ABI для JIT-кода; отличаются только W^X (`VirtualAlloc`/`VirtualProtect`), файловые хэндлы-токены поверх `std.Io.File` и резерв стека в PE. FFI: POSIX — `dlopen`/`dlsym`, Windows — `LoadLibraryA`/`GetProcAddress` и Win64-трамплин (регистры по позиции аргумента, 32 байта shadow space, стек для позиций ≥ 4).

Дополнительные абсолютные замеры бенчмарков (ReleaseFast, сравнение с Rust `-O3`) — в `bench/README.md`; в книге используются только замеры из `AGENTS.md` (P9p, E11) и `docs/monomorphization.md` (§4.6–4.7).

QSRD — бинарный формат с magic `"QSRD"`, u16 major/minor, таблицей секций с crc32 и секциями `meta`, `symbols`, `forms`, `data`, `model`; `save-weights`/`load-weights` работают с секцией `model`. Читатель `tools/qsrd` на Rust: `inspect`, `verify`, `cat`, `params`, `symbols`, `extract`; есть конформанс-корпус.

## 19.13 Ключевые константы

| Параметр | Значение |
|---|---|
| Арена кадров вызова | `1024 * 1024` слотов (~24 МБ адресного пространства); LIFO, исчерпание — типизированная ошибка |
| Резерв машинного стека | 256 МБ (`raiseStackLimit`, `stack_size` в PE) |
| Воркеры компиляции графа | `min(cpu_count, 8)` |
| Воркеры `kernel_pool` | `max(1, min(cpu-1, 8))`, лениво; `QUALISP_SERIAL=1` отключает пул |
| AVX2 | 8×f32 / 4×f64 |
| Reductions | 8 независимых f64-полос, фиксированный порядок сведения |
| Source map | ~24 байта на форму, `MapEntry{start,end,span}` |
| DType | `f16`, `f32`, `f64`, `u8` |
| Модули stdlib | 13 (`core` и `error` — prelude; `probe.qlsp`/`typepairs_probe.qlsp` — только в test-сборках) |
| Тесты | 910 (`src/tests.zig`, 32 файла `_test.zig`) |
| QSRD | magic `"QSRD"`, u16 major/minor, crc32 на секцию |

## 19.14 Известные ограничения

- **GPU/Device нет.** Вычисления идут на CPU x86-64 через JIT и AVX2-ядра; других архитектур и бэкендов нет.
- Deopt не реализован: страницы не специализированы, helper-путь и есть generic-путь; guard'ы ставят `RuntimeError`, а не перекомпилируют страницу. Специализация-lite измерена и откачена; рабочий выигрыш дают только unboxed fixnum-локальные (M3a).
- `:any` запрещён; HM-вывод между функциями не делается; пользовательских перегрузок нет.
- Writable views и `t-slice` отсутствуют; `reshape`/`transpose`/`t-rows`/`t-select-rows` копируют.
- Autograd: `grad!` привязан к последнему запуску графа в форме; в графе нет `tref`/`tset!`, `pow`/`argmax` (`sqrt`/`sin`/`cos`/`atan` есть, включая backward); reductions — только float; нет вложенных графов, рекурсии и data-dependent control flow; нет автоматических CSE/DCE/FUSION. `diff` не дифференцирует `relu`/`matmul` — там нужен `(grad term var)`.
- Параметры, не участвовавшие в запуске, сохраняют старый градиент; `clear` не освобождает граф и JIT-страницу до teardown.
- Struct-инстансы и `Result` живут до `Runtime.deinit`; ошибки не разматываются через нативный C-стек.
- `eval` возвращает `:syntax`/`:atom` консервативно; конкретика через `the`.
- Релизные ассеты содержат только `qualisp` и `qprof` (плюс Windows-версии, PDB и `SHA256SUMS.txt`): LSP поставляется в исходниках и собирается локально. Отдельного модуля `async` нет — только примитивы `future`/`await` и тип `(:future T)`; `FFI` и `NS` — не импортируемые модули. Параллельные ядра не покрывают `max`/`softmax`/`argmax`/`reduceToShape`; воркеры не имеют `Runtime`, а вложенный параллелизм на них отключён.
- Обновление `defvar`/`setq` глобала внутри `future` — runtime-ошибка; возвращать future из future нельзя.

## 19.15 Модель памяти (кратко)

Полное описание — гл. 5; здесь — сводка механики, на которой держится рантайм.

**Модель (SPEC/AGENTS):** три пула HibLin — `ConsCellPool` (стек скоупа: cons, rope-узлы, замыкания, структуры), `StableMemory` (`defvar`, веса, глобалы, JIT-страницы; освобождение — `clear`/teardown), `TensorBufferPool` (только HLO, exact-size free lists). GC, RC и аренщика общего назначения нет.

**Реализация в Zig:** `Store` — bump-регион фиксированных cons-ячеек и байтов с `mark()/restore()`; `Runtime` держит `world`/`scratch` и плотный массив глобалов; `cloneToWorld` — глубокая копия; кадры вызовов — отдельная bump-арена (§19.5). Исключения модели: бессмертные `nil`/`t`/символы; stable живёт до `clear`/teardown; циклы в мутабельных stable-структурах не собираются; JIT-страницы — выбрасываемый кэш; `eval` — единственная граница статики.

Stable никогда не ссылается на scratch: запись scratch-значения в stable — глубокая копия. Замыкание захватывает локальные биндинги копией, глобалы — live. `setq` перемещает значение в биндинг; `defvar` копирует. Возврат из скоупа — откат указателя `Store` и копия значения в родительский скоуп.

---

Подробности по подсистемам: гл. 6 (тензоры), гл. 7 (autograd), гл. 10 (HLO), гл. 18 (шпаргалка), гл. 20 (JIT copy-and-patch), гл. 21 (образы QSRD). Тесты — 910 плоских тестов в 32 файлах `_test.zig`, зелёные по `zig build test` на Linux и Windows.
