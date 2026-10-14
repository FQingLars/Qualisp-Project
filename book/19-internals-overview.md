# Глава 19. Устройство компилятора

Краткая экскурсия во внутренности Qualisp v2.8.6 (серия v2.4–v2.8: удаление легаси-AOT, JIT copy-and-patch, образы, единый HLO-стенсил-эмиттер, модуль Tablet, HibLin OOM-фиксы, эвикция пулов, контракты времени жизни, TCO, кэш раскрытия макросов, DEBUG-TRACE, подсчёт ссылок на стабильные ячейки, LSan-гейт утечек, rebrand QLISP → Qualisp и единая модель владения RC/пул). Полное описание — в `ARCHITECTURE.md` дерева разработки.

## 19.1 Пайплайн

```text
Исходник → CharStream → Readtable (reader) → SExpr-дерево в ConsCellPool (гл. 5)
  → MacroExpander (backquote, defmacro, compiler-macros, MACROLET)
  → Interpreter::eval (special forms, применение функций, примитивы)
    → Tensor-операции → GradientTape (запись autograd)
    → GradientTape::backward (reverse-topo AD, blocked nodes для NS-IF)
    → горячее замыкание → jit::compile_closure → copy-and-patch → RX-страница (гл. 20)
  → HloGraph (START-TRACE … HLO-COMPILE)
    → compile_hlo_stencil → AVX2-стенсилы → W^X-страница (гл. 10)
    → NS-router (NS-IF/NS-GRAD! → per-sample routing + guilt detector; ROUTE-нода в HLO)
                     ↓
                LSP Server ←→ Editor
```

**Инвариант одной формы (v2.4.0).** S-выражение — единственная авторитетная форма программы. Оба «компилятора» — JIT замыканий и HLO-стенсил-эмиттер — потребляют S-выражения (и трассированный HLO-граф, который сам round-trip-ится в S-выражение, §10.8) и строят нативные страницы одним проходом, без промежуточного IR и без LLVM. Нативный код — выбрасываемый кэш: `DISCARD-JIT` и инвалидация по эпохе макросов пересобирают его из master-копий без изменения поведения.

## 19.2 Структура исходников

```
src/
├── core/          Scope (HibLin), SExpr, Symbol, ConsCellPool, TensorBufferPool, StableMemory, simd.hpp (slab-ядра), image.{hpp,cpp} (образы, гл. 21)
├── reader/        CharStream, Readtable
├── eval/          Interpreter (80+ примитивов), специальные формы
├── types/         Type, TypeChecker (постепенная типизация)
├── macros/        MacroExpander, backquote
├── mop/           ClassObj, ClassRegistry, Instance (CLOS)
├── tensor/        Tensor, GradientTape (F16/F32/F64/I32/I64/U8)
├── gpu/           Device (CPU активен; CUDA/ROCm в плане)
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

Токены: символы интернируются в верхнем регистре через `SymbolTable::instance()` (регистронезависимость); сначала пробуется парсинг числа (int64 → double), иначе — символ; `nil` → синглтон nil, `t` → символ T.

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
        // JITPROG (v2.5.0): нативная RX-страница + master-замыкание + эпоха + счётчик вызовов
        struct { void *fn_ptr; bool is_stencil; SExpr *closure_; uint64_t epoch_, calls_; } native_;
    };
};
```

Все cons-ячейки выделяются в `ConsCellPool`; объекты — в `Scope` с деструктор-трекингом.

Замыкание: параметры + тело + захваченное окружение. Глобальные `defun`/`defmacro`/`defrule` сохраняются в `root_arena_` (stable) — код-как-данные живёт всю программу (гомоиконность рантайма).

## 19.5 Окружение и вычисление

```cpp
struct EnvFrame {
    const EnvFrame *parent;   // лексический родитель (nullptr для global)
    SExpr *bindings;          // alist: ((sym . val) (sym2 . val2) ...)
};
```

Реализация: `EnvFrame` использует C++ `std::map` для bindings (см. `TEMP_ISSUES.md` #2 — планируется SExpr-alist в Phase 12).

Вычисление: атом → сам; символ → поиск переменной; cons → special form или вызов функции. Замыкания захватывают цепочку окружений в плоский alist (`do_lambda` в `interpreter.cpp:572`).

## 19.6 Special forms

Из `sym_*` в `src/eval/interpreter.cpp:53-105`:

`QUOTE, IF, PROGN, LAMBDA(\), LET, LET*, DEFUN, DEFUSE, DEFUSE!, DEFMACRO, MACROLET, DEFVAR, SETQ, COND, WHILE, AND, OR, BACKQUOTE, IMPORT, THE, GENSYM, MACROEXPAND, MACROEXPAND-1, FUNCALL, APPLY, DEFCLASS, MAKE-INSTANCE, SLOT-VALUE, DEFSTRUCT, DEFGENERIC, DEFMETHOD, MATCH, UNIFY, SUBST, AMB-LET, REQUIRE, AMB-COLLECT, ASSERT-FACT, RETRACT-FACT, DEFRULE, QUERY, RUN-RULES, DEFINE-COMPILER-MACRO, DEFHLOOP, NS-IF, NS-GRAD!, TENSOR` — и **80+ примитивов** (арифметика, тензоры, autograd, HLO, FFI, IO, data).

## 19.7 Тензорная подсистема

```text
Tensor: dtype (F16/F32/F64/I32/I64/U8), shape, strides, numel,
        data (TensorBufferPool — exact-size, 0% перерасхода),
        requires_grad, grad, grad_node (лента), m/v/step (Adam)
```

- Поэлементные операции — SIMD: AVX2 (8×f32) на x86_64, NEON (4×f32) на ARM. С v2.3.3 — slab-ядра на raw pointers (`src/core/simd.hpp`: `sum_ps`/`max_ps`/`scale_ps`/`exp_ps` с полиномом AVX2), диспетчеризация по layout, одометр без аллокаций — sum-4096: 343ms → 4.7ms; тайловый transpose 64×64: 158ms → 4.8ms; векторизованный softmax + O(D) backward: 16.5ms → 0.40ms; memcpy-im2col conv2d: 107.8ms → 9.8ms.
- `MATMUL` — `cblas_sgemm` (OpenBLAS); с v2.3.3 — флаги `atrans`/`btrans` (транспонирования внутри GEMM), backward-ноги и шаги SGD/ADAM — на сырых указателях (ag-matmul-1024 31ms < PyTorch 34ms; train-sgd/adam-mlp-64 1.14ms).
- `reshape`/`transpose` — глубокие копии (views нет, гл. 4).
- `TensorBufferPool`: exact-size free list — буфер выделяется ровно под размер тензора, при освобождении возвращается в список своего размера; лимит пула — **256 МБ** (настраивается; в v2.7.4 понижен с 1 ГБ), при промахе точного размера на 3/4 лимита эвиктируется крупнейшая корзина free-list — иначе пул удерживал RSS за мёртвыми буферами (120k разных размеров: 1.13 ГБ → 52 МБ).
- F16: `launch_*_half` через `_Float16` (F16C на x86_64); matmul аккумулирует во float.
- `while` с v2.3.3: результат тела — в паре ping-pong scratch-скоупов (O(1) памяти на итерацию, в root продвигается один раз на выходе); утечки ловит soak-харнесс (`tests/memory_soak.qlsp` + `bench/leak_watch.py`, VmRSS по `/proc`). С v2.7.3 `setq` let-привязки, обновляемой в цикле, **re-home в loop store** (без promote-в-root на каждой итерации: 20k cons-setq — 12.5 ГБ OOM → пик 35 МБ), а un-backpropagated графы ленты сбрасываются на границе формы (30k форм без `GRAD!` — +494 МБ → пик 31.7 МБ).

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

`backward` идёт по dataflow-графу в reverse-topo (потребители раньше производителей) через `result → producer`-карту, накапливая `+=` градиенты в `inputs`. `GradNode::inputs` хранит `Tensor*` (не SExpr) — autograd не завязан на интерпретатор.

Нейросимвольные `NS-IF`/`NS-GRAD!` (см. гл. 14) добавляют «вину»: STE-градиент для роутера, блокировка ошибочных ветвей через `blocked_set_t`.

### Контракт tape v2.3.0 (исправления TBC #26)

С v2.3.0 контракт ужесточён и формализован:

1. **Tape-узлы принадлежат tape.** Аллоцируются в heap, удаляются в `clear()`. До v2.3.0 жили в scratch-блоках (UB: `heap-use-after-free` под ASan при `defvar` промежуточного тензора и `grad!` в следующей форме).

2. **`defvar` промоутит промежуточный тензор.** `defvar`/`setq` клонирует тензор в stable-scope и перенаправляет ребро `result` tape-ноды на stable-копию. Градиент течёт в тензор, который видит пользователь.

3. **`reset_scratch` «запечатывает» незакрытые графы.** Если форма верхнего уровня завершилась без `GRAD!`, и какие-то tape-узлы ссылаются на scratch-тензоры, эти тензоры один раз клонируются в stable, и рёбра переписываются. Обучающие циклы ничего не платят (tape пуст после каждого `GRAD!`).

4. **Биндинги не алиасят литералы.** `let`/`bind_and_eval` копируют скалярные значения (fixnum/flonum/symbol) в свежие ячейки — `setq` больше не мутирует исходные литералы в коде.

5. **Adam-состояние живёт на параметре.** `m`/`v`/`step` аллоцируются в stable-scope на самом параметре (исправление `f61426b`); раньше умирали с итерацией, momentum не работал.

6. **HLO/NS-прокси — строковые хендлы.** Биндинги копируют скаляры, и fixnum-прокси терял идентичность (ключ `trace_nodes_`). Теперь — строковые хендлы `"#hlo-node:<id>"`, детерминированно (уточнено в 2.3.1, #27: биндинг-копии и promote-клоны сохраняют идентичность; `HLO-COMPILE` на чужом хендле — внятная ошибка вместо сегфолта).

7. **Времена жизни градиентов (2.3.1, #28).** При входе в `backward()` на каждый нодовый тензор регистрируется guard (`Scope::track_dtor` + `Tensor::clear_grad`): при смерти scratch-региона `->grad` честно обнуляется — следующий backward не пишет в переиспользованный пул (истёкший grad — настоящий null).

8. **HLOPROG переживает promote (2.3.1, #29).** `clone_to` клонирует обёртку и делит compiled fn (живёт в кэше компилятора всё время процесса) — `defvar prog (HLO-COMPILE ...)` безопасен после сброса scratch.

9. **Владение стабильными ячейками (v2.7.7–v2.8.3).** Компилированная программа владеет своими артефактами (sealed-константы в storage-скопе, клон графа, W^X-страница) и умирает на границе формы, если не привязана к имени; привязка (`defvar`/`setq` через promote) делает её бессмертной, а перепись имени по лесному правилу (v2.8.0) выпускает старое поддерево. Кэш стенсила — non-owning: drop стирает запись перед освобождением, поэтому мёртвая программа не возвращается из кэша. Аудитор `QLISP_AUDIT=1` ловил «второе бессмертие» и сирот якорного реестра на границе формы; **с 2.8.3 якоря-владельцы и аудитор удалены**: стабильная ячейка несёт `refs_`, каждое место разделения берёт `stable_ref`, смерть последнего владельца — итеративный worklist-каскад `stable_release` (строки → malloc, программы → page_free + delete, тензоры → release_tracked). CI-шаг `leak_watch` заменён на ASan-sweep, а с 2.8.4 — на LSan-гейт с `detect_leaks=1` (см. 19.13), флаг `QUALISP_RC_POISON=1` заполняет освобождённые ячейки и валит use-after-release/double-release. Замена стабильного значения посреди формы не каскадит сразу: корень паркуется (`defer_stable_release`, метка эпохи) и умирает на следующем restore скраша или на границе формы (`flush_form_end`) — иначе под live-глобалами каскад убивал тело исполняемой функции (парковка **отменена в 2.8.6**, п. 14). Глобальный кадр замыканиями больше не захватывается: глобалы резолвятся live при вызове, захватываются только локальные кадры. Подробно — гл. 5.5.

10. **Tail-call optimization для self-хвостовых вызовов (v2.8.2).** Хвостовой вызов пере-биндит кадр (параметры копируются в heap-стор, итерация освобождается через `restore`) вместо роста стека. Гейт: при живых tape-узлах TCO для итерации выключается; `let`/`let*` разворачиваются в транполине до 8 уровней, глубже — обычный кадр. Замер: рекурсивные train-эпохи N=100/300/500 — было 34.8/51.2/67.5 МБ против плоского `while` ~25 МБ, стало 27.4 МБ плато.

11. **Кэш раскрытия макросов (v2.8.2).** `MacroExpander` кэширует раскрытие на форму-источник (хит по контенту; запись держит root-клон источника), инвалидация на `defmacro*`/`SLOT-SET`; `macrolet`-протяжения — вне кэша. Это закрыло класс-A утечки scratch: `let*` в цикле на 200k итераций 217 МБ → 22.7 МБ (дифференциал к `let` схлопнулся).

12. **DEBUG-TRACE (v2.8.2).** Кольцевой буфер на 4096 записей (`src/core/trace.{hpp,cpp}`) с категориями `scope/clone/anchor/expand/macro`; примитивы `DEBUG-TRACE`, `TRACE-STATS`, `TRACE-DUMP`, `TRACE-CLEAR`, env `QUALISP_TRACE`. Выключенный трейс стоит один relaxed load, сам буфер не течёт. Точки эмиссии: `reset_scratch` (границы форм), `clone_to` (мост disposable→stable), take/release якорей, кэш раскрытия, `defmacro`/`defcompiler-macro`.

13. **Предупреждение о непрочитанных градиентах (v2.8.5).** `GRAD!` регистрирует параметры с посчитанным градиентом в `grad_pending_`; `GRAD-OF`, `SGD-STEP` и (с v2.8.6) `ADAM-STEP` снимают параметр с учёта. На границе формы (`flush_form_end`) остаток множества — параметры, чьи градиенты никто не прочитал, — печатает предупреждение на stderr и очищается: градиент аллоцирован в scratch-арене формы и уже мёртв, поэтому значение после границы честно нулевое (гл. 7.8).

14. **Единая модель владения (v2.8.6).** RC-контракт v2.8.3 распространён на всю структурную память и вытеснил две страховки. `apply()` берёт `stable_ref` на исполняемое замыкание на время вызова (освобождение под `try/catch` — ref гас и на исключении); `Scope::cancel_dtor` стирает **все** guard'ы объекта, а не первый, и вызывается в момент смерти тензора (`release_dying_wrapper`/`flush_deferred_releases`) — это снимает гонку «замена мира через `LOAD-IMAGE` vs `clear_grad`» и сиротские guard'ы, бившие в освобождённый header. Парковка релизов (`defer_stable_release`) отменена: замена стабильного значения освобождает немедленно. Loop store заменён на промоут в stable + реестр `EscapeLedger` (кто и когда отпускает), пустой per-while `Scope` снесён (он аллоцировал 64 КБ на итерацию); суффикс значения, переживающего цикл, шарится `stable_ref`'ом вместо глубокой копии — это убрало клонирование весов при `(setq lst (cdr lst))`. `clone_to` просит heap-header для стабильных тензоров (`heap_header`), иначе 144 Б мёртвого header'а на промоут оставались в `stable_allocs_` навсегда; флаг проставляется явно, потому что сырые stable-тензоры (Adam m/v, константы графа) ячейки и пути смерти не имеют. TCO шарит **уже стабильные** параметры под rc (обход стабильного списка 3000: 193 → 35 мс), refs гасит scoped-guard'ом на любом выходе из цикла. Дорожная карта — `TODO_RC.md` (P0–P7). Гейты: release 92/92, ASan `detect_leaks=1` — 92/92 чисто впервые за серию. Подробно — гл. 5.5.

## 19.9 HLO-пайплайн

Граф из узлов `HloOp` (см. `src/hlo/graph.hpp`): `PARAMETER, CONSTANT, DOT, ADD, MUL, RELU, FUSION, TUPLE, OUTPUT, LOOKUP, ARGMAX, SOFTMAX, COMPOSITE, CODEGEN, WEIGHTED_LOOKUP, ROUTE` (ROUTE — v2.3.5: branch-подграфы, OR-overlap-метки, per-sample taken внутри ноды; clone/merge на pointer-identity maps, DCE держит ветви живыми, CSE не сливает ROUTE-ноды).

Оптимизации: CSE, DCE, fusion поэлементных цепочек, broadcast-shape, COMPOSITE-expansion. Кодоген (`src/codegen/hlo_codegen.cpp` → `hlo_stencil.cpp`): **единый стенсил-эмиттер** `compile_hlo_stencil` (v2.7.0) — один проход по графу выписывает байтовые AVX2-стенсилы с «дырками» (imm-формы, rel32, movabs-адреса хелперов), патчит их и копирует в W^X-страницу процесса. Кернел — C-ABI-функция `void fn(float** params, float* out)`; узел `DOT` → `call cblas_sgemm` (адрес прошивается movabs-патчем). Подмножество (v2.7.1, срезы H4a/H4b): PARAMETER, скалярный CONSTANT, DOT, ADD/MUL/RELU, CODEGEN (`tanh`/`fabs`/`fneg`/`clamp01`/`copy`), SOFTMAX, ARGMAX, LOOKUP, WEIGHTED_LOOKUP; фоллбэк — для ROUTE, COMPOSITE/TUPLE. Промежуточного IR нет — LLVM-путь (IR → объектник → lldELF → .so → dlopen) удалён в v2.7.0.

Кэш — in-process по fingerprint (версия эмиттера + shape-ключ; с v2.3.6 включает branch-подграфы и метки). Дискового кэша больше нет. Fallback: граф вне компилируемого подмножества → интерпретация клонированного графа (`HloGraph::clone()`); `HLO-RUN` с v2.3.7 защищён arg-count guard (ошибка вместо SIGSEGV).

С v2.3.7 граф — данные: `GRAPH-DATA`/`GRAPH-FROM-DATA` (S-выражение с ROUTE-ветвями и константами, точный круговой рейс print→read→print) и `GRAPH-RUN-PASSES` (CSE/DCE/Fusion из Lisp); COMPOSITE-ноды ездят через данные и материализуются fallback-исполнителем. Range-кернел `fn_range(params, out, start, end)` строится в той же странице (v2.6.2) — см. §10.6.

### Copy-and-Patch JIT (v2.5.0+, `src/jit/`)

Горячие замыкания компилируются тем же методом — без IR и без LLVM; детали — гл. 20 и `src/jit/CODE.md`. Ключевые факты для экскурсии:

- **ABI страницы**: `SExpr *entry(SExpr **argv, size_t argc, Interpreter *interp)`; локали — слоты `[rbp - 8*(slot+1)]` по 8 байт (boxed `SExpr*` или raw int64/double).
- **Один проход по S-дереву**: инлайнятся `if`/`let`/`while`/`setq`/`progn`, арифметика и сравнения с двумя числовыми аргументами; всё остальное — `lookup_var` + `call_value`, поэтому рекурсия jit-to-jit работает.
- **Bail-блок**: несовпадение типа или неизвестная ситуация → возврат bail-токена → интерпретатор повторяет тело по master-копии (transaction-retry). Хук — `Interpreter::apply` (case `JITPROG`).
- **Эпоха и heal**: любое `defmacro`/`define-compiler-macro` → `jit::invalidate_all()` (все страницы сбрасываются); stale-обёртка пересобирается после 64 интерпретируемых вызовов, засевая решётку типов наблюдаемой сигнатурой (`JIT-SIG`).
- **Только SysV x86-64** — стенсилы выписаны руками под эту ABI (Win64/ARM64 — в плане).

## 19.10 Neuro-Symbolic Router

`src/ns/route.{hpp,cpp}`:

- `RouteHop` — одна символьная развилка: `decision_logits`, `batch`, `taken[]`, `correct[]`, `branch_covers[]`, диапазон tape-узлов.
- `RouteHop::resolve_correct(label, taken)` — case-insensitive; OR-aware: taken-ветвь первая; если она покрывает label → маршрут верный (виновен лист).
- `Router` — упорядоченные хопы от корня к листу с depth-трекингом.
- `GuiltDetector::detect(router, labels)` → `GuiltVerdict { kind, guilty_hop, per_sample_hop, per_sample_kind }`.
- `NS-IF` (`do_ns_if`) выбирает ветвь per-sample по argmax, записывает хоп, исполняет ветвь первого образца (для чистого батча; для смешанного — маршрут/guilt per-sample, но исполнение одно).
- `NS-GRAD!` (`do_ns_grad`) блокирует tape-узлы виновных ветвей, добавляет softmax-weighted STE в `decision_logits->grad`.
- С v2.3.4 роутер работает и поверх макро-слотов (`ns-route`/`ns-route-sample`/`ns-route-index`): NS-RL-движок blackit (T1–T4) живёт на уровне stdlib + примитивов `SAMPLE`/`EVAL-SANDBOXED` — гл. 14.10.

## 19.11 LSP

`qualisp-lsp` (`src/lsp/server.cpp`, отдельная цель CMake без линковки движка) — с v2.3.8 полноценный Language Server, автономный (JSON-RPC over stdio, встроенный JSON-парсер с честным анескейпингом `\n \t \r \b \f \uXXXX` — старый `json_get` терял `\n`, склеивая документ в одну строку):

- **Ядро**: двухпроходный сканер S-форм, повторяющий конвенции ридера (строки с экранированием, `;`-комментарии, диспетч-макросы `#' #(` `#\`, quote/quasiquote/comma); инкрементальная синхронизация (`change: 2`) через офсетный сплайсинг — без перечитывания документа.
- **Диагностики**: незакрытая форма (с позицией открывающей скобки), лишняя `)`, незакрытая строка — точные диапазоны.
- **Воркспейс-индекс** на initialize: скан `*.qlsp` (глубина ≤8, кап 500 файлов/1 МБ, пропуск `.git/build*/node_modules/.cache`); didSave/didClose перечитывают файл с диска.
- **Навигация**: cross-file definition (Location | Location[], кап 20), references (кап 200), documentHighlight, rename (WorkspaceEdit.changes по индексу, кап 50 URI), workspace/symbol (кап 100), documentSymbol (14 def-видов + deftest со строковым именем); сопоставление символов без учёта регистра (ридер апикасит).
- **Комфорт**: foldingRange (пары OPEN/CLOSE на разных строках), formatting (Lisp-отступы: тело defun/let/if/… = col+2, аргументы после головы headCol+len+1; многострочные строки и пустые строки не трогаются; минимальные per-line TextEdits), ~140 автодополнений из `lsp_tables.inc` с hover-документацией.
- **Ограничения (2.3.9+)**: семантические диагностики (unbound vars, арность — нужен eval-контекст), semantic tokens, signatureHelp; колонки в байтах (не UTF-16).
- Регрессионный тест — `tests/lsp_test.sh` (JSON-RPC-драйвер over stdio, 35 проверок).

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

Отдельного AOT-компилятора (`qlispc`, до v2.4.0) и библиотек `libqlisp_runtime.a`/`libqlisp_tensor_runtime.a` больше нет — всё нативное строится на лету из S-выражений (v2.4.0).

Модули stdlib вшиваются в бинарник на этапе сборки (`cmake/stdlib_embed.cmake`, через `cmake/gen_stdlib_header.cmake` — без `xxd` для переносимости на MSYS2/MinGW) — поэтому дистрибутив Qualisp — один файл ~2 МБ.

### Windows-сборка

С v2.7.1 Windows-сборка полностью портируема (mingw-портируемость: коллизии enum, guard BLAS-линковки, native DOT) — CI зелёный на Linux и Windows.

Корректные пути для Windows-бинарника:

1. **CI** (`.github/workflows/build.yml`, job `windows`): MSYS2 MinGW64 + OpenBLAS нативно под Win64 ABI. Публикация релиза автоматическая при пуше тега (`git tag vX.Y.Z && git push origin main vX.Y.Z`); артефакты CI переиздаются в релизах документационного репозитория.
2. **MSYS2 локально**: установить `mingw-w64-x86_64-gcc`/`mingw-w64-x86_64-openblas`, затем `cmake -G "MinGW Makefiles" -S . -B build && cmake --build build -j`.
3. **Кросс-компиляция `qualisp-lsp`/`qvalent` из Linux** через toolchain-файл (см. выше).

## 19.13 Ключевые константы

| Параметр | Значение |
|---|---|
| Блок scratch-Scope | 16 МБ (`DEFAULT_BLOCK_SIZE`) |
| ConsCellPool | предвыделенный, 64k ячеек (~1.5 МБ) за блок (v2.7.4; бывшие блоки 1 М ячеек удерживали пик навсегда) |
| Guard глубины (v2.7.4) | 10 000 eval-кадров; ловимая ошибка + `(EVAL-MAX-DEPTH n)`; RAII-декремент, eval жив после catch |
| Кольцо DEBUG-TRACE (v2.8.2) | 4096 записей; категории `scope/clone/anchor/expand/macro`; выключено по умолчанию |
| rc-контракт (v2.8.3) | `refs_` в стабильной ячейке (rc=1 при создании, `nil` бессмертен); `QUALISP_RC_POISON=1` — QA-хук против use-after-release; `QLISP_AUDIT` больше не читается |
| LSan-гейт (v2.8.4) | CI-шаг `leak_watch`: ASan c `detect_leaks=1` + `ulimit -s unlimited` — корпус ловит утечки, а не только UAF; с v2.8.6 гейт собирает **все** падения за прогон и падает в конце, а не на первом файле |
| Владение (v2.8.6) | структура — RC (`stable_ref`/`stable_release`, каскад worklist, шарится); тензорные буферы — без RC: один владелец, точный размер, `TensorBufferPool`; стабильный header тензора — heap и умирает вместе с буфером |
| TensorBufferPool | exact-size, лимит 256 МБ с эвикцией крупнейшей корзины (v2.7.4) |
| SIMD (AVX2/FMA) | 8×f32 = 256 бит |
| SIMD (NEON) | 4×f32 = 128 бит |
| Кэш HLO | in-process fingerprint (версия эмиттера + shape-ключ); дисковый кэш удалён (v2.7.0) |
| Порог heal JIT | 64 интерпретируемых вызова (`k_hot_calls`, гл. 20) |
| Модули stdlib вшито | 13 (core string dl ns ml io regex audio datetime pkg errors visual tablet) |
| `dl` v2 | слои-как-данные, ~100 строк, `linear`/`net-forward`/`train-step`/`train-epochs` |
| Автодополнения LSP | 150+ |
| Стенсилы JIT | SysV x86-64 (Win64/ARM64 — в плане) |
| GCC/Clang (минимум) | GCC 13+ / Clang 16+ (C++20); LLVM не нужен |
| qvalent локальный каталог | `./qualisp-packages/<name>/` |
| qvalent кэш | `$HOME/.cache/qualisp/<deployer>/<repo>/` |
| QSRD v2 magic | `"QSRD"` (u16 ver) |
| Тесты | сюит зелёный (Release, Linux); soak-харнесс без роста RSS; production-чеки RSS v2.7.3–v2.8.1 (RSS-пороги спрашивают «под санитайзером?» и меряют только на обычной сборке); с 2.8.3 свип под ASan, с 2.8.4 — LSan-гейт `detect_leaks=1` по всему корпусу; **v2.8.6: release 92/92 и ASan 92/92 чисто впервые за серию** (плюс продакшен-сюит на закоммиченных CSV-датасетах `tests/data/*.csv`) |
| Tail-call (v2.8.2) | self-хвост пере-биндит кадр; гейт по живым tape-узлам; `let`/`let*` разворачиваются до 8 уровней |
| Контракт CI (v2.7.1) | `run-tests` считает падения в `*test-failed*`; процесс возвращает 1 при любом упавшем сюите |
| LSP | инкрементальный sync (change:2), cross-file def/refs/rename, folding, formatting (v2.3.8) |

## 19.14 Известные ограничения (см. TEMP_ISSUES.md и TODO.md)

- **CUDA** (`src/gpu/kernels.cu`) не зарегистрирован и не тестирован.
- **JIT-стенсилы** — только SysV x86-64; небольшое подмножество форм; `MACROLET`-тела не компилируются; bail = повтор тела интерпретатором (transaction-retry, гл. 20.7).
- **Диспетчеризация ROUTE стенсилами** — в плане; ROUTE исполняется fallback-программой (сама ROUTE-нода и её backward `HLO-ROUTE-GRAD!` реализованы в 2.3.5–2.3.6).
- **EnvFrame** использует C++ `std::map` — будет переписан на SExpr-alist в Phase 12.
- **`Tensor::is_stable`** и **`Tensor::size_bytes`** пока не реализованы.
- **Произвольное переписывание графа** (`graph-node`/`graph-rewrite`) — в плане; база готова: `GRAPH-DATA`/`GRAPH-FROM-DATA`/`GRAPH-RUN-PASSES` (v2.3.7).
- **HLO fallback-граф (без `HLO-COMPILE`)** — держит константы формы трассировки, кросс-форменное использование небезопасно (`TEMP_ISSUES` #23).
- **JIT bail-путь читает устаревшее master-замыкание** (`TEMP_ISSUES` #39, v2.8.6, открыто): вложенный вызов `defun` ровно в одной позиции аргумента — `(+ p (a q))` или `(+ (a p) q)` — валит тест `jit.qlsp` под ASan (в release проходит), `(+ (a p) (a q))` и `(+ p q)` — нет. `jit_closure()` возвращает nil сразу после нативного вызова: ячейка `jit_prog` переиспользована пулом, поэтому ASan молчит (pool-recycled UAF). Две попытки фикса дали SEGV и отвергнуты — ждёт TBC. Баг предсуществует и был замаскирован утечкой ref'а, которую починили в 2.8.6.
- **Полный P2 открыт** (`TEMP_ISSUES` #37): хвостовой перебиндинг TCO делит под rc только **уже стабильные** параметры; scratch-параметры (`drop`/`range`/`take`) и горячий ML-путь (`net-infer` клонирует scratch-активацию на слой, ~10% при batch 64×128) по-прежнему копируются. Нужна смена конвенции возврата — вся цепочка eval. Per-value RC для scratch отложен (`TODO_RC.md`): несовместим с оптовым `Scope::restore`.
- **Глубокая не-хвостовая рекурсия (~4000 кадров) переполняет стек интерпретатора** — давнее ограничение, не связанное с RC; нашлось попутно в раунде 2.8.6 и не чинилось.
- **Семантические диагностики LSP** (unbound vars, арность), semantic tokens, signatureHelp — roadmap.
- **CI** — GitHub Actions (`build.yml`) на Linux и Windows; тесты — `.qlsp`-скрипты в `test/` (standalone) и `tests/` (`deftest`/`run-tests`); артефакты CI публикуются как релизные бинарники.
- Исторические фейлы ранних 2.3.x (`ns-nested-breed-guilty`, `neurosym-rule-check`) устранены в 2.3.1–2.3.2 (см. гл. 7.8 и 14.4).

---

Подробности по подсистемам: гл. 10 (HLO-стенсилы), гл. 20 (JIT copy-and-patch), гл. 21 (образы), гл. 22 (Tablet). Тесты — сюиты в `tests/` (плюс фазовые в `test/`), включая `jit.qlsp`, `image.qlsp`, `hlo_stencils.qlsp`, `hlo_stencil_ops.qlsp`, `stdlib_losses.qlsp`, `tablet.qlsp` (156 проверок), `memory_setq.qlsp`, `loop_lifetime.qlsp`, `memory_pools.qlsp`, `plist_anchor.qlsp`, `production_mem.qlsp`, `expansion_cache.qlsp`, `debug_trace.qlsp`, `tail_call.qlsp`, `rc_contract.qlsp` (9 сценариев), `live_globals.qlsp` (9 проверок), `equal.qlsp` (рефлексивность, идентичность и равенство чисел через представления), `tref.qlsp` (границы), `print_arity.qlsp`, `homoiconic.qlsp`, а с v2.8.6 — продакшен-сюит `production_training.qlsp` (CSV-датасеты, классификация/регрессия, HLO-депл с проверкой против интерпретатора, плоский `(mem-report)` по эпохам, round-trip образа и весов через qserde), `rc_escape.qlsp` (характеризационные тесты побега из `while`/`setq` перед снятием loop store), `arena_stress.qlsp`, `io.qlsp` (масштаб `read-csv`). Контракты времени жизни (v2.7.5–v2.8.3), закрытый раунд памяти с LSan-гейтом (v2.8.4–v2.8.5) и единая модель владения RC/пул (v2.8.6) разобраны отдельно в гл. 5.5.