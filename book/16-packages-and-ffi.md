# Глава 16. Пакеты и FFI

В 0.19.15 «пакетная экосистема» в привычном смысле отсутствует. Внешний код попадает в программу тремя способами: быть частью встроенной stdlib (`import`), быть отдельным `.qlsp`-файлом, запущенным раннером, или быть разделяемой C-библиотекой через FFI. Ни `qvalent`, ни `pkg-install`, ни `http-get`, ни LSP в релизе нет. Зато есть собственный бинарный формат QSRD, инструменты `tools/qsrd` и `tools/qprof` и кросс-платформенный FFI-трамплин.

## 16.1 Способы подключить код

| Способ | Что это | Python-аналог |
|---|---|---|
| `(import модуль)` | один из 11 встроенных модулей, вшитых в бинарник | `import` из стандартной библиотеки |
| отдельный `script.qlsp` | файл, переданный раннеру `qualisp script.qlsp [args…]` | `python script.py` |
| `(ffi-load path)` + `(ffi-defun …)` | разделяемая C-библиотека с объявленными сигнатурами | `ctypes.CDLL` + `argtypes`/`restype` |
| `(save-image …)` / `(load-image …)` | снимок пользовательского мира в QSRD | `pickle` + восстановление сессии |

Пакетного менеджера `qvalent` **в релизе нет** — нет ни CLI, ни манифеста `qvalent.toml`, ни команды `qvalent get user/repo`. Нет и модуля `pkg` с `(pkg-install …)`: сетевых примитивов `http-get` в рантайме нет. Это не «урезанный» менеджер, а осознанный non-goal: среда целиком лежит в бинарнике, а расширение — либо FFI, либо свои файлы.

> 🐍 Вместо `pip install` + `pyproject.toml` — `import` из stdlib и `ctypes` для внешнего кода. Всё остальное — вне среза релиза.

## 16.2 Встроенные модули вместо пакетов

Раздел сохраняет номер прежнего описания `qvalent`, но описывает реальное состояние: «менеджером» служит загрузчик встроенных модулей.

```lisp
(import ml)                       ; → ML — загрузить и открыть экспорты
(import-from collections (sum find unique))  ; → nil — только три имени
```

Правила загрузки (подробнее — гл. 15.11–15.14):

- модули — Qualisp-код, вшитый через `@embedFile`; `core` — prelude, остальные ленивые;
- импорт идемпотентен, приватные имена недостижимы, экспорты — алиасы;
- конфликт публичных имён — `ImportConflict`;
- неизвестный модуль — `UnknownModule`, неизвестный экспорт — `UnknownExport`;
- `FFI` и `NS` не модули: `(import FFI)` даёт `error: UnknownModule`.

Для собственного кода нет реестра и установщика: положите файл рядом и запустите `qualisp mylib.qlsp` либо загрузите его как скрипт. Как «пакет» можно сохранить снимок мира через `save-image` и загрузить его позже — но это персистенция, а не менеджер зависимостей.

> 🐍 `import ml` ≈ `import sklearn.metrics` из уже установленного окружения; собственного `pip` нет, а `sys.path` заменяет явный путь к файлу в CLI.

## 16.3 Собственные файлы и образы

Как выглядит рабочий цикл без пакетного менеджера:

```bash
$ qualisp script.qlsp --verbose --name lena input.txt
```

Всё после имени скрипта — program args; `(command-line-args)` вернёт их список с argv[0] во главе (гл. 15.17). Флаги самого интерпретатора (`--trace`) разбираются **только до** имени скрипта.

Для сохранения состояния есть образы QSRD:

```lisp
(defvar base :fixnum 10)
(save-image "/tmp/world.qimg")     ; → nil

;; в другой сессии
(load-image "/tmp/world.qimg")     ; → nil
base                              ; → 10
```

`save-image` пишет только пользовательские формы/данные; stdlib не входит. `load-image` чистит пользовательский мир, исполняет формы (пересобирая JIT/HLO/defuse/макросы), затем накладывает снимок `defvar`; загрузка **не атомарна**. Не сериализуются `future`, `:ptr`, JIT-страницы, HLO/defuse-кернелы, FFI-указатели, stdin/fd. `ffi-load`/`ffi-defun` при загрузке образа переигрываются — библиотеки должны быть доступны.

> 🐍 Примерный аналог — `pickle.dump(session.__dict__)` и `pickle.load` в начале нового процесса; только это не объект Python, а снимок пользовательского мира языка.

## 16.4 FFI: вызов C-библиотек

FFI — примитивы рантайма, а не модуль. Реализация — `src/ffi/ffi.zig`: C-типы, `Signature`, `FfiArgs`, `FfiFunction` и генератор trampoline. Собственный x86-64-трамплин вызывается по SysV (как весь JIT), но целевая C-функция получает **нативный ABI ОС**: SysV на Linux, Win64 на Windows. libffi не используется — внешних зависимостей нет.

### 16.4.1 ffi-load: загрузка библиотеки

| Примитив | Что делает |
|---|---|
| `(ffi-load path)` | загрузить библиотеку, вернуть `:fixnum`-индекс |

```lisp
(ffi-load "libm.so.6")    ; → 0 — индекс загруженной библиотеки
```

На POSIX это `dlopen`/`dlsym` через `std.DynLib`. На Windows — `LoadLibraryA`/`GetProcAddress`/`FreeLibrary` напрямую: у `std.DynLib` в используемой версии Zig нет Windows-бэкенда. Хранилище — `Runtime.FfiLib`.

> 🐍 `(ffi-load "libm.so.6")` ≈ `ctypes.CDLL("libm.so.6")`; только символы не «угадываются» при вызове, а объявляются явно.

### 16.4.2 ffi-defun: объявленные сигнатуры

```lisp
(ffi-load "libm.so.6")
(ffi-defun "pow" c-pow f64 (x f64 y f64))
(c-pow 2.0 3.0)             ; → 8.0 — вызов как обычной функции
```

Форма `(ffi-defun "C-имя" lisp-имя ret-type (arg type …))`:

- C-имя — строка, **регистр важен**;
- `ret-type`/`type` — `f64 f32 i64 u64 i32 u32 i16 u16 u8 ptr string void`;
- `void` — только в возврате;
- символ ищется в уже загруженных библиотеках (`dlsym`); нет символа — ошибка на объявлении.

`ffi-defun` регистрирует обычный callable; checker видит `:flonum`/`:fixnum`/`:ptr`/`:string`/`:nil`. Типы проверяются строго, без сниффинга; арность проверяется в обе стороны.

```lisp
(ffi-defun "sqrtf" c-sqrtf f32 (x f32))
(ffi-defun "ldexp" c-ldexp f64 (x f64 n i32))
(ffi-defun "strlen" c-strlen i64 (s ptr))
(ffi-defun "free"   c-free   void (p ptr))

(c-sqrtf 9.0)    ; → 3.0
(c-ldexp 1.5 2)  ; → 6.0 — f64 (SSE) + i32 (INT)
(c-strlen "hello") ; → 5 — строка как char*
(c-free nil)     ; → nil — void-возврат; nil же и NULL для ptr
(c-pow 2.0)      ; → RuntimeError — мало аргументов
```

> 🐍 Как `ctypes.CDLL(...)` с назначенными `argtypes`/`restype`, только ошибки типа приходят до вызова C и являются типизированными `RuntimeError`, а объявления живут в форме-мастере.

### 16.4.3 Таблица C-типов

| Тип в сигнатуре | C-тип | Маршалирование |
|---|---|---|
| `u8` | `uint8_t` | `:fixnum`, диапазон 0..255 |
| `i16` / `u16` | `int16_t` / `uint16_t` | `:fixnum`, проверка ширины |
| `i32` / `u32` | `int32_t` / `uint32_t` | `:fixnum`, проверка ширины |
| `i64` / `u64` | `int64_t` / `uint64_t` | `:fixnum` (u64 ≤ max i64) |
| `f32` | `float` | `:flonum`, `movss` |
| `f64` | `double` | `:flonum`, `movsd` |
| `ptr` | `void*` | `:ptr`, `nil` = NULL, `:string` = временный NUL-терминированный буфер в scratch |
| `string` | `char*` | аргумент: `:string`/`nil`; возврат: копия C-строки в scratch, NULL → `""` |
| `void` | `void` | только возврат: `nil` |

Ошибки типа/диапазона — `RuntimeError`. Возврат `ptr` — `:ptr` или `nil` для NULL.

### 16.4.4 Трамплин ABI

`FfiArgs { ints[6], sses[8], target, ret_i }` — ABI-блок, общий для обеих ОС (массивы хранят значения по классу, а не по позиции).

| ОС | Распределение аргументов |
|---|---|
| Linux / SysV | int → `rdi,rsi,rdx,rcx,r8,r9`; float → `xmm0..xmm7`; ≥6 int или ≥8 SSE — отказ |
| Windows / Win64 | регистры выбираются **по позиции аргумента**: i<4 → i-й регистр своего класса (`rcx/rdx/r8/r9` либо `xmm0..xmm3`); i≥4 → стек `[rsp+32+8*(i-4)]`, 32 байта shadow space, кадр выровнен по 16 |

Ловушка Win64:
- `(f64, u64)` → `xmm0` + `rcx`;
- `(u64, f64)` → `rcx` + `xmm1` (xmm0 при этом мусор).

Генерируемый код: `push rbx; mov rbx,rdi` → аргументы → `call [target]` → возврат (`rax` для int/ptr, `xmm0` → `ret_i` для f32/f64 через `movq`/`movss`) → `pop rbx; ret`. Проверено под Wine на `kernel32` (`GetCurrentProcessId`, `lstrlenA`, `MultiByteToWideChar` с 6 аргументами), `msvcrt pow` и сгенерированных DLL со смешанными и стековыми аргументами. Тесты: 834 на Linux и 834 на Windows (3 POSIX-FFI пропускаются, 1 Windows-FFI проверяет kernel32/msvcrt и стековые аргументы).

| Ограничение | Комментарий |
|---|---|
| Нет variadic C-функций | `printf` и т.п. не вызываются |
| Нет структур по значению, union, callbacks | только number/ptr/string/void |
| Временные строки | C не должен удерживать указатель после вызова |
| Только top-level | `ffi-load`/`ffi-defun` не работают в `future`/`eval` |
| Образы | FFI-функции не сериализуются в образ |

> 🐍 Win64-трамплин — это та же «таможня», что `ctypes`, только правила рассадки регистров задаются позицией аргумента, а не его классом; в SysV наоборот.

## 16.5 QSRD v1 / Q2 — бинарная сериализация

QSRD — собственный бинарный формат Qualisp: **v1 с таблицей секций** и расширением Q2 (секция `model`).

```text
0    4   "QSRD"
4    2   major            // несовместимое изменение кодирования значений
6    2   minor            // добавленные секции; читатель обязан их пропускать
8    4   flags            // bit0: little-endian
12   4   section_count
16   …   section_count × { u32 kind, u32 offset, u32 length, u32 crc32 }
…        payload секций
```

| kind | имя | статус | содержимое |
|---|---|---|---|
| 1 | `meta` | есть | генератор, major/minor |
| 2 | `symbols` | есть | `u32 count`, затем `u32 len + байты` |
| 3 | `forms` | есть | `u32 count`, затем формы определений по порядку |
| 4 | `data` | есть | `u32 count`, затем `(u32 symbol-index, value)` |
| 5 | `model` | есть (Q2) | `u32 count`, затем `(symbol-index, tensor)` + одно значение состояния |

crc32 на секцию → `CorruptImage`; неизвестные секции пропускаются (механизм minor-совместимости); символы хранятся один раз, значения ссылаются по `u32`-индексу. Поток значений — теги `nil/t/int/float/char/symbol/string/cons/tensor/struct/result/ref/func`; `ref` даёт общие и циклические структуры.

### Образ пользовательского мира

```lisp
(defstruct pt ((x :fixnum) (y :fixnum)))
(defun inc (x :fixnum) :fixnum (+ x 1))
(defvar base :fixnum 10)
(defvar w (:tensor :f32 (2 2)) (param (tensor ((1.0 2.0) (3.0 4.0)))))
(defvar add (:func (:fixnum) :fixnum) (let ((k base)) (lambda (x :fixnum) :fixnum (+ x k))))
(defvar p (:struct pt) (pt 3 4))
(defvar items (:list :fixnum) (list 1 2 3))
(save-image "@OUT@")
```

Сериализуются: глобальные `defvar`-значения, мастер-формы `defun`/`defgraph`/`defuse`/макросов, определения структур/трейтов, тензоры (dtype+shape+байты), списки/строки/символы по именам, result, общие и циклические ссылки через memo/`ref`. Замыкания — мастер-форма лямбды + имена захватов + env, восстановление через `(let ((capture value)…) lambda)`. Никогда не сериализуются: JIT-страницы, скомпилированные HLO-графы, планы `ns-if`, `future`, сырые `:ptr` → `UnsupportedValue`.

### Веса: save-weights / load-weights

Q2 — секция `model`:

```lisp
(defvar w (:tensor :f32 (2 2)) (param (tensor ((1.0 2.0) (3.0 4.0)))))
(defvar b (:tensor :f32 (1)) (param (tensor (0.5))))
(save-weights "@OUT@" (list 7 8))
```

- `(save-weights path [state])` пишет QSRD только с `meta`, `symbols`, `model`; параметры — стабильные тензоры с `requires_grad` в порядке `SymbolId`; `state` — произвольные данные (шаг, lr, seed), схему формат не навязывает.
- `(load-weights path)` заменяет одноимённые биндинги и возвращает состояние как `:syntax`; сначала валидируются **все** параметры (неизвестное имя / несовпадение dtype-формы → ошибка без частичного применения); замыкания в состоянии → `UnsupportedValue`.

```lisp
(write-to-string (load-weights "/tmp/opencode/weights.qsrd"))  ; → "(7 8)"
```

**`qsrd-save`/`qsrd-load` в релизе нет** — это `unknown function`. Рабочие точки входа — `save-image`/`load-image` и `save-weights`/`load-weights`.

### Инструмент `tools/qsrd` (Rust)

Читатель формата на Rust: `inspect`, `verify`, `cat`, `params`, `symbols`, `extract`. Конформанс-корпус (`tools/qsrd/corpus/basic.qlsp`, `weights.qlsp`) и `check.sh` сверяют вывод. Исходник читателя в снапшоте релиза отсутствует — описание CLI берётся из SPEC; готовые бинари есть только у `qprof`.

> 🐍 `save-image` ≈ `pickle.dump`, `save-weights` ≈ `torch.save(model.state_dict())`, а `tools/qsrd verify` ≈ утилита проверки контрольных сумм архива: формат сам несёт crc32 и версию секций.

## 16.6 CLI, аргументы и трассировка

Раздел сохраняет номер описания embedding C API; в 0.19.15 вместо C-хоста — CLI и внешние инструменты.

```text
usage: qualisp [--trace <file.json>] [script.qlsp [args...]]
```

| Формы/флаги | Поведение |
|---|---|
| `--version` | `qualisp 0.19.15`, exit 0 |
| `--help`, `-h`, `-v` | usage + `error: BadArguments`, exit 1 |
| `--trace` без файла | `BadArguments` |
| произвольный аргумент до скрипта | `BadArguments` |
| нет скрипта | REPL: баннер `Qualisp`, промпт `> `, продолжение `... `; `quit` и `(exit)` — не команды |
| файловый раннер | не печатает результат последней формы — только явный `print` |

Ошибка при исполнении файла: `path:line:col: message` из diag, иначе `place: message` из `error_location`, иначе `error: …`.

Профайлер — **инструмент, не фича языка**: форм `profile-start`/`profile-stop` нет, включение только извне:

```bash
$ qualisp --trace trace.json script.qlsp
$ qprof trace.json --top 10
name                          calls       total_us        self_us   self%
FIB                           21891      1867136.0       152565.0  100.0%
```

Пишется Chrome Trace JSON: события `B`/`E`, микросекунды, `pid=1,tid=1`; трасса пишется и при ошибке. Инструментированы только невидимые из AST точки — границы вызовов функций, вызов HLO-графа, фаза `ns-if`. Компиляция JIT-страницы входит в inclusive-время первого вызова. Агрегатор `tools/qprof` на Rust: `qprof <trace.json> [--top N]`; без аргументов — `usage: qprof <trace.json> [--top N]` (exit 2), `qprof --help` трактует флаг как файл.

> 🐍 `qualisp --trace` + `qprof` ≈ `python -m cProfile script.py` и `pstats.Stats(...).sort_stats('tottime').print_stats(10)`: профиль включается снаружи, а не директивами в исходнике.
