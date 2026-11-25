# Глава 16. Пакеты и FFI

В 0.19.20 «пакетная экосистема» в смысле центрального реестра отсутствует, но пакеты как формат уже есть. Внешний код попадает в программу несколькими способами: быть частью встроенной stdlib (`import`), лежать в `.qlsl`-пакете, быть отдельным `.qlsp`-файлом, быть разделяемой C-библиотекой через FFI, быть встроенным через `libqualisp`/Python или подключить LSP-сервер из исходников для редактора. Ни `qvalent`, ни `pkg-install`, ни `http-get` нет: `.qlsl` распространяется файлом. Зато есть детерминированный упаковщик `qualisp-pack`, поиск пакетов по `--path`/`QUALISP_PATH`, lock-файл `qualisp.lock`, edge-профиль stdlib, исполняющий C ABI `libqualisp`, Python-wheel, QSRD и кросс-платформенный FFI-трамплин. LSP (`libqualisplsp` + `qualisp-lsp`) по-прежнему собирается из исходников, и его бинарник/VSIX **в релизные ассеты не входят** (§16.7).

## 16.1 Способы подключить код

| Способ | Что это | Python-аналог |
|---|---|---|
| `(import модуль)` | один из 13 модулей stdlib; в обычной сборке вшиты в бинарник | `import` из стандартной библиотеки |
| `(import модуль)` + `--package pkg.qlsl` | модуль из подключённого `.qlsl`-пакета; поиск идёт после встроенных | `pip install` + `import` |
| `--path dir` / `QUALISP_PATH` | каталог с `*.qlsl` или отдельный архив; каталог сканируется лексикографически | `PYTHONPATH` |
| `--lock qualisp.lock` / `QUALISP_LOCK` | фиксированный набор пакетов с проверкой SHA-256, имён и версий | `pip freeze` + lock-файл |
| отдельный `script.qlsp` | файл, переданный раннеру `qualisp script.qlsp [args…]` | `python script.py` |
| `(ffi-load path)` + `(ffi-defun …)` | разделяемая C-библиотека с объявленными сигнатурами | `ctypes.CDLL` + `argtypes`/`restype` |
| `libqualisp` C ABI / Python `Session` | встроить исполняющий рантайм в C-хост или Python-процесс | CPython embedding / `ctypes` |
| `(save-image …)` / `(load-image …)` | снимок пользовательского мира в QSRD | `pickle` + восстановление сессии |
| LSP из исходников | `libqualisplsp` (analysis-only C ABI) + `qualisp-lsp` (Rust) + расширение VSCodium | `pyright`/`rust-analyzer` из исходников |

Пакетного менеджера `qvalent` **нет** — нет ни CLI, ни манифеста `qvalent.toml`, ни команды `qvalent get user/repo`. Нет и модуля `pkg` с `(pkg-install …)`: сетевых примитивов `http-get` в рантайме нет. Это не «урезанный» менеджер, а осознанный non-goal: среда лежит в бинарнике или в `.qlsl`, а расширение — либо пакет, либо FFI, либо встраивание.

> 🐍 Вместо `pip install` + `pyproject.toml` — `import` из stdlib, `.qlsl`-файл или `ctypes` для внешнего кода. Центрального индекса пакетов нет.

## 16.2 Встроенные модули и пакеты `.qlsl`

```lisp
(import ml)                       ; → ML — загрузить и открыть экспорты
(import-from collections (sum find unique))  ; → nil — только три имени
```

Правила загрузки (подробнее — гл. 15.11–15.14):

- модули — Qualisp-код; в обычной сборке вшиты через `@embedFile`, `core` и `error` — prelude, остальные ленивые;
- `import` ищет модуль сначала среди встроенных, затем в подключённых `.qlsl`-пакетах;
- пакеты подключаются явно (`--package`), каталогом (`--path`/`QUALISP_PATH`) или lock-файлом; каталог сканируется лексикографически, повторный тот же архив идемпотентен, другой пакет с тем же именем — `DuplicatePackage`;
- импорт идемпотентен, приватные имена недостижимы, экспорты — алиасы;
- конфликт публичных имён — `ImportConflict`;
- неизвестный модуль — `UnknownModule`, неизвестный экспорт — `UnknownExport`;
- `FFI` и `NS` не модули: `(import FFI)` даёт `error: UnknownModule`.

### Формат `.qlsl` и `qualisp-pack`

`.qlsl` — детерминированный ZIP с хранимыми (stored) записями и манифестом `manifest.qlsp`; порядок файлов фиксирован, CRC32 проверяется при чтении. Упаковщик `qualisp-pack` (`src/qlsl.zig`, `src/pack_main.zig`) собирает пакет из каталога модулей; штатная сборка stdlib — `zig build pack`, результат `zig-out/share/qualisp/stdlib.qlsl`. Раннер принимает пакет флагом `--package` (можно повторять), каталог — `--path`, набор — `--lock`; Python — через `Session.add_package`, `add_search_path` и `load_lock`.

```bash
zig build pack
qualisp --package zig-out/share/qualisp/stdlib.qlsl script.qlsp
```

Edge-профиль `zig build -Dembed-stdlib=false` оставляет в бинарнике только bootstrap `core`+`error`, а остальные 11 модулей ищет в `stdlib.qlsl` рядом с бинарником (`bin/../share/qualisp/stdlib.qlsl`) или в `--package`. `-Dstrip=true` уменьшает артефакты. В тестовой сборке модули всегда вшиты (`builtin.is_test`), чтобы тесты не зависели от файла рядом.

### Поиск пакетов: `--path` и `QUALISP_PATH`

`Session.addSearchPath(path)` принимает каталог или один `.qlsl`-архив. Каталог сканируется **лексикографически**; все найденные архивы подключаются по порядку. Повторное подключение того же архива — no-op, а другой архив с уже занятым именем пакета даёт `DuplicatePackage`. Отсутствующий путь — `MissingPackage`.

```bash
qualisp --path ./packages app.qlsp
QUALISP_PATH=./packages:./vendor qualisp app.qlsp
```

`--path` можно повторять. Переменная `QUALISP_PATH` разделяется разделителем путей ОС (`:` на Linux, `;` на Windows). В Python то же делает `Session(search_path="packages")` или `ql.add_search_path("packages")`.

### Lock-файл `qualisp.lock`

Lock фиксирует точный набор пакетов: относительный путь, имя, версию и SHA-256 каждого архива. Формат — S-выражение:

```lisp
(lock :version 1
      :packages ((:path "packages/my-lib.qlsl"
                  :name "my-lib"
                  :version "1.2.0"
                  :sha256 "<64 lowercase hex>")))
```

`qualisp-pack lock <out.lock> <pkg.qlsl>...` создаёт детерминированный lock: записи сортируются, пути хранятся относительно lock-файла и всегда с `/`. Поля обязательны, порядок не важен; дубликаты путей, неизвестные поля и версия не `1` — `BadLock`.

```bash
qualisp-pack lock qualisp.lock packages/*.qlsl
qualisp --lock qualisp.lock app.qlsp
qualisp app.qlsp          # qualisp.lock рядом со скриптом подхватится сам
qualisp --no-lock app.qlsp
```

Перед применением проверяются **все** записи: SHA-256, имя и версия. Несовпадение — `LockMismatch`, отсутствие файла или архива — `MissingPackage`; частичного применения нет. Когда lock активен, поисковые пути и соседний edge-`stdlib.qlsl` не используются — набор пакетов фиксирован. Явные `--package` при этом всё ещё разрешены, но конфликт имён диагностируется. `QUALISP_LOCK` задаёт lock через окружение; `--no-lock` отключает автоподхват. В Python — `Session(lock="qualisp.lock")` или `ql.load_lock("qualisp.lock")`; `lock` и `search_path` взаимоисключающие.

> 🐍 `--path` ≈ `PYTHONPATH`, `qualisp.lock` ≈ `pip freeze`/lock-файл пакетного менеджера, только без сети и индекса: архив всё равно распространяется файлом.

### 16.2.1 `libqualisp`: исполняющий C ABI

`src/runtime_abi.zig` собирает `libqualisp.so`/`qualisp.dll` — отдельную библиотеку исполняющего рантайма, не путать с analysis-only `libqualisplsp` для LSP. Внутри — настоящая `Session` с JIT, собственным `std.Io.Threaded`, c-аллокатором и мьютексом вокруг вызовов.

```c
uint32_t ql_abi_version(void);
Handle  *ql_session_new(void);
void     ql_session_free(Handle *);
int      ql_session_add_package(Handle *, const char *path);
int      ql_session_add_search_path(Handle *, const char *path);
int      ql_session_load_lock(Handle *, const char *path);
char    *ql_session_eval(Handle *, const char *source, const char *file);
void     ql_string_free(char *);
const char *ql_session_last_error(Handle *);
```

`ql_session_eval` возвращает NUL-терминированную копию напечатанного значения последней формы; её освобождает `ql_string_free`. `ql_session_last_error` — заимствованная строка до следующего вызова. `add_package`, `add_search_path` и `load_lock` возвращают `0` при успехе и `-1` при ошибке, которую можно прочитать через `last_error`; ABI-версия остаётся `1`. Один handle нельзя закрывать во время вызова из другого потока; разные сессии можно вести на разных потоках. `ql_abi_version()` сейчас равен `1`; несовпадение с обёрткой — ошибка загрузки.

### 16.2.2 Python-пакет `qualisp`

Wheel `qualisp-0.19.20-py3-none-<platform>` содержит `libqualisp` и `stdlib.qlsl`; обёртка — тонкий ctypes-слой, вся работа (reader, checker, JIT, исполнение) остаётся в Zig. `ctypes` отпускает GIL на время вызова, поэтому разные `Session` могут работать параллельно; одна сессия сериализует вызовы мьютексом.

```python
from qualisp import Session

with Session() as ql:              # при наличии wheel сам найдёт stdlib.qlsl
    print(ql.eval("(+ 1 2)"))      # 3
    ql.add_package("extra.qlsl")
    print(ql.eval("(import dl) (mse-loss (tensor ((1.0))) (tensor ((0.0))))"))

with Session(search_path="packages") as ql:   # все *.qlsl из каталога
    ql.add_search_path("vendor")
    ql.eval("(import my-lib)")

with Session(lock="qualisp.lock") as ql:      # фиксированный набор + SHA-256
    ql.load_lock("other.lock")
```

`Session(*packages, library=None, search_path=None, lock=None)`: `lock` и `search_path` взаимоисключающие (`ValueError`). Если не передано ни пакетов, ни lock/search path, подключается встроенный `stdlib.qlsl`. `Session()` ищет библиотеку через `QUALISP_LIB`, затем рядом с пакетом и в `zig-out/lib`; `stdlib.qlsl` — через `QUALISP_STDLIB` или рядом с пакетом. Ошибки приходят как `QualispError`. Есть также кодогенератор `Program`/`Sym`/`Type`/`SList`/`Call`: он только рендерит S-выражения (`defun`, `defvar`, `call`, литералы), а `Session.run(program)` отправляет текст в тот же конвейер; второго AST или Python-макросов нет.

> 🐍 `import ml` ≈ `import sklearn.metrics` из уже установленного окружения; `.qlsl` ≈ локальный wheel без индекса, а `Session` ≈ встроенный CPython, только язык другой.

## 16.3 Собственные файлы и образы

Как выглядит рабочий цикл без центрального реестра (модули — встроенные, из `.qlsl` или свои файлы):

```bash
$ qualisp script.qlsp --verbose --name lena input.txt
```

Всё после имени скрипта — program args; `(command-line-args)` вернёт их список с argv[0] во главе (гл. 15.17). Флаги самого интерпретатора (`--trace`, `--package`, `--path`, `--lock`, `--no-lock`) разбираются **только до** имени скрипта.

Для сохранения состояния есть образы QSRD:

```lisp
(defvar base (10 . :fixnum))
(save-image "/tmp/world.qsrd")     ; → nil

;; в другой сессии
(load-image "/tmp/world.qsrd")     ; → nil
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

Генерируемый код: `push rbx; mov rbx,rdi` → аргументы → `call [target]` → возврат (`rax` для int/ptr, `xmm0` → `ret_i` для f32/f64 через `movq`/`movss`) → `pop rbx; ret`. Проверено под Wine на `kernel32` (`GetCurrentProcessId`, `lstrlenA`, `MultiByteToWideChar` с 6 аргументами), `msvcrt pow` и сгенерированных DLL со смешанными и стековыми аргументами. Тесты: **1095 на Linux и 1095 на Windows** (3 POSIX-FFI пропускаются, 1 Windows-FFI проверяет kernel32/msvcrt и стековые аргументы).

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
(defstruct pt ((x . :fixnum) (y . :fixnum)))
(defun inc ((x . :fixnum)) ((+ x 1) . :fixnum))
(defvar base (10 . :fixnum))
(defvar w ((param (tensor ((1.0 2.0) (3.0 4.0)))) . (:tensor :f32 (2 2))))
(defvar add ((let ((k base)) (lambda ((x . :fixnum)) (+ x k))) . (:func (:fixnum) :fixnum)))
(defvar p ((pt 3 4) . (:struct pt)))
(defvar items ((list 1 2 3) . (:list :fixnum)))
(save-image "@OUT@")
```

Сериализуются: глобальные `defvar`-значения, мастер-формы `defun`/`defgraph`/`defuse`/макросов, определения структур/трейтов, тензоры (dtype+shape+байты), списки/строки/символы по именам, result, общие и циклические ссылки через memo/`ref`. Замыкания — мастер-форма лямбды + имена захватов + env, восстановление через `(let ((capture value)…) lambda)`. Никогда не сериализуются: JIT-страницы, скомпилированные HLO-графы, планы `ns-if`, `future`, сырые `:ptr` → `UnsupportedValue`.

### Веса: save-weights / load-weights

Q2 — секция `model`:

```lisp
(defvar w ((param (tensor ((1.0 2.0) (3.0 4.0)))) . (:tensor :f32 (2 2))))
(defvar b ((param (tensor (0.5))) . (:tensor :f32 (1))))
(save-weights "@OUT@" (list 7 8))
```

- `(save-weights path [state])` пишет QSRD только с `meta`, `symbols`, `model`; параметры — стабильные тензоры с `requires_grad` в порядке `SymbolId`; `state` — произвольные данные (шаг, lr, seed), схему формат не навязывает.
- `(load-weights path)` заменяет одноимённые биндинги и возвращает состояние как `:syntax`; сначала валидируются **все** параметры (неизвестное имя / несовпадение dtype-формы → ошибка без частичного применения); замыкания в состоянии → `UnsupportedValue`.

```lisp
(write-to-string (load-weights "/tmp/opencode/weights.qsrd"))  ; → "(7 8)"
```

**`qsrd-save`/`qsrd-load` в релизе нет** — это `unknown function`. Рабочие точки входа — `save-image`/`load-image` и `save-weights`/`load-weights`.

### Инструмент `tools/qsrd` (Rust)

Читатель формата на Rust (`tools/qsrd/main.rs`): `inspect`, `verify`, `cat`, `params`, `symbols`, `extract`. Конформанс-корпус (`tools/qsrd/corpus/basic.qlsp`, `weights.qlsp`) и `check.sh` сверяют вывод. Готовые бинари есть только у `qprof`; читатель собирается из исходников.

> 🐍 `save-image` ≈ `pickle.dump`, `save-weights` ≈ `torch.save(model.state_dict())`, а `tools/qsrd verify` ≈ утилита проверки контрольных сумм архива: формат сам несёт crc32 и версию секций.

## 16.6 CLI, аргументы и трассировка

C-хост теперь существует (`libqualisp`, §16.2.1), а CLI остаётся основным способом запуска. Отдельный C ABI у LSP (`libqualisplsp`) — analysis-only и собирается из исходников (§16.7).

```text
usage: qualisp [--trace <file.json>] [--package <pkg.qlsl>] [--path <dir>]
               [--lock <file>] [--no-lock] [script.qlsp [args...]]
```

| Формы/флаги | Поведение |
|---|---|
| `--version` | `qualisp 0.19.20`, exit 0 |
| `--help`, `-h`, `-v` | usage + `error: BadArguments`, exit 1 |
| `--trace` без файла | `BadArguments` |
| `--package` без файла | `BadArguments`; флаг можно повторять, пакеты подключаются по порядку |
| `--path` без значения | `BadArguments`; флаг можно повторять, каталог сканируется лексикографически |
| `--lock` без файла | `BadArguments`; проверяет все записи до применения |
| `--no-lock` | отключает автоподхват `qualisp.lock` и `QUALISP_LOCK` |
| `qualisp.lock` рядом со скриптом | подхватывается автоматически, если нет `--lock`/`QUALISP_LOCK` и не задан `--no-lock` |
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

## 16.7 LSP и редактор: собирается из исходников

LSP — технологический инструмент, а не модуль и не примитив языка. В `(import LSP)` получается `error: UnknownModule`, а Lisp-фор `lsp-hover`, `lsp-diagnostics` и т. п. в бинаре нет. Сервер — отдельный процесс, говорящий по stdio JSON-RPC.

**Один источник истины.** Синтаксис, типы и диагностика живут в Zig-компиляторе; Rust — только протокольная оболочка: нет второго ридера, чекера или AST. Инкрементального парсера нет, на `didChange` делается полный reparse. Анализатор **никогда не исполняет формы верхнего уровня** документа: `read → macroexpand → typecheck`; единственное выполнение кода — тела макросов. `emit`/`run` для анализируемого файла не вызываются. Библиотека называется `libqualisplsp`; исполняющий `libqualisp` — отдельная библиотека (`src/runtime_abi.zig`, §16.2.1), и LSP её не использует.

| Компонент | Что это |
|---|---|
| `src/lsp_abi.zig` | Zig-библиотека с analysis-only C ABI (`addLibrary`, `link_libc`, rpath `$ORIGIN`) |
| `tools/qualisp-lsp/` | Rust-сервер, stdio JSON-RPC; зависимости только `libloading`, `serde`, `serde_json`, фрейминг написан вручную |
| `editors/vscode/` | расширение VSCodium/VS Code на чистом JS, без npm-зависимостей: язык `qualisp`, `*.qlsp`, JSON-RPC, коллекция диагностик, 4 провайдера |

`libloading` ищет библиотеку в порядке `$QUALISP_LIB` → каталог сервера → `../lib/` → системные пути; POSIX — `libqualisplsp.so`, Windows — `qualisplsp.dll`. Параллельные запросы сериализуются Rust-мьютексом: на процесс — один анализатор.

C ABI (`src/lsp_abi.zig`) экспортирует `ql_analyzer_new`/`free`, `ql_analyzer_set_source`/`check`/`diagnostics` (+ `ql_diagnostics_free`), `ql_analyzer_document_symbols` (+ free), `ql_analyzer_definition` (+ `ql_locations_free`), `ql_analyzer_hover` (+ `ql_symbol_info_free`), `ql_analyzer_completion` (+ free), `ql_analyzer_document_highlight` (+ free), `ql_analyzer_semantic_tokens` (+ free), `ql_analyzer_register_stdlib`. Возврат `int`: `0` — ok, `1` — не найдено/диагностики, `-1` — внутренняя ошибка. Все буферы освобождаются парной `ql_*_free` из той же библиотеки, строки — NUL-терминированные копии. Позиции: line 1-based, column/offset — **байты**; перевод UTF-16 ↔ bytes делает Rust.

Что реально работает:

| Запрос | Поведение |
|---|---|
| `textDocument/publishDiagnostics` | ошибки reader'а и чекерера, включая старый позиционный вид аннотаций с позицией |
| `textDocument/definition` | `defun`/`defvar`/`defstruct`/`deftrait`; поиск по `.qlsp` воркспейса и встроенным stdlib-модулям (`qualisp/registerDocuments`, `ql_analyzer_register_stdlib`) |
| `textDocument/hover` | статический тип + исходная форма определения и комментарий над ней (докстрингов в языке нет) |
| `textDocument/documentSymbol` | outline верхнего уровня |
| `textDocument/completion` | примитивы, экспорты stdlib, локальные биндинги |
| `textDocument/documentHighlight` | все вхождения символа под курсором |
| `textDocument/semanticTokens/full` | 7 типов: type/function/variable/keyword/string/number/comment; легенда и дельта-кодирование — в Rust |
| `didChange` | полная пересылка текста |

Проверка: 16 Zig ABI-тестов и 12 Rust-интеграционных тестов над stdio (initialize/capabilities, пустые диагностики, old-syntax diagnostic, didChange, UTF-16-колонки, documentSymbol, definition+hover, completion, documentHighlight, semanticTokens, hover с формой и комментарием, definition из другого файла); мок-тест расширения — `node editors/vscode/test/smoke.js`.

**Честно про поставку.** В ассетах 0.19.20 восемь файлов: два `qualisp`, три `qprof`, два Python-wheel (`qualisp-0.19.20-py3-none-linux_x86_64.whl`, `...-win_amd64.whl`) и `SHA256SUMS.txt`. `libqualisp` лежит внутри wheel, отдельного ассета нет. Готового `qualisp-lsp`, standalone `libqualisplsp.so/.dll` и `.vsix` там **нет**. LSP поставляется в исходниках: чтобы получить сервер и расширение, соберите их локально — `bash tools/lsp/build.sh` (результат — `zig-out/lsp/`), затем `node editors/vscode/test/smoke.js` или `python3 editors/vscode/package-vsix.py` (самодостаточный `zig-out/qualisp.vsix`, ставится `codium --install-extension zig-out/qualisp.vsix --force`, без `vsce` и npm). Пути переопределяются настройками `qualisp.lsp.serverPath` / `qualisp.lsp.libraryPath`. Демо — `examples/lsp-demo.qlsp` (гл. 17.8).

> 🐍 Как `rust-analyzer`: бинарник всегда можно собрать из исходников, но в готовый релиз языка он не обязательно входит. Расширение при этом тонкое, как клиент LSP на чистом JS, а не сам компилятор на TypeScript.
