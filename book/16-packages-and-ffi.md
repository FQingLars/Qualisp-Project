# Глава 16. Пакеты и FFI

## 16.1 Два способа ставить пакеты

| Способ | Что это | Python-аналог |
|---|---|---|
| `qvalent` (CLI) | Полноценный пакетный менеджер с манифестом `qvalent.toml` | `pip` + `pyproject.toml` |
| `(pkg-install "user/repo")` | Лёгкая установка из stdlib (через `http-get` примитив) | `curl install.sh` |

## 16.2 qvalent: pip для Qualisp

CLI `qvalent` (`src/qvalent/main.cpp`) — отдельная цель CMake. Команды (из `print_usage()`):

```bash
qvalent init                                 # создать qvalent.toml в текущей директории
qvalent get user/repo[@version]              # установить (версия опциональна)
qvalent list                                 # установленные пакеты
qvalent info <name>                          # детали пакета
qvalent update <name>                        # обновить
qvalent remove <name>                        # удалить
qvalent search <query>                       # поиск по GitHub (topic:qualisp)
qvalent help                                 # эта справка
```

Пакет — репозиторий GitHub с манифестом `qvalent.toml`:

```toml
[package]
name = "my-package"
version = "0.1.0"
description = "A Qualisp package"
deployer = "your-github-username"
repo = "repo-name"

[dependencies]
```

Реализация: парсер TOML — крошечный встроенный (`parse_qvalent_toml` в `main.cpp`). Скачивание — через `curl -L` tarball/zip с GitHub API → кэш `$HOME/.cache/qualisp/<deployer>/<repo>/` → распаковка в локальный каталог `qualisp-packages/<name>/`. Поиск (`qvalent search`) ходит в GitHub по метке `topic:qualisp`.

> 🐍 `qvalent get user/repo` ≈ `pip install git+https://github.com/user/repo`; `qvalent search` ≈ `pip search` (который в pip давно отключили).

## 16.3 pkg-install: минимум

Модуль `PKG` (`stdlib/pkg.qlsp`) — качает `pkg.qlsp` из ветки `main` репозитория и перечисленные в нём файлы:

```lisp
(import PKG)
(pkg-install "user/repo")                 ;; → /tmp/qualisp-pkg-<repo>/
(pkg-install-branch "user/repo" "dev")    ;; другая ветка
```

Внутри (`pkg-install-branch`):
1. `http-get` на `https://raw.githubusercontent.com/<repo>/<branch>/pkg.qlsp`.
2. Парсинг манифеста (список файлов).
3. `mkdir -p /tmp/qualisp-pkg-<repo>` + `spit` манифеста.
4. Для каждого файла в манифесте: `http-get` + `spit`.

Годится для простых библиотек-однофайловок; для всего остального — `qvalent`.

## 16.4 FFI: вызов C-библиотек

Qualisp умеет загружать разделяемые библиотеки и вызывать C-функции напрямую. Реализация — `src/ffi/ffi.cpp`, примитивы регистрируются через `register_ffi_primitives`. Два пути:

- **динамический** — `FFI-LOAD`/`FFI-CALL`/`FFI-IMPORT`: типы сниффаются по значениям, набор сигнатур фиксирован;
- **объявленный** — `FFI-DEFUN`/`FFI-DEFSTRUCT`: сигнатуры и раскладка структур пишутся явно, типы проверяются строго, вызов идёт через трамплин ABI.

### Динамический путь: FFI-LOAD / FFI-CALL / FFI-IMPORT

| Примитив | Что делает |
|---|---|
| `(FFI-LOAD "libm.so.6")` | `dlopen`, возвращает индекс загруженной библиотеки |
| `(FFI-CALL "fn-name" "rtype" args...)` | вызвать C-функцию по имени |
| `(FFI-IMPORT "libfoo.so" "fn-name")` | загрузить и сделать символ видимым (low-level) |

Поддерживаемые комбинации `(rtype, arg-types)` (см. `prim_ffi_call` в `ffi.cpp`):

- `FLOAT <- FLOAT`
- `FLOAT <- FLOAT FLOAT`
- `FLOAT <- INT`
- `INT <- STRING`
- `TENSOR <- TENSOR TENSOR INT` — для вызовов ядер, принимающих тензоры

Аргументы типов:

| Lisp-значение | atype | C-тип |
|---|---|---|
| `3.14` (flonum) | `FLOAT` | `float` |
| `42` (fixnum) | `INT` | `int` |
| `"hello"` (string) | `STRING` | `const char*` |
| Тензор | `TENSOR` | `float*` (или `uint8_t*` для U8) |

Примеры:

```lisp
(ffi-load "libm.so.6")              ;; → 0 (handle id)

;; синус: float sin(float)
(ffi-call "sin" "float" 0.0)        ;; → 0.0

;; atan2: float atan2(float, float)
(ffi-call "atan2" "float" 1.0 1.0)  ;; → 0.7853982...

;; strlen: int strlen(const char*)
(ffi-call "strlen" "int" "hello")   ;; → 5
```

> 🐍 **Python-аналогия.** `ctypes.CDLL("libm.so.6").sin(0.0)`. Qualisp-вариант компактнее, но поддерживает ограниченный набор сигнатур (KISS-принцип, см. `AGENTS.md`). Для сложных случаев — `FFI-IMPORT` через `define-compiler-macro`-подход или ручной C-binding.

### Кросс-платформенность

На Linux/macOS — `dlopen`/`dlsym`/`dlclose` (`src/core/platform.hpp`). На Windows — `LoadLibraryA`/`GetProcAddress`/`FreeLibrary`. На macOS библиотеки имеют расширение `.dylib` (`SHARED_LIB_EXT`), на Windows — `.dll`.

### Известные ограничения динамического пути

- Поддерживаются только перечисленные комбинации `(rtype, atypes)`; для других — `runtime_error: "ffi-call: unsupported signature"`.
- Нет автоматического маршалинга структур, массивов фиксированного размера, указателей на функции.
- Возвращаемое значение — `float` или `int` или `TENSOR` (для kernel-варианта). Строки-C как возврат не поддерживаются — для этого есть `FFI-DEFUN` с типом возврата `string`.

### FFI-DEFUN: объявленные сигнатуры

```lisp
(FFI-LOAD "libm.so.6")
(FFI-DEFUN "pow" c-pow "f64" (x: f64 y: f64))   ;; C-имя, лисп-имя, возврат, аргументы
(c-pow 2.0 3.0)                                  ;; → 8.0 — вызов как обычной функции
```

Форма `(FFI-DEFUN "C-имя" bind "ret" (a: t1 ...))`: C-имя — строка (регистр важен!), `bind` — лисп-имя, далее тип возврата и список аргументов `имя: тип`. `dlsym` выполняется при объявлении: **нет символа — ошибка на определении**, неизвестный тип — ошибка. Под капотом — примитив `FFI-INVOKE` (тело формы).

`FFI-DEFUN` создаёт **функцию-значение**: `(type-of c-pow)` → `FUNCTION`, `(functionp c-pow)` → `t`. Сигнатура живёт в форме-мастере (мастер авторитетен, образ её переносит) — переопределение, JIT и образы видят одно и то же.

Типы аргументов проверяются **строго**, без сниффинга, диапазоны ширины — как у typed-полей структур; арность проверяется в обе стороны:

```lisp
(c-pow 2 3.0)              ;; fixnum во float-месте — ошибка
(c-ldexp 1.0 2.0)          ;; flonum в int-месте — ошибка
(c-ldexp 1.0 4294967296)   ;; 2^32 в i32-месте — ошибка (диапазон ширины)
(c-pow 2.0)                ;; мало аргументов — ошибка
(c-pow 2.0 3.0 4.0)        ;; много аргументов — ошибка
(c-free 123)               ;; fixnum в ptr-месте — ошибка
(FFI-DEFUN "nope_xyz_fn" c-nope-xyz "void" ())   ;; нет символа — ошибка на определении
(FFI-DEFUN "fabs" c-bad "void" (x: nosuchtype))  ;; неизвестный тип — ошибка
```

Возвраты: числа, `ptr`, `string`, `void`. Строки в `ptr`-местах идут как `char*`, u8-тензоры — как буферы; память арендуется **на время вызова** — C не должен удерживать указатель. `nil` в `ptr`-месте — это `NULL`.

```lisp
(FFI-DEFUN "sqrtf" c-sqrtf "f32" (x: f32))
(FFI-DEFUN "ldexp"  c-ldexp  "f64" (x: f64 n: i32))
(FFI-DEFUN "strlen" c-strlen "i64" (s: ptr))
(FFI-DEFUN "strchr" c-strchr "string" (s: ptr c: i32))
(FFI-DEFUN "free"   c-free   "void" (p: ptr))

(c-sqrtf 9.0)          ;; → 3.0
(c-ldexp 1.5 2)        ;; → 6.0 — f64 (SSE) + i32 (INT): регистры по классам аргументов
(c-strlen "hello")     ;; → 5 — строка как char*
(c-strchr "hello" 101) ;; → "ello" — char* возвращается строкой
(c-free nil)           ;; → nil — void-возврат; nil же и NULL для ptr-мест
```

> 🐍 **Python-аналогия.** `ctypes.CDLL("libm.so.6")` с назначенными `argtypes`/`restype` — только проверки происходят до C, а ошибка приходит структурой от трейта `ERROR` (гл. 15.7), и объявления живут в форме-мастере, а не в питоновском классе.

### Таблица C-типов

Типы сигнатур — единая таблица (`ctype_by_name` в `src/mop`):

| Тип в сигнатуре | C-тип | Марширование |
|---|---|---|
| `byte` (он же `u8`) | `uint8_t` | raw 0..255 |
| `i16` / `u16` | `int16_t` / `uint16_t` | диапазон ширины проверяется |
| `i32` / `u32` | `int32_t` / `uint32_t` | диапазон ширины проверяется |
| `i64` / `u64` | `int64_t` / `uint64_t` | диапазон ширины проверяется |
| `f32` | `float` | конвертация `cvtsd2ss`/`cvtss2sd` |
| `f64` | `double` | |
| `ptr` | `void*` | ptr-атом, `nil` = `NULL`; строка — `char*`; u8-тензор — буфер |
| `string` | `char*` | только возврат: `char*` читается строкой |
| `void` | `void` | только возврат: даёт `nil` |

Диапазоны целых ширин — общий `ctype_check_int` с полями структур; на границе проверяются и тип, и диапазон, ошибки — структуры от трейта `ERROR`.

### Трамплин ABI: SysV и Win64

Один W^X-stub на сигнатуру (copy-and-patch, кэш живёт до конца процесса); никакого libffi — зависимостей нет, трамплин общий для всех языков проекта.

- **SysV x86-64**: регистры по классам аргументов — 6 целых + 8 SSE, переполнение уходит на стек (седьмой целый аргумент едет по стеку — проверено на `syscall(39)` = `getpid`).
- **Win64**: по позициям — первые 4 аргумента + тень 32 байта, `al` = число SSE-регистров (variadic-контракт).
- Возврат `f32`/`f64` сворачивается в `rax`.
- На прочих платформах трамплин не построен — внятная ошибка.
- Байты кодогена сверены с gcc+objdump.

### FFI-DEFSTRUCT: C-структуры

```lisp
(FFI-DEFSTRUCT fs-pad (a: byte b: f64))   ;; C-layout: выравнивание и padding как в C
(FFI-OFFSET 'fs-pad 'a)                   ;; → 0 — byte-смещение поля
(FFI-OFFSET 'fs-pad 'b)                   ;; → 8 (padding после byte)
(FFI-SIZE 'fs-pad)                        ;; → 16 — sizeof
```

Раскладка — по правилам C: поле стоит по своему выравниванию, размер блока кратен максимальному выравниванию, padding — как в C. Поля — **только C-типы** (boxed-поле — ошибка). Доступ — обычные typed-поля: конструктор-функция, `slot-value`/`set-slot`, и диапазоны, и типы действуют так же, как в обычных структурах.

| Структура | Поля | Смещения | `sizeof` |
|---|---|---|---|
| `fs-pair` | `x: f64 y: f64` | 0, 8 | 16 |
| `fs-pad` | `a: byte b: f64` | 0, 8 | 16 |
| `fs-two` | `a: f32 b: f32` | 0, 4 | 8 |
| `fs-mix` | `a: i16 b: byte` | 0, 2 | 4 (выравнивание 2) |
| `fs-wide` | `a: u64 b: u8 c: u64` | 0, 8, 16 | 24 |

`ptr`-поле пишется указателем или `nil` (это `NULL`), читается атомом `ptr`. Проверки границ живут и в C-структурах:

```lisp
(FFI-DEFSTRUCT fs-img (rows: i32 cols: i32 data: ptr))

(let ((p (fs-img 1 2 nil)))
  (set-slot p 'rows 2147483648)   ;; вне диапазона i32 — ошибка
  (set-slot p 'rows 1.5)          ;; не тот тип — ошибка
  (set-slot p 'data (FFI-ADDR p)) ;; ptr-поле принимает указатель
  (set-slot p 'data nil))         ;; и nil (= NULL)
```

> 🐍 **Python-аналогия.** `class S(ctypes.Structure): _fields_ = [("a", ctypes.c_uint8), ("b", ctypes.c_double)]`; `FFI-SIZE` ≈ `ctypes.sizeof(S)`, `FFI-OFFSET` ≈ `S.b.offset`, доступ к полям — как к атрибутам, только через `slot-value`/`set-slot`.

### FFI-ADDR / FFI-OFFSET / FFI-SIZE

| Форма | Что делает |
|---|---|
| `(FFI-ADDR st)` | `ptr`: адрес блока C-структуры |
| `(FFI-ADDR st 'field)` | `ptr`: адрес поля — для out-параметров C |
| `(FFI-OFFSET 'st 'field)` | byte-смещение поля (как `offsetof`) |
| `(FFI-SIZE 'st)` | размер блока (как `sizeof`) |

`(type-of (FFI-ADDR p))` → `PTR`; указатели сравниваются по адресу. Out-параметры — u64-поле C-структуры плюс его адрес через `FFI-ADDR`.

### Приёмка: zlib round-trip

Настоящая C-библиотека через объявленные сигнатуры — compress/uncompress round-trip на байтах: u8-тензоры как буферы, u64-поле как out-параметр:

```lisp
(FFI-LOAD "libz.so.1")
(FFI-DEFUN "compressBound" c-compress-bound "u64" (n: u64))
(FFI-DEFUN "compress" c-compress "i32"
           (dst: ptr dst-len: ptr src: ptr src-len: u64))
(FFI-DEFUN "uncompress" c-uncompress "i32"
           (dst: ptr dst-len: ptr src: ptr src-len: u64))

(FFI-DEFSTRUCT zl-len (n: u64))    ;; uLongf* — out-параметр

(defvar src (TENSOR-U8 (tensor ((104 101 108 108 111 32 119 111 114 108 100)))))
(defvar dst (TENSOR-U8 (ZEROS (list 64))))
(defvar ln (zl-len 64))            ;; до вызова — доступный размер, после — фактический

(c-compress dst (FFI-ADDR ln 'n) src 11)   ;; → 0 (Z_OK)
(slot-value ln 'n)                          ;; → размер сжатого
```

Полный round-trip (`c-uncompress` обратно в буфер 11 байт) восстанавливает байты бинарно; маленький буфер даёт ненулевой код ошибки, а не краш. Всё то же самое работает из JIT-кода.

### Вызовы FFI из JIT-кода

Вызовы FFI-DEFUN компилируются: паритет с интерпретатором, цепочки FFI-вызовов, вызовы в циклах:

```lisp
(let ((f (lambda (x) (c-pow (c-fabs x) 2.0))))
  ((JIT f) -3.0))   ;; → 9.0 — цепочка FFI-вызовов в jit
```

### Известные ограничения FFI

- Колбэков C→Lisp нет.
- `FFI-CALL` — динамический легаси-путь: цепочка известных сигнатур, новые сигнатуры туда не добавляются (для этого — `FFI-DEFUN`).
- Кэши dlsym/сигнатур/трамплинов процессные: после `LOAD-IMAGE` `FFI-DEFUN` нужно переобъявить (как `FFI-LOAD`).
- Трамплин — только SysV x86-64 и Win64.

## 16.5 QSRD v2 — бинарная сериализация

QSRD — собственный бинарный формат Qualisp для персистенции данных. Версия 2 добавила:

- Блоб типа `2` — S-выражение в бинарной сериализации (cons-ячейки, символы, числа, строки).
- Блоб типа `3` — типизированный тензор (dtype сохраняется вместе с шейпом и данными).

Магический заголовок: `"QSRD"` + u16 версия.

```lisp
;; Сохранить данные
(defvar model (list (linear 1 4) 'relu (linear 4 1)))
;; ... обучить ...
(qsrd-save "model.qsrd" model)

;; Загрузить
(defvar loaded (qsrd-load "model.qsrd"))

;; Получить поле из структуры (если QSRD — структура)
(qsrd-get loaded 'weights)
```

Примитивы (`prim_qsrd_save`/`prim_qsrd_load`/`prim_qsrd_get` в `src/eval/interpreter.cpp:3942-3944`).

> 🐍 **Python-аналогия.** `pickle.dump/load` в Python. Главное отличие: QSRD v2 хранит **dtype тензора** нативно (F16/F32/F64/I32/I64/U8) — `pickle` теряет эту информацию (приходится отдельно вызывать `torch.save` + `np.save` или передавать через `protocol=2`/`5`). DL-модели (список-структура) сериализуются QSRD как обычные cons-ячейки — без отдельного протокола.

### Тензоры на любой глубине

Тензор — данные **где угодно в структуре**: тег `6` кодируется в любой позиции cons-дерева (dtype + ndim + dims + raw-элементы), круговой рейс сохраняет dtype (включая F16), shape и значения; на любой глубине и рядом со скалярами/символами/строками. Модель (alist слоёв с весами) сохраняется целиком, хотя `stdlib/dl.qlsp` обещает «слои — данные, сериализуются напрямую».

```lisp
(defvar model (list "layer-1" (ones (list 2 2)) "layer-2" (ones (list 2 1))))
(qsrd-save "model.qsrd" model)
(defvar back (qsrd-load "model.qsrd"))
(tensor-shape (qsrd-get back "layer-1"))   ;; → (2 2) — веса слоя по имени
```

### Что хранится в QSRD v2

| Тип | Как кодируется |
|---|---|
| S-выражение (cons) | rec-обход cons-дерева с длинами |
| Число fixnum | int64 little-endian |
| Число flonum | IEEE 754 double |
| Строка | u32 length + bytes |
| Символ | intern-id (lookup в symbol-table) |
| Тензор (верхний уровень и любая вложенность) | тег `6`: dtype (u8) + ndim (u32) + dims (u32) + data (raw bytes) |
| NIL | один байт 0x00 |
| T | один байт 0x01 |

### Use cases

- Сохранение/загрузка обученных DL-моделей.
- Передача данных между процессами (с `--binary-mode` qvalent).
- Снапшоты rule engine (факты, правила).
- Кэш промежуточных вычислений в long-running пайплайнах.

### Известные ограничения

- Нет версионирования схемы внутри блоба — добавление нового типа меняет формат.
- Нет сжатия (только raw bytes) — для больших моделей используйте внешний `.tar.gz`.
- Не читается из других языков напрямую — нужен отдельный reader (формат документирован, но reference-парсера на Python нет).