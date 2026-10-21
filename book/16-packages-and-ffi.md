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

Qualisp умеет загружать разделяемые библиотеки и вызывать C-функции напрямую. Реализация — `src/ffi/ffi.cpp`, примитивы регистрируются через `register_ffi_primitives`.

### Примитивы

| Примитив | Что делает |
|---|---|
| `(FFI-LOAD "libm.so.6")` | `dlopen`, возвращает индекс загруженной библиотеки |
| `(FFI-CALL "fn-name" "rtype" args...)` | вызвать C-функцию по имени |
| `(FFI-IMPORT "libfoo.so" "fn-name")` | загрузить и сделать символ видимым (low-level) |

### Сигнатуры FFI-CALL

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

### Известные ограничения

- Поддерживаются только перечисленные комбинации `(rtype, atypes)`; для других — `runtime_error: "ffi-call: unsupported signature"`.
- Нет автоматического маршалинга структур, массивов фиксированного размера, указателей на функции.

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
- Возвращаемое значение — `float` или `int` или `TENSOR` (для kernel-варианта). Строки-C как возврат пока не поддерживаются.