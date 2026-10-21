# Глава 4. Система типов

## 4.1 Постепенная типизация

Qualisp — язык с **постепенной типизацией** (gradual typing): аннотации необязательны, но там, где они есть, компилятор их проверяет и использует для оптимизации. Реализовано через `TypeChecker` (`src/types/typecheck.cpp`) и тип `Type::T_UNKNOWN` (`src/types/type.hpp`) — если аннотации нет, тип считается `unknown`, проверки пропускаются.

```lisp
;; Без типов — полностью динамический код
(defun add (a b) (+ a b))

;; С типами — как аннотации Python
(defun add (a: Int b: Int) -> Int (+ a b))

(defun square (x: Float) -> Float (* x x))

(defun mixed (a: Int) -> Int a)
```

> 🐍 **Python-аналогия.** Прямое соответствие с type hints:
> ```python
> def add(a: int, b: int) -> int: return a + b
> def square(x: float, /) -> float: return x * x
> ```

Поддерживаемые аннотации (см. `Type::Kind`):

| Имя | Что означает |
|---|---|
| `Int` (по умолчанию) | целое — `T_INT` (int64 в C++) |
| `Float` | по умолчанию `T_FLOAT` (F64) |
| `F32` / `F64` | явный IEEE 754 |
| `I32` / `I64` | явный целый тип |
| `Bool` | булево |
| `Str` | строка |
| `Char` | символ |
| `Tensor` | `(Tensor elem [dims...])` |
| `Linear` | линейная обёртка для владения |

Аннотации парсятся из `defun`-тела: `do_defun` (`src/eval/interpreter.cpp:595`) видит первый элемент body — символ `->` — и парсит тип возврата через `Type::parse`.

## 4.2 Форма `the`

`(the T expr)` — утверждение «это выражение имеет тип T»:

```lisp
(the Float 3.14)        ; float, как 3.14
(the Int 42)            ; int
(the Float (mixed 10))  ; int-значение под float-типом
```

В реализации `sym_the` обрабатывается как special form в `Interpreter::eval_cons`. Это аналог `typing.cast(T, expr)`.

## 4.3 Что дают аннотации

1. **Ранние ошибки.** Несовпадение типов ловится на этапе компиляции, а не падением в рантайме (`type_checker_` внутри `Interpreter`).
2. **Оптимизации.** Нативный код строится без теговых проверок там, где тип доказан: JIT copy-and-patch (гл. 20) ведёт решётку типов Unknown/Fixnum/Flonum, а по наблюдаемым сигнатурам аргументов специализирует протолог — proven-параметр guard'ится и распаковывается один раз, тело течёт сырыми int64/double.
3. **Документация.** Сигнатура видна в LSP-подсказках (`qualisp-lsp`).

## 4.4 Линейные типы тензоров

Для `Tensor` действуют **линейные правила использования**: тензор — это не «ссылка на буфер», а значение с одним владельцем. Тип `T_LINEAR` (см. `Type::make_linear`) оборачивает внутренний тип и говорит: «у значения один владелец, освобождение детерминированное».

Многие операции **потребляют** вход:

| Операция | Что происходит с аргументами |
|---|---|
| `(T+ x y)` | `x`, `y` не тронуты, результат — новый тензор в TensorBufferPool |
| `(MATMUL x w)` | `x`, `w` не тронуты, результат — новый тензор в TensorBufferPool |
| `(SETQ w v)` | обновление `w` на месте |
| `(DEFVAR w v)` | `v` клонируется (`clone_to` в root_arena) в StableMemory |

Это не borrow-checker (никаких lifetime-аннотаций) и не счётчик ссылок — просто одно правило: **у значения один владелец; нужно оставить себе копию — сохраните через `defvar` (гл. 5)**.

> 🐍 **Python-аналогия.** В NumPy/PyTorch почти всё — shared views: `b = a[0]` даёт представление над тем же буфером, и `b` неожиданно меняет `a`. В Qualisp views нет: `reshape` и `transpose` делают глубокую копию (см. `Tensor::reshape`, `Tensor::transpose` в `src/tensor/tensor.cpp`). Это медленнее на копирование, но исключает целый класс багов aliasing'а и позволяет пулам (гл. 5) мгновенно переиспользовать буферы.

## 4.5 Интроспекция типов

Примитив `TYPE-OF` (`prim_type_of`) возвращает имя типа **в верхнем регистре** — по конвенции intern-символов (`T+`, `MATMUL!` и т.д.):

```lisp
(type-of 3)         ; → INT
(type-of 3.5)       ; → FLOAT
(type-of "s")       ; → STRING
(type-of '(1 2))    ; → LIST
(type-of (tensor ((1.0))))  ; → TENSOR
```

> 🐍 `type-of` ≈ `type(x).__name__`, только с uppercase-именами.

## 4.6 Проверка типов вручную

Модуль `ERRORS` (`stdlib/errors.qlsp`) предоставляет проверки:

```lisp
(import ERRORS)
(assert-type 3 'fixnum "v должен быть целым")   ; бросает ошибку, если не целое
(type-check 3 'int)                             ; → t
```

`type-check` проверяет по имени типа (`int`/`float`/`string`/`list`/`tensor`/`nil`), а точные предикаты `is-fixnum?`, `is-flonum?`, `is-number?`, `is-string?`, `is-cons?`, `is-tensor?`, `is-nil?` смотрят на теги самих значений — их результат не зависит от строкового имени типа.

## 4.7 `type-of` возвращает uppercase-имена

`type-of` нормализует имена типов в **uppercase** через `intern` (как и все остальные символы Qualisp):

| Значение | `type-of` | `type(x).__name__` (Python) |
|---|---|---|
| `42` | `INT` | `int` |
| `3.14` | `FLOAT` | `float` |
| `"hi"` | `STRING` | `str` |
| `'sym` | `SYMBOL` | (нет аналога) |
| `(1 2 3)` | `LIST` | `list` |
| `(tensor ((1.0)))` | `TENSOR` | `numpy.ndarray` / `torch.Tensor` |
| `(lambda (x) x)` / `(defun ...)` | `FUNCTION` | `function` |
| `#(1 2 3)` вектор | `VECTOR` | (нет аналога) |
| `nil` | `NIL` | `NoneType` |

> 🐍 В Python имена типов в lowercase, в Qualisp — uppercase. Это согласовано с конвенцией intern-символов (`T+`, `MATMUL!` и т.д.).

Cons-ячейки сообщаются как `LIST`. Это упрощает DSL-валидацию и проверки типов: `(if (equal (type-of x) 'LIST) ...)`.

## 4.8 Известные ограничения

- `let` под биндингом с 1- или 2-байтным dtype теряет свои биндинги — обходной путь: сохранить значение через `defvar`.
- `Tensor::is_stable` и `Tensor::size_bytes` пока **не реализованы** (см. `ARCHITECTURE.md` шапка и `TODO.md`).
- `nn.qlsp`/`optim.qlsp`/`data.qlsp` — планируемые stdlib-расширения для типизированных слоёв.
- Self-hosting перепишет `EnvFrame` с C++ `std::map` на SExpr-alist и откроет дорогу для более богатого вывода типов в `defun`-сигнатурах.
