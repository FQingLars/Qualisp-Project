# Глава 15. Стандартная библиотека

Стандартная библиотека Qualisp 0.19.20 — это **13 модулей**, написанных на самом Qualisp. В обычной сборке они вшиты в бинарник через `@embedFile` (`src/stdlib.zig`); edge-профиль `-Dembed-stdlib=false` оставляет в бинаре только bootstrap `core`+`error`, а остальные 11 модулей читает из `stdlib.qlsl`. Бинарные ассеты 0.19.20 собраны именно в edge-профиле, поэтому для них нужен пакет из wheel (гл. 2.1). Никакой внешней среды с `pip`-пакетами: всё, что нужно программе, лежит в исполняемом файле или рядом в `.qlsl`; каталоги поиска задаются `--path`/`QUALISP_PATH`, а `qualisp.lock` фиксирует SHA-256 и версии пакетов. `core` и `error` грузятся при старте как **prelude**, остальные модули подключаются **лениво** при первом `import`; повторный импорт идемпотентен. Порядок `base_modules` фиксирован: `core`, `error`, `string`, `collections`, `test`, `ml`, `dl`, `symbolic`, `argparse`, `tablet`, `io`, `datetime`, `neuro` (`src/stdlib.zig`). `Session.loadPrelude` грузит и импортирует ровно `core` и `error` (`src/session.zig`).

| Модуль | Файл | Экспорты | Содержание | Python-аналог |
|---|---|---|---|---|
| `core` | `src/stdlib/core.qlsp` | 2 | prelude: `caar`, `identity` — базовые утилиты над `:syntax` и generic-тождество | `builtins` |
| `error` | `src/stdlib/error.qlsp` | 17 | prelude: `error` / `call-frame`, категории, конструкторы и аксессоры | `Exception` + `traceback` |
| `string` | `src/stdlib/string.qlsp` | 11 | строки: поиск, паддинг, разбиение на строки/слова, приведение к fixnum | `str` |
| `collections` | `src/stdlib/collections.qlsp` | 14 | свёртки, поиск, срезы по предикату, `unique`, `zip-with` | `functools` + `itertools` |
| `test` | `src/stdlib/test.qlsp` | 2 | `assert-fixnum`, `assert-string` — аварийные утверждения | `assert` / `pytest` |
| `io` | `src/stdlib/io.qlsp` | 4 | файлы, построчное чтение, потоковые хэндлы, макрос `with-file` | `open`, `pathlib` |
| `ml` | `src/stdlib/ml.qlsp` | 13 | метрики, разбиение выборки, масштабирование, `t-mse` | `sklearn.metrics` + `sklearn.preprocessing` |
| `dl` | `src/stdlib/dl.qlsp` | 36 | слои, потери, метрики, init, dropout, conv/pool, attention, embedding, SGD/Adam/momentum | `torch.nn` + `torch.optim` |
| `symbolic` | `src/stdlib/symbolic.qlsp` | 17 | сопоставление, переписывание, упрощение, `diff`, `grad`, `unify`, полиномы | `sympy` |
| `argparse` | `src/stdlib/argparse.qlsp` | 4 | сканер аргументов: `flag?`, `option-value`, `require-option`, `positionals` | `argparse` (сканирование) |
| `tablet` | `src/stdlib/tablet.qlsp` | 2 | числовая таблица: `defstruct` + `deftrait`/`impl`, методы на `:f32` | `pandas.DataFrame` |
| `datetime` | `src/stdlib/datetime.qlsp` | 14 | UTC-календарь, epoch, ISO-8601, таймеры | `datetime` (UTC-часть) |
| `neuro` | `src/stdlib/neuro.qlsp` | 3 | мост символьный терм → тензорный код: `term`, `term->code`, `term-problem` | (нет прямого аналога) |

Есть ещё **`src/stdlib/probe.qlsp`** и **`src/stdlib/typepairs_probe.qlsp`** — test-only фикстуры: они вшиваются только в test-сборки (`builtin.is_test`) и **не являются модулями для пользователя**. Модулей ровно тринадцать.

> 🐍 «13 модулей в одном бинарнике» — это монолитный Docker-образ с предустановленными `numpy/torch/sklearn/pandas/sympy`, только без GIL, без 4 ГБ зависимостей и без `pip install`. Edge-профиль — тот же образ, но stdlib вынесена в отдельный `.qlsl`-архив.

Модули `audio`, `async`, `ns`, `pkg` и менеджер `qvalent` в исходниках отсутствуют. Пакетного реестра тоже нет, но есть формат `.qlsl` и упаковщик `qualisp-pack` (`zig build pack`): `stdlib.qlsl` собирается из `src/stdlib/manifest.qlsp`; `import` ищет модуль сначала среди встроенных, затем в подключённых пакетах (`--package`, `--path`, `QUALISP_PATH`). Для воспроизводимого окружения `qualisp-pack lock` пишет `qualisp.lock`, а `--lock`/`QUALISP_LOCK` проверяют хэши, имена и версии до применения. LSP (`libqualisplsp`, Rust-сервер `qualisp-lsp`, расширение для VSCodium) есть **в исходниках**, но его бинарник и VSIX в релизные ассеты не входят (гл. 16.7). Вместо тестового фреймворка `deftest`/`run-tests` есть только `assert-*`, падающие через `throw`; прогон — внешний (`zig build test`).

## 15.1 core и error — всегда загружены

`core` и `error` — два модуля, которые не нужно импортировать: `Session.loadPrelude` грузит их при старте сессии. У `core` два экспорта:

```lisp
(defun caar ((x . :syntax)) ((car (car x)) . :syntax))
(defun identity ((x . :a)) (x . :a))
```

```lisp
(caar '((1 2) 3))   ; → 1
(identity 5)        ; → 5
```

`identity` — generic-функция с переменной типа `:a`: это демонстрация параметрического полиморфизма. `caar` работает над `:syntax`, а не над `(:list T)`: если нужен `car` от обычного списка, он доступен как примитив. Второй prelude-модуль, `error`, нужен также рантайму: `throw` и рантайм-ошибки создают структуры по его тегам. Его API разобран в §15.13.

> 🐍 `identity` ≈ тождественная функция `lambda x: x`; `caar` ≈ `lambda x: x[0][0]`, только аргумент — S-выражение как данные (`:syntax`), а не питоновский список.

## 15.2 string — строковые операции

Модуль даёт 11 функций поверх **глобальных строковых примитивов** (`string-length`, `string-split`, `substr`, `strcat`, `string-upcase`, `string-downcase`, `string-trim`, `string-index`, `string-to-number`, `number-to-string`, `string=`, `to-string` и др.). Индексы — **кодовые точки**, не байты.

```lisp
(import string)

(string-contains "abc" "b")          ; → t
(string-starts-with "hello" "he")    ; → t  — сначала Compare длин
(string-ends-with "hello" "lo")      ; → t  — пустой суффикс даёт t
(string-repeat "ab" 3)               ; → "ababab" — удвоение блоков
(string-blank? "   ")                ; → t
(string-pad-left "7" 3 "0")          ; → "007"
(string-pad-right "7" 3 "0")         ; → "700"
(string-lines "a\nb\n")              ; → ("a" "b") — хвостовая пустая убрана
(string-words "a b  c")              ; → ("a" "b" "c")
(string-to-fixnum "42")              ; → 42; нецелое/мусор → nil
(string-capitalize "hELLO")          ; → "Hello"
```

| Экспорт | Сигнатура | Что делает | Python |
|---|---|---|---|
| `string-contains` | `:string :string → :bool` | вхождение подстроки | `sub in s` |
| `string-starts-with` | `:string :string → :bool` | префикс | `s.startswith(p)` |
| `string-ends-with` | `:string :string → :bool` | суффикс | `s.endswith(p)` |
| `string-repeat` | `:string :fixnum → :string` | `n` копий | `s * n` |
| `string-blank?` | `:string → :bool` | пустая после trim | `not s.strip()` |
| `string-pad-left` | `:string :fixnum :string → :string` | паддинг слева | `s.rjust(w, pad)` |
| `string-pad-right` | `:string :fixnum :string → :string` | паддинг справа | `s.ljust(w, pad)` |
| `string-lines` | `:string → (:option (:list :string))` | строки без хвостовой пустой | `s.splitlines()` |
| `string-words` | `:string → (:option (:list :string))` | слова, пустые куски отброшены | `s.split()` |
| `string-to-fixnum` | `:string → (:option :fixnum)` | сужение `string-to-number` до int | `int(s)` с проверкой |
| `string-capitalize` | `:string → :string` | первая вверх, остальные вниз | `s.capitalize()` |

Два порядка аргументов часто путают — запомните их раз и навсегда:

| Форма | Порядок | Пример |
|---|---|---|
| `string-split` | **(строка разделитель)** | `(string-split "a,b,c" ",")` → `("a" "b" "c")` |
| `string-join` | **(список разделитель)** | `(string-join (list "a" "b") ",")` → `"a,b"` |
| `substr` | `(строка начало длина)` | `(substr "abcdef" 1 3)` → `"bcd"` |

> 🐍 `string-split` ≈ `s.split(delim)`, а **не** `delim.split(s)`; `substr` ≈ `s[from:from+len]`, третий аргумент — длина, а не конец.

Реализация `string-repeat` — удвоение блоков: каждая `strcat` копирует вдвое больший кусок, поэтому суммарная работа пропорциональна длине результата, а не `n × len`. Результат `string-lines`/`string-words` — option-список, потому что в языке нет пустого списка: пустой результат честно выражается `nil`.

## 15.3 io — файлы и потоки

Уровень примитивов (работают без импорта):

| Примитив | Результат | Python |
|---|---|---|
| `(read-file path)` | `:string` | `open(p).read()` |
| `(write-file path text)` | `:nil` | `open(p,'w').write(s)` |
| `(append-file path text)` | `:nil` | `open(p,'a').write(s)` |
| `(file-exists? path)` | `:bool` | `os.path.exists(p)` |
| `(open-file path mode)` | `:fixnum`-хэндл (`#\r`, `#\w`, `#\a`) | `os.open` |
| `(read-line h)` | `(:option :string)` | `f.readline()` |
| `(read-all h)`, `(write-string h s)`, `(write-line h s)`, `(close-file h)` | потоковый ввод-вывод | `read`/`write`/`close` |

Хэндлы — токены поверх `std.Io.File`; чтение буферизовано, закрытие идемпотентно, незакрытые хэндлы закрываются при teardown. Ошибки файловой системы — runtime-ошибки и ловятся обычным `catch-error`.

Модуль `io` добавляет построчные helper'ы и макрос:

```lisp
(import io)

(write-file "/tmp/t.txt" "a\nb\n")       ; → nil
(read-file "/tmp/t.txt")                 ; → "a\nb\n"
(read-lines "/tmp/t.txt")                ; → ("a" "b") — хвостовая пустая убрана
(write-lines "/tmp/out.txt" (list "a" "b"))

;; потоковый хэндл с гарантированным закрытием
(print (with-file (f "/tmp/t.txt" #\r) (read-line f)))  ; → "a"
(each-line "/tmp/t.txt" (lambda ((s . :string)) (print s) t))  ; → nil
```

`(with-file (name path mode) body...)` — **макрос**: тело остаётся обычным кодом, файл закрывается и на успехе, и на `throw`. Внутри используется `catch-error` + `result-ok?`/`result-value`/`result-message`; `throw` внутри тела перевыбрасывается после закрытия. `each-line` явно повторяет тот же протокол open/catch/close. `or-else` распаковывает option после `null`-проверки:

```lisp
(let ((line (read-line h)))
  (if (null line) nil (funcall f (or-else line ""))))
```

CSV, regex и аудио в stdlib 0.19.20 **не входят**: для разбора строк используйте `string-split`/`substr`, для данных — `tablet`. Дата и время теперь есть — это модуль `datetime` (§15.18). Вызов `with-file` не менялся: `defmacro` — единственное место, где параметры по-прежнему записываются позиционно (гл. 12).

> 🐍 `with-file` ≈ `with open(path, mode) as f:`, только это макрос, раскрывающийся в `let` + `catch-error`, а не контекстный менеджер рантайма.

## 15.4 Глобальные примитивы: строки, приведения, типы

Часть API — не Lisp-код, а встроенные примитивы рантайма. К ним относятся `map`/`filter`/`reduce`, `substr`, все `string-*`, математика (`sqrt`, `pow`, `exp`, `log`, `sin`, `cos`, `tanh`, `atan`), `command-line-args`, файловые примитивы, `save-image`/`load-image`, `save-weights`/`load-weights`, `future`/`await`, `eval`, `macroexpand*`, `throw`/`catch-error`.

Приведения всегда возвращают option — это честная замена «тихому» приведению Python:

```lisp
(to-fixnum "42")     ; → 42
(to-fixnum "4.5")    ; → nil
(to-flonum "2.5")    ; → 2.5
(to-string 42)       ; → "42"
(or-else (to-fixnum "x") 0)   ; → 0
```

`type-of` печатает **uppercase-имена**:

| Значение | `type-of` | Python `type(x).__name__` |
|---|---|---|
| `42` | `INT` | `int` |
| `3.14` | `FLOAT` | `float` |
| `"hi"` | `STRING` | `str` |
| `'sym` | `SYMBOL` | (нет аналога) |
| `#\a` | `CHAR` | (нет аналога) |
| `t` / `nil` | `BOOL` / `NIL` | `bool` / `NoneType` |
| `(1 2 3)` | `LIST` | `list` |
| `(tensor ((1.0)))` | `TENSOR` | `numpy.ndarray` |
| `(lambda …)` / `(defun …)` | `FUNCTION` | `function` |
| `(pt 1 2)` | `STRUCT` | `class` instance |
| `(read-from-string "(1 2)")` | `LIST` | `ast.parse(...)` (данные-форма) |

> 🐍 В Python имена типов lowercase; здесь они интернируются в верхний регистр. `type-of` — диагностика, а не замена статической типизации: типы известны на этапе проверки.

## 15.5 test — утверждения

Модуль `test` — это не фреймворк с реестром тестов, а два аварийных утверждения:

```lisp
(import test)

(assert-fixnum (+ 2 2) 4)        ; → t
(assert-string (string-upcase "a") "A")  ; → t
(assert-fixnum 1 2)               ; → throw "assert-fixnum: expected 2, got 1"
```

Упавший `assert` **прерывает прогон** через `throw` — возврат `nil` ничего не защитил бы. Сообщение содержит и ожидаемое, и фактическое значение, потому что бэктрейс знает только место вызова.

> 🐍 `assert-fixnum` ≈ `assert a == b, f"expected {b}, got {a}"`; `run-tests`/`pytest` в релизе нет, прогон тестов — внешний (`zig build test`).

В 0.19.20 тесты есть **на каждую экспортируемую функцию** `tablet`, `ml`, `dl`, `symbolic`, `neuro`, `argparse`, `io`, `string`, `collections`. Всего `zig build test` прогоняет **1095 тестов** (1094 именных `test "…"` плюс один анонимный блок в `src/tests.zig`) на Linux и Windows.

## 15.6 ml — метрики и подготовка данных

13 экспортов. Метрики определены на непустом `(:list :flonum)`; fixnum-входы расширяются до flonum при унификации, поэтому `(mean (list 1 2 3))` типизируется без второго имени.

```lisp
(import ml)

(mean (list 1.0 2.0 3.0))                 ; → 2.0
(mse (list 1.0 2.0) (list 1.0 4.0))       ; → 2.0
(mae (list 1.0 2.0) (list 1.0 4.0))       ; → 1.0
(rmse (list 1.0 2.0) (list 1.0 4.0))      ; → 1.414…
(accuracy (list 1 0 1) (list 1 1 1))      ; → 0.666…
(precision (list 1 0 1) (list 1 1 1))     ; → 1.0
(recall (list 1 0 1) (list 1 1 1))        ; → 0.666…
(f1 (list 1 0 1) (list 1 1 1))            ; → 0.8
(minmax-scale (list 1.0 2.0 3.0))         ; → (0.0 0.5 1.0)
(standardize (list 1.0 2.0 3.0))          ; → (-1.224… 0.0 1.224…)
(train-part (list 1 2 3 4) 0.5)           ; → (1 2)
(test-part (list 1 2 3 4) 0.5)            ; → (3 4)
```

| Экспорт | Контракт | Python |
|---|---|---|
| `mean`, `mse`, `mae`, `rmse`, `accuracy` | непустой список, результат `:flonum` | `statistics.mean`, `sklearn.metrics.*` |
| `train-part`, `test-part` | `(:option (:list :a))`, пустая часть — `nil`, ratio вне `0..1` — `throw` | `train_test_split` |
| `minmax-scale`, `standardize` | вырожденный вход → нули, не NaN | `MinMaxScaler`, `StandardScaler` |
| `t-mse` | MSE для 1-D `(:tensor :f32 (_))`, через `tmean` | `F.mse_loss` |
| `precision`, `recall`, `f1` | бинарные метки 1/0; нулевой знаменатель → `0.0` | `sklearn.metrics.precision_score` |

`standardize` делит на **n** (генеральное СКО), а не на `n-1`. `ratio-count` округляет `ratio*n` с допуском `1e-9`, чтобы `0.3*10 = 2.9999999999999996` не терял элемент.

> 🐍 Это «sklearn без объектов”: никаких классов-estimator, только чистые функции над списками и тензорами, типы которых видны до запуска.

## 15.7 dl — слои и оптимизаторы

36 экспортов. Почти все функции чистые: переданные тензоры не мутируются, результат — новый тензор или структура. Исключения — `sgd-update!`/`adam-update!`: они меняют stable-тензор и состояние на месте. Работает только `:f32`; ранг частично полиморфен — потери и состояние оптимизатора принимают `(:tensor :f32 :any)`, слои с фиксированной формой результата требуют конкретный ранг. Bias — строка `(1 m)`, а не вектор `(m)`.

| Группа | Экспорты | Что делает |
|---|---|---|
| Прямой ход | `linear-forward`, `mlp-forward` | логиты, активация — выбор вызывающего |
| Нормализация и внимание | `layernorm`, `scaled-dot-attention`, `embedding-lookup` | LayerNorm, attention `[B,T,d]`, выборка строк `:i64`-индексами |
| Свёртки и pooling | `conv2d-forward` | `[N,C,H,W]` × `[O,C,KH,KW]`, bias `[O,1,1]`, stride/pad; pooling — примитивы `t-max-pool2d`/`t-avg-pool2d` |
| Dropout | `dropout`, `apply-mask` | eager inverted dropout; traceable-вариант с готовой маской |
| Потери и метрики | `mse-loss`, `squared-error`, `mae-loss`, `cross-entropy`, `cross-entropy-labels`, `accuracy-labels` | MSE/MAE, CE по логитам/меткам, accuracy |
| Инициализация | `init-weights`, `init-bias`, `glorot-scale`, `he-scale` | runtime-формы `(list n m)`, множитель к `randn` |
| Оптимизаторы | `sgd-step`, `sgd-update!`, `adam-init`, `adam-step`, `adam-param`, `adam-steps`, `adam-update!`, `momentum-init`, `momentum-step`, `momentum-param` | SGD, Adam (β₁=0.9, β₂=0.999, eps=1e-7, bias correction), momentum; `*update!` — in-place |
| Расписание и клип | `clip-by-norm`, `lr-decay` | clip по норме, шаг learning rate |
| Без графа | `linear-grads`, `linear-regression-step` | замкнутый градиент линейной регрессии через `matmul`/`transpose`/`tsum` |
| Структуры | `adam-state`, `linear-gradients`, `linear-model`, `momentum-state` | состояние в `defstruct` |

```lisp
(import dl)

(defvar x ((tensor ((1.0 2.0))) . (:tensor :f32 (1 2))))
(defvar w ((tensor ((1.0 0.0) (0.0 1.0))) . (:tensor :f32 (2 2))))
(defvar b ((tensor ((0.5 -0.5))) . (:tensor :f32 (1 2))))
(defvar y ((tensor ((1.5 1.5))) . (:tensor :f32 (1 2))))
(print (mse-loss (linear-forward x w b) y))  ; → 0.0 — скаляр :flonum
(print (linear-forward x w b))               ; → (_ _) — логиты
(defvar st ((adam-init w) . (:struct adam-state)))
(setq st (adam-step st (tensor ((0.5 -0.5) (0.1 -0.1))) 0.01))
(print (adam-steps st))                       ; → 1
```

Пример проверен бинарём 0.19.20. Обратите внимание на формы: `linear-forward` считает `xW + b`, поэтому веса `w` имеют форму `(2 2)` при входе `(1 2)`, а градиент для `adam-step` должен совпадать с формой весов.

В autograd-цикле можно обновлять веса **на месте** `tset!` по `(grad-of w)`, `sgd-update!` или `adam-update!`; чистый `adam-step` возвращает новое состояние с wildcard-формой `(_ _)`, поэтому его результат не присваивается переменной с конкретной формой `(2 1)` (см. гл. 17). `adam-update!` копирует моменты `m`/`v` в уже существующие stable-буферы через `t-copy!`, а не перепривязывает слоты к scratch. Имя `mse` уже занято модулем `ml`, поэтому тензорная потеря называется `mse-loss` — два экспорта с одним именем из разных модулей дали бы `ImportConflict`.

> 🐍 `adam-update!` ближе всего к `optimizer.step()` в PyTorch: состояние — обычная структура `adam-state`, а не скрытый словарь optimizer'а. `adam-step` — чистая функция, возвращающая новое состояние; `linear-regression-step` ≈ ручной `W -= lr * Xᵀerr` без autograd.

## 15.8 symbolic — символьный движок

17 экспортов. Терм — `:syntax`, обычное S-выражение; переменные образца — символы с префиксом `?`. Окружение — alist; **пустое окружение — `t`**, а не `nil` (nil уже значит «не совпало»). `subst` принимает `t`, `nil` и настоящий alist.

```lisp
(import symbolic)

(variable? '?x)                       ; → t
(free-of? 'x '(+ x 1))                ; → nil
(match '(+ ?x 1) '(+ 2 1))            ; → ((?X . 2))
(subst '((?X . 2)) '(* ?x ?x))        ; → (* 2 2)
(rewrite '((+ ?x 0) ?x) '(+ y 0))     ; → Y
(rewrite-all '(((+ ?x 0) ?x)) '(+ (+ y 0) 0) 10)  ; → Y
(simplify '(+ (* x 1) 0))             ; → X
(diff '(* x x) 'x)                    ; → (+ X X)
(diff-canonical '(* x x) 'x)          ; → (* 2 X)
(collect '(+ x (* 2 x)))              ; → (* 3 X)
(unify '(+ ?x 1) '(+ 2 1))            ; → ((?X . 2))
(poly? '(+ (* 2 (^ x 2)) (* 3 x) 1) 'x)      ; → t
(poly-coeffs '(+ (* 2 (^ x 2)) (* 3 x) 1) 'x) ; → (1 3 2)
(poly-eval '(+ (* 2 (^ x 2)) (* 3 x) 1) 'x 2.0) ; → 15.0
(derive '((p 1)) '(((p ?x) (q ?x))) 3) ; → ((Q 1) (P 1)) — прямой вывод
(matches '((p 1) (p 2)) '(p ?x))      ; → окружения совпадений
```

| Группа | Экспорты | Контракт |
|---|---|---|
| Базовые | `variable?`, `free-of?`, `match`, `subst` | `match` → `(:option :syntax)` |
| Переписывание | `rewrite`, `rewrite-all`, `simplify` | один шаг / до неподвижной точки с обязательным `limit` / константы и `x+0`, `x*1`, `x*0` |
| Дифференцирование | `diff`, `diff-canonical`, `grad` | `diff`: n-арные `+ - *`, унарный минус, `/`, `^`, унарные `sin/cos/exp/log/sqrt/tanh/atan`; `diff` **не приводит подобные**: `d(x*x)/dx = (+ x x)`, а `diff-canonical` даёт `(* 2 x)`. `grad` — reverse-mode градиент скалярного лосса, см. ниже |
| Логика | `unify`, `derive`, `matches` | унификация с occurs-check; `derive`/`matches` возвращают `:syntax` |
| Полиномы (S3) | `collect`, `poly?`, `poly-coeffs`, `poly-eval` | нормальная форма «сумма мономов»; плотные коэффициенты от степени 0; нулевой полином — `(0)` |

`(grad term var)` — **reverse-mode градиент скалярного лосса `(tsum term)`**, а не поэлементная производная, как `diff`. Формы сохраняются отдельными тензорными операциями, чтобы `simplify` не склеивал их в арифметику:

```lisp
(grad '(tsum x) 'x)             ; → (ONES-LIKE X)
(grad '(tmax x) 'x)             ; → (TMAX-MASK X)
(grad '(tsum (relu x)) 'x)      ; → (* (ONES-LIKE (RELU X)) (RELU-MASK X))
(grad '(tsum (matmul x y)) 'x)  ; → (MATMUL (ONES-LIKE (MATMUL X Y)) (TRANSPOSE Y))
```

Правила покрывают `+ - * / ^`, `tsum/tmean/tmax`, `matmul`, `relu`, `softmax` (1-D), `transpose`, `reshape`, `ones-like`/`tmax-mask`/`relu-mask` и унарные `sin…atan`; `diff` на `relu`, `matmul`, `transpose`, `reshape`, `softmax` отвечает внятным отказом с указанием перейти к `grad`. Для n-арной арифметики: `(diff '(+ x 1 2) 'x)` → `1`, `(diff '(* 2 3 x) 'x)` → `6`, `(diff '(sin x) 'x)` → `(COS X)`.

`(^ base n)` — синтаксис символьных термов, а не функция языка; для составного основания подтерм связывается один раз через `let`: `(^ (+ x 1) 3)` → `(LET ((POW-0 (+ X 1))) (* POW-0 POW-0 POW-0))`.

`collect` — нормальная форма: `(* x x)` → `(^ x 2)`, `-` сворачивается в сумму с отрицательным коэффициентом, порядок — первое появление. Списки термов типизируются как `:syntax`, а не `(:list :syntax)`: `cons` с `:syntax`-аргументом даёт `:syntax`.

> 🐍 `diff` ≈ `sympy.diff`, `simplify` ≈ `sympy.simplify`, `unify` ≈ логический вывод с унификацией; только термы — те же самые S-выражения, что и код программы.

## 15.9 neuro — мост терм → тензор

Три экспорта: макрос `term`, функция `term->code` и `term-problem`.

```lisp
(import neuro)

(defun f ((x . (:tensor :f32 (4))))
  ((term '(+ (sin x) (* x x)) '(x)) . (:tensor :f32 (4))))

(defvar c ((param (tensor (2.0))) . (:tensor :f32 (1))))
(defgraph sq ((c . (:tensor :f32 (1))) (x . (:tensor :f32 (4))))
  ((term '(tsum (* c (* x x))) '(c x)) . :flonum))   ; grad! заполнит grad-of c
```

| Возможность | Что транслируется |
|---|---|
| Арифметика | n-арные `+ - * /`, унарный `-`; скаляры идут broadcast |
| Матрицы | `matmul`, `transpose` |
| Унарные | `sin cos tanh exp log sqrt relu atan`, `softmax` |
| Свёртки | `tsum tmean tmax` |
| Формо-сохраняющие ops | `ones-like`, `tmax-mask`, `relu-mask`, `reshape-like`, `reshape` |
| Степень | `^ n` → произведение; `n=0` → `1.0`, `n<0` → `(/ 1.0 (* …))`; составное основание связывается один раз через `let` |

Оба аргумента `term` обязаны быть **литералами**: макрос раскрывается до типизации. `term->code` — то же над значением `:syntax`. Переменные обязаны быть перечислены в `vars`: глобал в графе — константа без градиента. Проверка отделена от генерации: `term-problem` возвращает первое сообщение, не бросая; `term->code` бросает ровно один раз — иначе после `throw` вторичная диагностика («unsupported atom 1» вместо «unknown variable Y»). Один терм годится и для eager-функции, и для `defgraph` (в том числе в ветках `ns-if`), и для сверки аналитического `diff` с autograd — см. гл. 17.

> 🐍 Это как `sympy.lambdify`, только результатом является statically типизированный код над тензорами, который можно сразу трассировать в `defgraph`.

## 15.10 tablet — таблица данных

`tablet` — числовая таблица одного dtype `:f32`: имена колонок + один 2-D тензор. API — псевдо-ООП: `deftrait tablet-ops` + `defstruct tablet` + `impl`, диспетч по тегу первого аргумента. Модуль экспортирует только `(tablet make-tablet)`; методы приходят вместе со структурой и отдельными экспортами не являются.

```lisp
(import tablet)

(defvar tab ((make-tablet (list 'a 'b)
                          (tensor ((1.0 2.0) (3.0 4.0)))) . (:struct tablet)))
(nrows tab)          ; → 2
(ncols tab)          ; → 2
(column-names tab)   ; → (A B)
(head tab 1)         ; → (:struct tablet) с одной строкой
```

| Метод | Результат | Python |
|---|---|---|
| `nrows`, `ncols`, `column-names` | метаданные | `df.shape`, `df.columns` |
| `column` | `(:tensor :f32 (_))` | `df[name]` |
| `row` | `(:tensor :f32 (_ _))` формы `(1 C)` | `df.iloc[[i]]` |
| `head` | первые `n` строк | `df.head(n)` |
| `where` | фильтр по маске > 0 | `df[mask]` |
| `select-columns` | проекция/перестановка колонок через `transpose` + `t-select-rows` | `df[cols]` |
| `sort-by` | сортировка строк по колонке (индексы сортируются общим `sort`) | `df.sort_values(by)` |

Формы тензоров в слотах — wildcard `(_ _)`; конкретная форма `(N C)` в слот не пишется, `(t-shape data)` доступен только в рантайме. Нулевых таблиц нет: `where` без совпадений — `throw`, `head 0` разрешён. Методы чистые: `t-*`-операции копируют.

> 🐍 `tablet` ≈ `pandas.DataFrame`, но только с колонками `:f32` (гетерогенные dtype потребовали бы sum-типов) и без индекса — вместо него позиция строки.

## 15.11 import и import-from

```lisp
(import ml)                 ; → ML — все публичные экспорты модуля
(import-from core (identity caar))   ; → nil — только названные имена
(identity 5)                ; → 5
```

| Форма | Действие |
|---|---|
| `(import M)` | загрузить модуль (если ещё не загружен) и открыть **все** его экспорты; возвращает имя модуля символом |
| `(import-from M (a b …))` | загрузить модуль целиком, но открыть только перечисленные экспорты; возвращает `nil` |

Важно: `(import-from m (a))` всё равно **исполняет верхний уровень всего модуля** и компилирует его — экономится не загрузка, а только число видимых имён. Повторный импорт — дешёвый no-op. Ошибки: `UnknownModule` для неизвестного модуля, `UnknownExport` для имени, которого модуль не экспортирует, `BadImport` для несимвола/пустой формы.

```lisp
(import FFI)                        ; → error: UnknownModule — FFI не модуль, а примитивы
(import-from string (string-split)) ; → error: UnknownExport — примитив не экспорт модуля
```

> 🐍 `(import m)` ≈ `import m`, `(import-from m (a))` ≈ `from m import a`. Отличие: приватные имена недостижимы на практике, а модуль — не объект first-class, его нельзя присвоить переменной.

## 15.12 Приватные имена, экспорты-алиасы, ImportConflict

Всё, что модуль определяет, живёт под **недостижимым из исходника id**: обратиться к приватному имени через `module/name` или как к глобалу нельзя — это подтверждается на практике: `(symbolic/diff …)` → `unknown function SYMBOLIC/DIFF`, хотя `(import symbolic)` делает `diff` доступным.

Экспорт — это **алиас** на публичное имя. При импорте алиас проецируется в текущее пространство имён. Правила:

- приватные имена при импорте не открываются, даже если модуль исполнен целиком;
- структура экспортируется как единица: `(export tablet make-tablet)` открывает тег, конструктор и интерфейс методов; отдельного экспорта метода нет;
- конфликт двух публичных имён с одним названием — **`ImportConflict`**, а не «последний победил». Именно поэтому `mse` из `ml` и тензорная потеря названы `mse`/`mse-loss`, а теги структур в `dl` сделаны dl-специфичными;
- импорт — проекция приватного тега на публичный алиас, поэтому два модуля могут определять одноимённые структуры без конфликтов в пользовательском коде.

> 🐍 Ближе всего `from m import a as b` и `_private`-соглашение, только приватность здесь обеспечивается не соглашением, а недостижимостью идентификатора; конфликт — жёсткая ошибка импорта, а не перезапись.

## 15.13 Ошибки и бектрейс

Ошибки — значения `(:result T)` (`ok T | err Error`) плюс примитивы `throw : :a → :never` и `catch-error : (:func () T) → (:result T)`. `try` как форма в бинаре недоступна (`unknown function TRY`); рабочий путь — `catch-error` + `result-ok?`/`result-value`/`result-message`, а для разбора неудачи — `result-error` и `format-error`.

Модуль `error` уже загружен как prelude; `(import error)` можно написать явно для читаемости. Он определяет две структуры и типизированный доступ к ним:

```lisp
(defstruct call-frame
  ((name . :symbol)
   (file . :string)
   (line . :fixnum)
   (form . :string)))

(defstruct error
  ((message . :string)
   (cause . (:option :atom))
   (traceback . (:option (:list (:struct call-frame))))
   (kind . :symbol)))
```

Категории — интернированные символы, сравниваются `eq`: `runtime-error` = `'runtime`, `parse-error` = `'parse`, `type-error` = `'type`, `io-error` = `'io`, `user-error` = `'user`. Конструкторы и аксессоры: `make-error message kind`, `with-cause e cause`, `error-message`, `error-cause`, `error-traceback`, `error-kind`, `error-of?` (сравнивает `kind` с символом), `call-frame-name/-file/-line/-form`. Экспортируются и сами структуры, и категории.

`throw` кладёт брошенное значение в `cause`, в `message` пишет его печатное представление, `kind` = `'user`; у рантайм-ошибки `kind` = `'runtime`. Кадры `call-frame` наполняются из source map Q4a: `name/file/line/form`, при отсутствии места — `file="<unknown>"`, `line=0`, `form=""`.

Два примитива связывают структуру с печатью:

| Примитив | Контракт | Что делает |
|---|---|---|
| `result-error` | `(:result T) → (:struct error)` | достаёт структуру ошибки из неудачного результата; на ok-результате — `RuntimeError` |
| `format-error` | `(:struct error) → :string` | собирает текст `Error: <message> (<kind в нижнем регистре>)`, затем строку `cause:` (если причина отличается от сообщения) и кадры `at <name>  <file>:<line>  in <form>` |

Кадры без источника печатаются без фиктивного `<unknown>:0`; внутренние теги (`ERROR::…`) в отчёт не попадают. Пример из REPL (диагностика `error-message` и `format-error` проверена бинарём):

```lisp
(import error)
(result-error (catch-error (lambda () (throw 42))))
;; → #<ERROR::ERROR "42" 42 (#<ERROR::CALL-FRAME LAMBDA "<repl>" 1 "(THROW 42)">) USER>
(error-message (result-error (catch-error (lambda () (throw 42))))) ; → "42"
(error-kind (result-error (catch-error (lambda () (throw 42)))))    ; → USER
(error-cause (result-error (catch-error (lambda () (throw 42)))))   ; → 42
(format-error (result-error (catch-error (lambda () (throw 42)))))
;; → "Error: 42 (user)\n  at LAMBDA  <repl>:1  in (THROW 42)"
```

Принтер структур печатает `#<tag …>` вместо `#<struct>`; тег ошибки — приватный псевдоним модуля (`ERROR::ERROR`, `ERROR::CALL-FRAME`), а `print` и `write-to-string` расходятся только в кавычках: `print` даёт `#<ERROR::ERROR x nil nil IO>`, `write-to-string` — `"#<ERROR::ERROR \"x\" nil nil IO>"`. Глобальной функции `traceback` нет — доступ к кадрам даёт `error-traceback` (`unknown function TRACEBACK` при попытке вызвать `traceback`). На кадре из `error-traceback` аксессоры `call-frame-*` в текущем бинаре дают `no overload matches the arguments`; для чтения трейса используйте `format-error`.

Непойманный `throw` в файловом раннере печатает `file:line:col: message` и завершается с exit 1; в REPL — только `error: RuntimeError`.

| Поле кадра | Что несёт | Python |
|---|---|---|
| `name` | имя функции | `FrameSummary.name` |
| `file` | файл-источник | `filename` |
| `line` | номер строки | `lineno` |
| `form` | форма, вычислявшаяся в кадре | `line` (текст, а не AST) |

Порядок кадров — oldest → newest. Кадр из JIT-кода несёт **мастер-форму** вызываемого: нативная страница — выбрасываемый кэш S-выражения. Хвостовые вызовы (TCO) схлопывают кадры: хвостовой цикл даёт один кадр независимо от глубины — это контракт, а не потеря информации. Поэтому в демо `examples/error-demo.qlsp` функции `middle`/`outer` намеренно не хвостовые — иначе в трейсе не было бы нескольких кадров.

> 🐍 `traceback.extract_tb(e.__traceback__)` отдаёт `FrameSummary(filename, lineno, name, line)`; Qualisp-кадр вместо текста строки хранит саму форму, а схлопывание хвостовых кадров — семантика Scheme, в Python её нет. `error-kind` и `error-of?` заменяют разбор текста исключения вручную.

## 15.14 Загрузка модулей: лениво и идемпотентно

Загрузчик раскрывает макросы модуля **до проверки** — поэтому модуль может пользоваться своими макросами (это опровергает распространённый комментарий в `io.qlsp`, но подтверждено поведением загрузчика и test-only `probe.qlsp`). Тело `defgraph`/`defuse` тоже раскрывает макросы и клонируется в world, иначе тело `defuse` висело бы на scratch.

Правила загрузки:

- модуль грузится лениво при первом `import`/`import-from`;
- поиск идёт по порядку **встроенные модули → подключённые `.qlsl`-пакеты**; исходник из пакета читается лениво и освобождается после загрузки;
- повторный импорт идемпотентен, в том числе из пакета; тот же архив можно передать повторно, другой пакет с тем же именем даёт `DuplicatePackage`;
- каталог `--path` сканируется лексикографически; `QUALISP_PATH` разделяется разделителем путей ОС;
- `qualisp.lock` (флаг `--lock`, env `QUALISP_LOCK`, автопоиск рядом со скриптом) проверяет SHA-256, имя и версию всех пакетов до применения; при активном lock поисковые пути и соседний edge-`stdlib.qlsl` не используются;
- `core` и `error` — предустановленный prelude, остальные требуют явного импорта;
- в edge-сборке (`-Dembed-stdlib=false`) встроены только `core`+`error`; остальные 11 модулей ищутся в `stdlib.qlsl` рядом с бинарником (`bin/../share/qualisp/stdlib.qlsl`) или в пакете, переданном `--package`;
- явного `MODULE`-заголовка с описью зависимостей в 0.19.20 нет — зависимости модуля определяются его телом;
- `macroexpand-all` раскрывает рекурсивно, не заходя в `quote`.

CLI раннера: `qualisp [--trace <file.json>] [--package <pkg.qlsl>] [--path <dir>] [--lock <file>] [--no-lock] [script.qlsp [args…]]`; `--package` и `--path` можно повторять. Флаги интерпретатора разбираются только до имени скрипта, всё после — program args (`command-line-args`). Файловый раннер не печатает результат последней формы — только явный `print`.

> 🐍 Как импорт Python: тело модуля исполняется при первом импорте, дальше — из `sys.modules`; только зависимости здесь не описываются заголовком, а выводятся из самого кода.

## 15.15 Сводка модулей

| Модуль | Назначение | Python-аналог |
|---|---|---|
| `core` | `caar`, `identity`, prelude | `builtins` |
| `error` | `error`/`call-frame`, категории, `result-error`, `format-error`, prelude | `Exception` + `traceback` |
| `string` | поиск, паддинг, строки/слова, `string-to-fixnum` | `str` |
| `collections` | `sum`, `find`, `position`, `take/drop-while`, `zip-with`, `unique` | `functools` + `itertools` |
| `test` | `assert-fixnum`, `assert-string` | `assert` |
| `io` | файлы, `read-lines`, `with-file`, `each-line` | `open` + контекстный менеджер |
| `ml` | метрики, split, масштабирование, `t-mse` | `sklearn.metrics` / `preprocessing` |
| `dl` | слои, LayerNorm/attention/embedding, conv/pool, dropout, потери, init, SGD/Adam/momentum | `torch.nn` / `torch.optim` |
| `symbolic` | match/rewrite/diff/grad/unify, полиномы | `sympy` |
| `argparse` | флаги, значения опций, позиционные аргументы | `argparse` |
| `tablet` | числовая таблица `:f32`, методы через `deftrait` | `pandas.DataFrame` |
| `datetime` | UTC, epoch, ISO-8601, таймеры | `datetime` (UTC-часть) |
| `neuro` | символьный терм → тензорный код, `term-problem` | `sympy.lambdify` |

Итого 13. `probe.qlsp` и `typepairs_probe.qlsp` — test-only фикстуры, не модули.

## 15.16 collections — свёртки и поиск по спискам

`collections` — 14 экспортов. Важное ограничение языка: `cdr` от `(:list T)` даёт `(:option (:list T))`, поэтому обход списков сделан рекурсией **по индексу** через `nth` ( `(nth список индекс)` ), а не по cons-цепочке. Там, где результат может опустеть, тип — `(:option …)`, потому что пустого списка в языке нет.

```lisp
(import collections)

(sum (list 1 2 3))                ; → 6
(keep-positive (list -1 2 3))     ; → (2 3); всё отрицательное → nil
(any (lambda ((x . :fixnum)) (> x 2)) (list 1 2 3))   ; → t
(all (lambda ((x . :fixnum)) (> x 0)) (list 1 2 3))   ; → t
(count-if (lambda ((x . :fixnum)) (> x 1)) (list 1 2 3)) ; → 2
(last (list 1 2 3))               ; → 3
(range 5)                         ; → (0 1 2 3 4); n ≤ 0 → nil
(member? 2 (list 1 2 3))          ; → t
(find (lambda ((x . :fixnum)) (> x 2)) (list 1 2 3))  ; → 3
(position (lambda ((x . :fixnum)) (= x 2)) (list 1 2 3)) ; → 1
(take-while (lambda ((x . :fixnum)) (< x 3)) (list 1 2 3)) ; → (1 2)
(drop-while (lambda ((x . :fixnum)) (< x 3)) (list 1 2 3)) ; → (3)
(unique (list 1 2 1 3))           ; → (1 2 3) — порядок сохранён
(zip-with (lambda ((a . :fixnum) (b . :fixnum)) (+ a b)) (list 1 2) (list 10 20)) ; → (11 22)
```

| Экспорт | Сигнатура (упрощённо) | Заметка |
|---|---|---|
| `sum` | `(:list :fixnum) → :fixnum` | `reduce` с нулём |
| `keep-positive` | `(:list :fixnum) → (:option (:list :fixnum))` | пустой результат — `nil` |
| `any`, `all`, `count-if` | предикат + список | `reduce` + `or`/`and` |
| `last` | `(:list :a) → :a` | `car (reverse xs)` |
| `range` | `:fixnum → (:option (:list :fixnum))` | `0..n-1` |
| `find`, `position` | предикат + список | первый satisfying элемент / его индекс |
| `take-while`, `drop-while` | предикат + список | option-список |
| `zip-with` | `f` + два списка | до короче; option-список |
| `member?` | элемент + option-список | `equal` |
| `unique` | `(:list :a) → (:list :a)` | первый элемент остаётся всегда |

`map`/`filter`/`reduce` — **глобальные примитивы**, не экспорты `collections`; их колбэки обязаны быть типизированы: `(lambda ((x . :fixnum)) (* x x))`. У `lambda` аннотируются только параметры, тип результата выводится.

> 🐍 `find` ≈ `next(filter(p, xs), None)`, `position` ≈ `next(i for i,x in enumerate(xs) if p(x))` с `None` вместо `StopIteration`, `take/drop-while` ≈ `itertools.takewhile/dropwhile`, `zip-with` ≈ `map(f, xs, ys)`.

## 15.17 argparse — аргументы командной строки

Примитив `(command-line-args)` → `(:list :string)`; первый элемент — путь скрипта (argv[0]), список непуст. Флаги интерпретатора разбираются только до имени скрипта; всё после — program args, включая `--foo`.

Модуль `argparse` — **сканер, а не парсер с словарём опций**: приложение само решает, какие опции существуют и как преобразовать значения. Имена передаются без дефисов: `"verbose"` соответствует `--verbose` и `-verbose`.

```lisp
(import argparse)

(print (command-line-args))
(print (flag? (command-line-args) "verbose"))
(print (option-value (command-line-args) "name"))
(print (positionals (command-line-args) (list "name")))
```

```bash
$ qualisp examples/args.qlsp --verbose --name lena input.txt
(examples/args.qlsp --verbose --name lena input.txt)
t
lena
(input.txt)
```

| Экспорт | Контракт | Python |
|---|---|---|
| `flag?` | есть ли `--name` или `-name` | `('--name' in sys.argv)` |
| `option-value` | `--name value` или `--name=value`, иначе `nil` | ручной разбор `sys.argv` |
| `require-option` | то же, но absence — `throw` | `parser.error` |
| `positionals` | всё после argv[0], кроме опций и их значений; список `value-options` обязателен | `[a for a in argv[1:] if not a.startswith('-')]` |

Без явного списка `value-options` сканер не отличит `--out x.txt` (значение) от `--verbose input.txt` (позиционный аргумент).

> 🐍 Это не `argparse.ArgumentParser` с автогенерацией `--help`, а скорее `sys.argv` + несколько аккуратных хелперов: конфигурации нет, типы не конвертируются, зато статические типы известны заранее.

## 15.18 datetime — UTC, ISO-8601 и таймеры

`datetime` — один из 13 встроенных модулей. Он грузится лениво: `(import datetime)`, как и у остальных модулей, кроме prelude `core`/`error`. Время берётся из примитивов поверх `std.Io.Clock`, одинаково на Linux и Windows: `(now)` и `(now-ms)` — epoch UTC в секундах/мс (`:fixnum`), `(monotonic-ns)` — монотонные часы для интервалов, `(sleep-ms n)` — пауза (`:fixnum → :nil`).

```lisp
(import datetime)

(epoch->utc 0)          ; → #<DATETIME::UTC-DATETIME 1970 1 1 0 0 0>
(utc->epoch (epoch->utc 0))  ; → 0
(iso-utc 0)             ; → "1970-01-01T00:00:00Z"
(iso-utc-date 0)        ; → "1970-01-01"
(parse-iso-utc "1970-01-01T00:00:00Z")  ; → 0
(seconds->hms 3661)     ; → (1 1 1)
(now-iso)               ; → "2026-11-22T20:44:22Z" — живое время
(timer-elapsed-ms (timer-start))  ; → 0
(elapsed (+ 1 2))       ; печатает "elapsed: 0 ms", возвращает 3
```

Экспорт модуля: `utc-datetime epoch->utc utc->epoch iso-utc iso-utc-date parse-iso-utc now-iso seconds->hms timer-start timer-elapsed-ms elapsed-ms elapsed div-floor pad`.

| Экспорт | Контракт | Что делает |
|---|---|---|
| `utc-datetime` | `(year month day hour minute second)` — все `:fixnum` | структура UTC-даты |
| `epoch->utc`, `utc->epoch` | `:fixnum ↔ (:struct utc-datetime)` | алгоритм Хиннанта (`civil_from_days`/`days_from_civil`); даты до 1970 работают |
| `iso-utc`, `iso-utc-date` | `:fixnum → :string` | `YYYY-MM-DDTHH:MM:SSZ` и первые 10 символов |
| `parse-iso-utc` | `:string → :fixnum` | принимает и дату без времени; ошибки — `throw` с текстами `datetime: timestamp is too short`, `datetime: expected YYYY-MM-DD`, `datetime: expected T after the date`, `datetime: invalid integer in timestamp` |
| `seconds->hms` | `:fixnum → (:list :fixnum)` | часы/минуты/секунды |
| `timer-start`, `timer-elapsed-ms`, `elapsed-ms` | `:fixnum`-метка времени, миллисекунды | таймеры на `monotonic-ns` |
| `elapsed` | макрос | печатает `elapsed: … ms` и возвращает значение формы |
| `div-floor`, `pad` | `:fixnum :fixnum → :fixnum`, `:string :fixnum → :string` | утилиты (эвклидово деление и ведущие нули) |

Дробное деление в языке всегда даёт flonum, поэтому целое деление в модуле сделано через `to-fixnum` и поправку на отрицательный остаток; `div-floor -7 2` даёт `-4`. `or-else` — **функция, без короткозамыкания**: `(or-else v (throw …))` бросает всегда, поэтому в `parse-int` используется `if` + `null`. Макрос `elapsed` написан в позиционной форме `(defmacro elapsed (expr :syntax) :syntax …)` — это исключение из общего правила парных аннотаций (гл. 12).

Локальное время и таймзоны в v1 не входят: все функции работают в UTC. В `src/stdlib_datetime_test.zig` шесть именных тестов: epoch zero, leap years, ISO roundtrip, ISO errors, time splits and timers, primitives typed.

> 🐍 `epoch->utc`/`utc->epoch` ≈ `datetime.datetime.utcfromtimestamp` и `.timestamp()`, `iso-utc` ≈ `.strftime("%Y-%m-%dT%H:%M:%SZ")`, а `timer-elapsed-ms` ≈ `time.monotonic()` с разницей в миллисекундах; таймзоны и `localtime()` пока не поддерживаются.
