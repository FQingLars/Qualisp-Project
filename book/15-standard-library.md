# Глава 15. Стандартная библиотека

Стандартная библиотека Qualisp 0.19.15 — это **11 модулей**, написанных на самом Qualisp и вшитых в бинарник через `@embedFile` (`src/stdlib.zig`). Никакой внешней среды с `pip`-пакетами: всё, что нужно программе, уже лежит в исполняемом файле. `core` грузится при старте как **prelude**, остальные модули подключаются **лениво** при первом `import`; повторный импорт идемпотентен. Порядок загрузки в `Session.init` фиксирован: `core`, `string`, `collections`, `test`.

| Модуль | Файл | Экспорты | Содержание | Python-аналог |
|---|---|---|---|---|
| `core` | `src/stdlib/core.qlsp` | 2 | `caar`, `identity` — базовые утилиты над `:syntax` и generic-тождество | `builtins` |
| `string` | `src/stdlib/string.qlsp` | 11 | строки: поиск, паддинг, разбиение на строки/слова, приведение к fixnum | `str` |
| `collections` | `src/stdlib/collections.qlsp` | 14 | свёртки, поиск, срезы по предикату, `unique`, `zip-with` | `functools` + `itertools` |
| `test` | `src/stdlib/test.qlsp` | 2 | `assert-fixnum`, `assert-string` — аварийные утверждения | `assert` / `pytest` |
| `io` | `src/stdlib/io.qlsp` | 4 | файлы, построчное чтение, потоковые хэндлы, макрос `with-file` | `open`, `pathlib` |
| `ml` | `src/stdlib/ml.qlsp` | 13 | метрики, разбиение выборки, масштабирование, `t-mse` | `sklearn.metrics` + `sklearn.preprocessing` |
| `dl` | `src/stdlib/dl.qlsp` | 17 | логиты слоёв, MSE, SGD/Adam/momentum, замкнутый шаг линейной регрессии | `torch.nn` + `torch.optim` |
| `symbolic` | `src/stdlib/symbolic.qlsp` | 16 | сопоставление, переписывание, упрощение, `diff`, `unify`, полиномы | `sympy` |
| `argparse` | `src/stdlib/argparse.qlsp` | 4 | сканер аргументов: `flag?`, `option-value`, `require-option`, `positionals` | `argparse` (сканирование) |
| `tablet` | `src/stdlib/tablet.qlsp` | 2 | числовая таблица: `defstruct` + `deftrait`/`impl`, методы на `:f32` | `pandas.DataFrame` |
| `neuro` | `src/stdlib/neuro.qlsp` | 2 | мост символьный терм → тензорный код: `term`, `term->code` | (нет прямого аналога) |

Есть ещё **`src/stdlib/probe.qlsp`** — но это test-only фикстура: она вшивается только в test-сборки (`builtin.is_test`) и **не является модулем для пользователя**. Модулей ровно одиннадцать.

> 🐍 «11 вшитых модулей в одном бинарнике» — это монолитный Docker-образ с предустановленными `numpy/torch/sklearn/pandas/sympy`, только без GIL, без 4 ГБ зависимостей и без `pip install`.

Модули `audio`, `async`, `ns`, `pkg` в релиз не входят; `qualisp-lsp` и `qvalent` в исходниках отсутствуют. Вместо тестового фреймворка `deftest`/`run-tests` есть только `assert-*`, падающие через `throw`.

## 15.1 core — всегда загружен

`core` — единственный модуль, который не нужно импортировать: он установлен как prelude при старте сессии. Два экспорта:

```lisp
(caar x :syntax) :syntax = (car (car x))
(identity x :a) :a       ; x
```

```lisp
(caar '((1 2) 3))   ; → 1
(identity 5)        ; → 5
```

`identity` — generic-функция с переменной типа `:a`: это демонстрация параметрического полиморфизма. `caar` работает над `:syntax`, а не над `(:list T)`: если нужен `car` от обычного списка, он доступен как примитив.

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
(each-line "/tmp/t.txt" (lambda (s :string) :bool (print s)))  ; → nil
```

`(with-file (name path mode) body...)` — **макрос**: тело остаётся обычным кодом, файл закрывается и на успехе, и на `throw`. Внутри используется `catch-error` + `result-ok?`/`result-value`/`result-message`; `throw` внутри тела перевыбрасывается после закрытия. `each-line` явно повторяет тот же протокол open/catch/close. `or-else` распаковывает option после `null`-проверки:

```lisp
(let ((line (read-line h)))
  (if (null line) nil (funcall f (or-else line ""))))
```

CSV, regex, аудио и работа с датой-временем в stdlib 0.19.15 **не входят**: для разбора строк используйте `string-split`/`substr`, для данных — `tablet`.

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

17 экспортов, все чистые: переданные тензоры не мутируются, результат — новый тензор или структура. Работает только `:f32`, только 2-D `batch × features`; bias — строка `(1 m)`, а не вектор `(m)`.

| Группа | Экспорты | Что делает |
|---|---|---|
| Прямой ход | `linear-forward`, `mlp-forward` | логиты, активация — выбор вызывающего |
| Потери | `mse-loss`, `squared-error` | скалярная MSE по тензору / по числу |
| Оптимизаторы | `sgd-step`, `adam-init`, `adam-step`, `adam-param`, `adam-steps`, `momentum-init`, `momentum-step`, `momentum-param` | SGD, Adam (β₁=0.9, β₂=0.999, eps=1e-7, bias correction), momentum |
| Без графа | `linear-grads`, `linear-regression-step` | замкнутый градиент линейной регрессии через `matmul`/`transpose`/`tsum` |
| Структуры | `adam-state`, `linear-gradients`, `linear-model`, `momentum-state` | состояние в `defstruct` |

```lisp
(import dl)

(defvar wx (:tensor :f32 (_ _)) (tensor ((1.0 2.0))))
(defvar gy (:tensor :f32 (_ _)) (tensor ((0.5 -0.5))))
(print (mse-loss wx gy))           ; → скаляр :flonum
(print (linear-forward wx gy gy))  ; → (_ _) — логиты
(defvar st (:struct adam-state) (adam-init wx))
(setq st (adam-step st gy 0.01))
(print (adam-steps st))            ; → 1
```

В autograd-цикле веса обновляются **на месте** `tset!` по `(grad-of w)`, а не `sgd-step`: результат шага имеет wildcard-форму `(_ _)` и не присваивается конкретной форме `(2 1)` (см. гл. 17). Имя `mse` уже занято модулем `ml`, поэтому тензорная потеря называется `mse-loss` — два экспорта с одним именем из разных модулей дали бы `ImportConflict`.

> 🐍 `adam-step` ≈ `optimizer.step()` в PyTorch, только состояние — обычная структура `adam-state`, а не скрытый словарь optimizer'а; `linear-regression-step` ≈ ручной `W -= lr * Xᵀerr` без autograd.

## 15.8 symbolic — символьный движок

16 экспортов. Терм — `:syntax`, обычное S-выражение; переменные образца — символы с префиксом `?`. Окружение — alist; **пустое окружение — `t`**, а не `nil` (nil уже значит «не совпало»). `subst` принимает `t`, `nil` и настоящий alist.

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
| Дифференцирование | `diff`, `diff-canonical` | `+ - * / ^`, упрощение; `diff` **не приводит подобные**: `d(x*x)/dx = (+ x x)`; `diff-canonical` даёт `(* 2 x)` |
| Логика | `unify`, `derive`, `matches` | унификация с occurs-check; `derive`/`matches` возвращают `:syntax` |
| Полиномы (S3) | `collect`, `poly?`, `poly-coeffs`, `poly-eval` | нормальная форма «сумма мономов»; плотные коэффициенты от степени 0; нулевой полином — `(0)` |

`collect` — нормальная форма: `(* x x)` → `(^ x 2)`, `-` сворачивается в сумму с отрицательным коэффициентом, порядок — первое появление. Списки термов типизируются как `:syntax`, а не `(:list :syntax)`: `cons` с `:syntax`-аргументом даёт `:syntax`.

> 🐍 `diff` ≈ `sympy.diff`, `simplify` ≈ `sympy.simplify`, `unify` ≈ логический вывод с унификацией; только термы — те же самые S-выражения, что и код программы.

## 15.9 neuro — мост терм → тензор

Два экспорта: макрос `term` и функция `term->code`.

```lisp
(import neuro)

(defun f (x (:tensor :f32 (4))) (:tensor :f32 (4))
  (term '(+ (sin x) (* x x)) '(x)))

(defvar c (:tensor :f32 (1)) (param (tensor (2.0))))
(defgraph sq (c (:tensor :f32 (1)) x (:tensor :f32 (4))) :flonum
  (term '(tsum (* c (* x x))) '(c x)))   ; grad! заполнит grad-of c
```

| Возможность | Что транслируется |
|---|---|
| Арифметика | `+ - * /`, унарный `-`; скаляры идут broadcast |
| Матрицы | `matmul` |
| Унарные | `sin cos tanh exp log sqrt relu atan` |
| Свёртки | `tsum tmean tmax` |
| Степень | `^ n` → произведение; `n=0` → `1.0`, `n<0` → `(/ 1.0 (* …))` |

Оба аргумента `term` обязаны быть **литералами**: макрос раскрывается до типизации. `term->code` — то же над значением `:syntax`. Переменные обязаны быть перечислены в `vars`: глобал в графе — константа без градиента. Проверка отделена от генерации: `term-problem` возвращает первое сообщение, не бросая; `term->code` бросает ровно один раз — иначе после `throw` вторичная диагностика («unsupported atom 1» вместо «unknown variable Y»). Один терм годится и для eager-функции, и для `defgraph` (в том числе в ветках `ns-if`), и для сверки аналитического `diff` с autograd — см. гл. 17.

> 🐍 Это как `sympy.lambdify`, только результатом является statically типизированный код над тензорами, который можно сразу трассировать в `defgraph`.

## 15.10 tablet — таблица данных

`tablet` — числовая таблица одного dtype `:f32`: имена колонок + один 2-D тензор. API — псевдо-ООП: `deftrait tablet-ops` + `defstruct tablet` + `impl`, диспетч по тегу первого аргумента. Модуль экспортирует только `(tablet make-tablet)`; методы приходят вместе со структурой и отдельными экспортами не являются.

```lisp
(import tablet)

(defvar tab (:struct tablet) (make-tablet (list 'a 'b)
                                 (tensor ((1.0 2.0) (3.0 4.0)))))
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

Ошибки — значения `(:result T)` (`ok T | err Error`) плюс примитивы `throw : :a → :never` и `catch-error : (:func () T) → (:result T)`. `try` как форма в REPL-пробе недоступен; рабочий путь — `catch-error` + `result-ok?`/`result-value`/`result-message`.

`Error` несёт `message`, `cause`, `traceback`; бэктрейс — кадры `(name file line form)`. Наконец, source maps в эмиттере дают позицию в ошибках рантайма: CLI печатает `file:line:col: message`, а при отсутствии diag — `place: message` из `error_location`, иначе `error: …`.

| Поле кадра | Что несёт | Python |
|---|---|---|
| `name` | имя функции | `FrameSummary.name` |
| `file` | файл-источник | `filename` |
| `line` | номер строки | `lineno` |
| `form` | форма, вычислявшаяся в кадре | `line` (текст, а не AST) |

Порядок кадров — oldest → newest. Кадр из JIT-кода несёт **мастер-форму** вызываемого: нативная страница — выбрасываемый кэш S-выражения. Хвостовые вызовы (TCO) схлопывают кадры: хвостовой цикл даёт один кадр независимо от глубины — это контракт, а не потеря информации.

> 🐍 `traceback.extract_tb(e.__traceback__)` отдаёт `FrameSummary(filename, lineno, name, line)`; Qualisp-кадр вместо текста строки хранит саму форму, а схлопывание хвостовых кадров — семантика Scheme, в Python её нет.

## 15.14 Загрузка модулей: лениво и идемпотентно

Загрузчик раскрывает макросы модуля **до проверки** — поэтому модуль может пользоваться своими макросами (это опровергает распространённый комментарий в `io.qlsp`, но подтверждено поведением загрузчика и test-only `probe.qlsp`). Тело `defgraph`/`defuse` тоже раскрывает макросы и клонируется в world, иначе тело `defuse` висело бы на scratch.

Правила загрузки:

- модуль грузится лениво при первом `import`/`import-from`;
- повторный импорт идемпотентен;
- `core` — предустановленный prelude, остальные требуют явного импорта;
- явного `MODULE`-заголовка с описью зависимостей в 0.19.15 нет — зависимости модуля определяются его телом;
- `macroexpand-all` раскрывает рекурсивно, не заходя в `quote`.

CLI раннера: `qualisp [--trace <file.json>] [script.qlsp [args…]]`; флаги интерпретатора разбираются только до имени скрипта, всё после — program args (`command-line-args`). Файловый раннер не печатает результат последней формы — только явный `print`.

> 🐍 Как импорт Python: тело модуля исполняется при первом импорте, дальше — из `sys.modules`; только зависимости здесь не описываются заголовком, а выводятся из самого кода.

## 15.15 Сводка модулей

| Модуль | Назначение | Python-аналог |
|---|---|---|
| `core` | `caar`, `identity`, prelude | `builtins` |
| `string` | поиск, паддинг, строки/слова, `string-to-fixnum` | `str` |
| `collections` | `sum`, `find`, `position`, `take/drop-while`, `zip-with`, `unique` | `functools` + `itertools` |
| `test` | `assert-fixnum`, `assert-string` | `assert` |
| `io` | файлы, `read-lines`, `with-file`, `each-line` | `open` + контекстный менеджер |
| `ml` | метрики, split, масштабирование, `t-mse` | `sklearn.metrics` / `preprocessing` |
| `dl` | логиты, MSE, SGD/Adam/momentum, линейная регрессия | `torch.nn` / `torch.optim` |
| `symbolic` | match/rewrite/diff/unify, полиномы | `sympy` |
| `argparse` | флаги, значения опций, позиционные аргументы | `argparse` |
| `tablet` | числовая таблица `:f32`, методы через `deftrait` | `pandas.DataFrame` |
| `neuro` | символьный терм → тензорный код | `sympy.lambdify` |

Итого 11. `probe.qlsp` — test-only фикстура, не двенадцатый модуль.

## 15.16 collections — свёртки и поиск по спискам

`collections` — 14 экспортов. Важное ограничение языка: `cdr` от `(:list T)` даёт `(:option (:list T))`, поэтому обход списков сделан рекурсией **по индексу** через `nth` ( `(nth список индекс)` ), а не по cons-цепочке. Там, где результат может опустеть, тип — `(:option …)`, потому что пустого списка в языке нет.

```lisp
(import collections)

(sum (list 1 2 3))                ; → 6
(keep-positive (list -1 2 3))     ; → (2 3); всё отрицательное → nil
(any (lambda (x :fixnum) :bool (> x 2)) (list 1 2 3))   ; → t
(all (lambda (x :fixnum) :bool (> x 0)) (list 1 2 3))   ; → t
(count-if (lambda (x :fixnum) :bool (> x 1)) (list 1 2 3)) ; → 2
(last (list 1 2 3))               ; → 3
(range 5)                         ; → (0 1 2 3 4); n ≤ 0 → nil
(member? 2 (list 1 2 3))          ; → t
(find (lambda (x :fixnum) :bool (> x 2)) (list 1 2 3))  ; → 3
(position (lambda (x :fixnum) :bool (= x 2)) (list 1 2 3)) ; → 1
(take-while (lambda (x :fixnum) :bool (< x 3)) (list 1 2 3)) ; → (1 2)
(drop-while (lambda (x :fixnum) :bool (< x 3)) (list 1 2 3)) ; → (3)
(unique (list 1 2 1 3))           ; → (1 2 3) — порядок сохранён
(zip-with (lambda (a :fixnum b :fixnum) :fixnum (+ a b)) (list 1 2) (list 10 20)) ; → (11 22)
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

`map`/`filter`/`reduce` — **глобальные примитивы**, не экспорты `collections`; их колбэки обязаны быть типизированы: `(lambda (x :fixnum) :fixnum (* x x))`.

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
