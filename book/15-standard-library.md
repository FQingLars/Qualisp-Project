# Глава 15. Стандартная библиотека

Все модули стандартной библиотеки **вшиты в бинарник** (`stdlib/*.qlsp` генерируются в заголовки через `xxd -i` и компилируются в бинарник на этапе сборки — `cmake/stdlib_embed.cmake`). **16 модулей** загружаются **при старте** REPL/раннера (итерация по `embedded_modules[]` в `src/main.cpp`), причём порядок фиксирован: `collections` — первым (`map`/`filter`/`reduce` нужны модулям, которые грузятся после), `tablet` — последним. `(import M)` — идиома для явности (повторный импорт — дешёвый no-op). Поддержан частичный импорт: `(import-from (ml knn-predict nb-fit))`. Семантика импорта и `MODULE`-заголовок с описью зависимостей — в 15.14.

| Модуль | Файл | Размер | Содержание |
|---|---|---|---|
| `core` | `stdlib/core.qlsp` | ~124 строки | утилиты списков (итеративные: `length`/`reverse`/`member`/`last`/`butlast`/`nth` не жгут стек), `let*`, тестовый фреймворк |
| `string` | `stdlib/string.qlsp` | ~60 строк | строковые операции, окна (`STRING-REF`/`STRING-VIEW`), char-коды, примитивы + композитные |
| `ml` | `stdlib/ml.qlsp` | ~414 строк | классический ML: KNN, NB, Tree, RF, GB, KMeans, CV, GridSearch, потери |
| `dl` | `stdlib/dl.qlsp` | ~159 строк | слои-как-данные, `linear`, `net-forward`, `train-step`, `train-epochs`, `attention`, `softmax!`, чекпоинт Adam |
| `ns` | `stdlib/ns.qlsp` | ~126 строк | экспорт нейросимвольного слоя: маршрутизация по слотам, RL-агент-кодер (гл. 14.10) |
| `io` | `stdlib/io.qlsp` | ~80 строк | файлы, CSV (RFC-4180), `slurp`, `spit` |
| `regex` | `stdlib/regex.qlsp` | ~17 строк | регулярки: `re-find`, `re-replace`, `re-split` |
| `audio` | `stdlib/audio.qlsp` | ~102 строки | DSP: окна, синус, DFT, mel-шкала, dB |
| `datetime` | `stdlib/datetime.qlsp` | ~90 строк | время: `now-ts`, `sleep`, `elapsed` |
| `pkg` | `stdlib/pkg.qlsp` | ~74 строки | `pkg-install` через `http-get` |
| `errors` | `stdlib/errors.qlsp` | ~94 строки | ошибки-структуры от трейта `ERROR` (бектрейс, цепочка `cause`), `throw`, `assert`, `try`, тип-предикаты |
| `visual` | `stdlib/visual.qlsp` | ~33 строки | TensorBoard-подобный HTTP-дашборд |
| `tablet` | `stdlib/tablet.qlsp` | ~1313 строк | DataFrame-аналог pandas — гл. 22 |
| `async` | `stdlib/async.qlsp` | ~100 строк | асинхронный слой: фоновые задачи, каналы, batch-выполнение — гл. 15.12 |
| `collections` | `stdlib/collections.qlsp` | ~235 строк | `map`/`filter`/`reduce`, изменяемые `dict`/`set`/`queue`/`heap` — гл. 15.16 |
| `argparser` | `stdlib/argparser.qlsp` | ~222 строки | разбор аргументов командной строки, `--help` — гл. 15.17 |

> 🐍 «16 вшитых модулей в одном бинарнике» — это «один fat-пакет в одном бинарнике». Python-аналог: монолитный Docker-образ с предустановленными `numpy/torch/sklearn/pandas/tensorboard` — только без GIL и без 4 ГБ зависимостей.

## 15.1 core — всегда загружен

Утилиты списков (`length`, `reverse`, `member`, `nth`, `second`…), `let*`, тестовый фреймворк:

```lisp
(deftest "два-плюс-два"
  (assert-eq (+ 2 2) 4 "математика сломалась"))
(run-tests)
```

> 🐍 `pytest` в миниатюре: `deftest` ≈ `def test_*`, `run-tests` ≈ запуск pytest, `assert-eq` ≈ `assert a == b`. Имена тестов склеиваются через `strcat` с префиксом `"test::"`. Упавший `assert` проваливает тест, а сводка прогона печатает **счётчики** — числа пройденных и всех тестов вида `N/N passed`, а не имена типов.

Оговорка о числах: release собирается с `-ffast-math`, поэтому сравнение плавающих в release ненадёжно (для NaN оно истинно). Единственный надёжный способ проверить конечность — примитив `(FINITE? t)` (битовая проверка), а не сравнение; числовые пороги в тестах ставятся выше baseline вырожденного решения.

Для тензоров есть `assert-zero` — проверка, что тензор почти нулевой (стандартная проверка в тестах градиентов):

```lisp
(defun assert-zero (tensor msg)
  (let ((s (tensor-scalar (TSUM (T* tensor tensor)))))
    (if (> (ABS s) 0.1)
      (progn (print "    FAIL:") (print msg) nil)
      t)))
```

Также определены car/cdr-сокращения: `caar`, `cadr`, `cdar`, `cddr`, `caaar`, `caadr`, `cadar`, `caddr`, `cdaar`, `cdadr`, `cddar`, `cdddr`, плюс `second`, `third`, `fourth`, `first`, `rest`.

## 15.2 string — строковые операции

Модуль зарегистрирован в `embedded_modules[]` сразу после `core`. Два слоя — примитивы и композитные функции.

### Примитивы (C++, регистрируются через `register_string_primitives`)

| Примитив | Что делает | Python-аналог |
|---|---|---|
| `STRING-LENGTH s` | длина строки | `len(s)` |
| `STRING-SPLIT разделитель строка` | список подстрок; **разделитель — первый аргумент** | `s.split(delim)` |
| `STRING-JOIN list delim` | объединение | `delim.join(list)` |
| `STRING-UPCASE s` | uppercase | `s.upper()` |
| `STRING-DOWNCASE s` | lowercase | `s.lower()` |
| `STRING-TRIM s` | обрезка пробелов по краям | `s.strip()` |
| `STRING-REPLACE s old new` | замена всех вхождений | `s.replace(old, new)` |
| `STRING-INDEX s sub` | позиция подстроки или `nil` | `s.find(sub)` |
| `STRING-REVERSE s` | разворот | `s[::-1]` |
| `CHAR-CODE ch` | ASCII-код символа | `ord(ch)` |
| `CODE-CHAR n` | символ по ASCII-коду | `chr(n)` |
| `STRING-TO-NUMBER s` | парсинг числа | `float(s)` / `int(s)` |
| `STRING-CONTAINS s sub` | предикат вхождения | `sub in s` |
| `STRING-REF s i` | байт `0..255` по индексу, без аллокации | `s[i]` у `bytes` |
| `STRING-VIEW s from len` | окно `[from, from+len)` без копирования | `s[from:from+len]` |
| `STRING< a b` | лексикографическое `<` | `a < b` (для строк) |

### Композитные (на примитивах, в `stdlib/string.qlsp`)

```lisp
(import STRING)

(string-contains s sub)         ; STRING-INDEX, потом nil-check (есть и примитив STRING-CONTAINS)
(string-starts-with s prefix)
(string-ends-with s suffix)
(string-slice s start end)      ; подстрока
(string-lines s)                ; split по '\n'
(string-words s)                ; split по whitespace
(string-repeat s n)             ; n копий через STRING-JOIN
(to-string x)                   ; STRING(NUMBER-TO-STRING ...) для скаляров
```

> 🐍 Аналог модуля `str` Python (Python 3.6+). В Python многое встроено в сам язык (`len`, `+`, `[]`), здесь — отдельные примитивы из-за Lisp-стиля (один prim = одна функция).

### Строковые окна: `STRING-REF` и `STRING-VIEW`

Строка — не только значения, но и **окна** в неё: `(STRING-VIEW s from len)` даёт окно без копирования, `(STRING-REF s i)` — байт по индексу (0..255), тоже без аллокации.

```lisp
(STRING-VIEW "hello" 1 3)              ;; → "ell" — окно
(type-of (STRING-VIEW "hello" 1 3))    ;; → STRING
(equal (STRING-VIEW "hello" 1 3) "ell") ;; → T — сравнение по содержимому
(STRING-REF "hello" 1)                 ;; → 101 — байт 'e'
(STRING-LENGTH (STRING-VIEW "hello" 1 3)) ;; → 3
(STRCAT (STRING-VIEW "hello" 1 3) "!")    ;; → "ell!"
```

Прозрачность — главное свойство окна: **для всех строковых операций окно — обычная строка**. `TYPE-OF` → `STRING`, `equal` сравнивает по содержимому, `STRING-LENGTH`, `STRCAT`, `SUBSTR`, `STRING-SPLIT`, `STRING-CONTAINS` работают одинаково. `STRING-SPLIT` над stable-строкой отдаёт куски-**окна**, а `string-slice`/`string-starts-with`/`string-ends-with` построены на окнах.

Стоимость, а не семантика, — вот что меняется: над скретч-строкой и над окном (`view-of-view`) операции дают копию, а escape в stable (`defvar`) материализует копию. Границы — ошибка с именем операции:

```lisp
(STRING-VIEW "abc" 1 5)    ;; ERROR: string-view: range [1,6) out of [0,3)
(STRING-REF "abc" 9)       ;; ERROR: string-ref: index 9 out of range (3)
```

> 🐍 В Python `s[1:3]` — тоже срез, но он всегда создаёт новую строку; в Qualisp окно живёт в исходной строке, пока его не материализуют копией.

### Замечание о `type-of`

`type-of` возвращает uppercase-имена типов:

| Значение | `type-of` | Python `type(x).__name__` |
|---|---|---|
| `42` | `INT` | `int` |
| `3.14` | `FLOAT` | `float` |
| `"hi"` | `STRING` | `str` |
| `'sym` | `SYMBOL` | (нет аналога) |
| `(1 2 3)` | `LIST` | `list` |
| `(tensor ...)` | `TENSOR` | `numpy.ndarray` / `torch.Tensor` |
| `(lambda ...)` / `(defun ...)` | `FUNCTION` | `function` |
| `#(...)` вектор | `VECTOR` | (нет аналога) |
| `nil` | `NIL` | `NoneType` |

> 🐍 В Python имена типов в lowercase, в Qualisp — uppercase (после `intern`).

## 15.3 IO — файлы и строки

```lisp
(import IO)

(slurp "data.txt")                 ;; прочитать весь файл      open().read()
(spit "out.txt" "содержимое")      ;; записать                 open().write()
(file? "out.txt")                  ;; существует?              os.path.exists
(read-lines "data.txt")            ;; список строк             open().readlines()
(read-csv "table.csv")             ;; список списков полей     csv.reader
(write-csv "out.csv" rows)         ;; строки → CSV-файл        csv.writer
```

CSV читается и пишется по RFC-4180: поля с запятыми/кавычками/переводами строк берутся в кавычки, `""` — экранирование кавычки внутри поля, CRLF читается, переводы строк **внутри** кавычек корректно разворачиваются. Пустые строки пропускаются (семантика `skip_blank_lines=True` у pandas). Круговой рейс точен:

```lisp
(write-csv "/tmp/t.csv" (list (list "a" "b") (list "1" "2,5") (list "3" "4")))
(read-csv "/tmp/t.csv")            ;; → (("a" "b") ("1" "2,5") ("3" "4"))
```

Реализация — на строковых примитивах (`STRING-SPLIT`/`STRING-JOIN`/`SUBSTR`), без посимвольных проходов; нижележащие примитивы файлов — `READ-FILE`/`WRITE-FILE`/`FILE-EXISTS?`.

> ⚠️ **Парсер итеративен.** `read-csv` не рекурсирует по одному символу: это конечный автомат по RFC-4180 на `while` + `setq`-накопление (идиома train-step — scratch итерации освобождается на чекпоинте, накопленные записи живут в биндингах `let`). Так `read-csv` держит продакшен-размерные CSV на дефолтной глубине `eval`: файл в 400 строк читается без поднятия `EVAL-MAX-DEPTH` (а поднятый лимит при стеке 8 МБ падает в SIGSEGV вместо ошибки — ровно то, от чего guard и защищает). Семантика прежняя: кавычки держат запятые и переводы строк, `""` — экранирование, CRLF и одиночный CR, пропуск пустых строк, хвостовая неполная запись. Тест: `io-read-csv-scales-400-rows`.

## 15.4 REGEX — регулярные выражения

```lisp
(import REGEX)

(re-find "[0-9]+" "abc 123")        ;; первое совпадение        re.search(...).group()
(re-matches? "^[a-z]+$" "abc")      ;; bool                     bool(re.search(...))
(re-replace "[0-9]+" "N" "a1b22")   ;; замена                   re.sub
(re-split "," "a,b,c")              ;; разбиение                re.split (через \\x1F-маркер)
```

Нижележащие примитивы: `REGEX-MATCH`, `REGEX-REPLACE` (POSIX regex).

## 15.5 DATETIME — время

```lisp
(import DATETIME)

(now-ts)                 ;; unix-время (сек)          time.time()
(now-iso-str)            ;; ISO-строка                datetime.now().isoformat()
(sleep 2)                ;; пауза, сек                time.sleep
(wait-ms 500)            ;; пауза, мс                 time.sleep(0.5)
(timer-start)            ;; метка времени             t0 = time.time()
(timer-elapsed t0)       ;; сколько прошло            time.time() - t0
(seconds->hms 3665)      ;; → (1 1 5)

;; макрос замера времени
(elapsed (expensive-computation))    ;; напечатает elapsed: N s
```

> 🐍 `(elapsed expr)` ≈ contextmanager `timeit`/декоратор с замером времени. NB: `sleep` принимает секунды, `wait-ms` — миллисекунды.

## 15.6 AUDIO — DSP

```lisp
(import AUDIO)

(hann-window 1024)           ;; окно Ханна           scipy.signal.hann
(hamming-window 1024)        ;; окно Хэмминга
(sine-wave 440 2.0 44100)    ;; синус 440 Гц, 2 сек  np.sin(2πf t)
(fft-magnitudes signal)      ;; модули спектра       np.abs(np.fft.fft)
(stft signal 1024 256)       ;; оконное преобразование  librosa.stft
(hz-to-mel 440)              ;; мел-шкала            librosa.hz_to_mel
(mel-to-hz mel)
(lin-to-db amplitude)        ;; децибелы             20*log10
```

Примитив `FFT` (ядро) — быстрое преобразование Фурье Коули–Тьюки над тензором (см. `prim_fft` в `interpreter.cpp`).

`fft-magnitudes` возвращает список Python-стиля (не тензор) — для удобства работы с DS-пайплайнами.

## 15.7 ERRORS — обработка ошибок

```lisp
(import ERRORS)

(throw "что-то не так")            ;; бросить ошибку (структуру от трейта ERROR)
(assert condition "сообщение")     ;; упасть, если ложно
(assert-type v 'float)             ;; упасть, если тип не тот
(try (risky-op)
  (e (print (error-message e))))   ;; try/catch

(error-message e) (error-cause e) (error-traceback e)  ;; чтение полей ошибки
(print-backtrace e)                                     ;; печать кадров бектрейса
(error? x)                                              ;; структура ли это

(type-check v 'float)              ;; проверка типа (int/float/string/list/tensor/nil)
(is-fixnum? 5) (is-flonum? 2.5)    ;; точные предикаты типов    isinstance
(is-number? 2.5) (is-string? "x")
(is-cons? '(1)) (is-tensor? (tensor ((1.0))))
(is-nil? nil)
```

Предикаты — честные (по тегам значений, не по `type-of`-строкам); `type-check` проверяет по имени типа (`int`/`float`/`string`/`list`/`tensor`/`nil`), `assert-type` бросает ошибку при несовпадении. Всё, что бросается через `throw`/`assert` и возникает в рантайме, — структуры ниже.

### Ошибка — структура от трейта ERROR

Любая ошибка — **структура** от трейта `ERROR` с тремя полями:

| Поле | Что несёт |
|---|---|
| `message` | текст ошибки |
| `cause` | рекурсивная ссылка на ошибку, пойманную под ней; у корневой ошибки — `nil` |
| `traceback` | список кадров бектрейса (см. 15.13) |

Поля читают методы трейта — обычные функции `error-message`, `error-cause`, `error-traceback` (`type-of` → `FUNCTION`, `functionp` → `t`):

```lisp
(error-message (catch-error (lambda () (throw "boom"))))  ;; → "boom"
(error-message (catch-error (lambda () (/ 1 0))))         ;; → "...division by zero..."
(error-cause (catch-error (lambda () (throw "root"))))     ;; → nil — корневая ошибка
(error-traceback e)                                       ;; → список кадров
```

Арифметическая ошибка — та же структура, никакого особого канала нет: `(/ 1 0)` бросает ошибку с `message` «division by zero». JIT и интерпретатор дают одну и ту же ошибку (паритет); ошибка в непройденной ветке не мешает компиляции формы — constant folding сворачивает её ещё при компиляции.

> 🐍 Как объект исключения Python, только все три части — декларативный контракт одного трейта: `message` ≈ `str(e)`, `cause` ≈ `e.__cause__`/`e.__context__` (цепочка «поверх словленной»), `traceback` ≈ `e.__traceback__`.

### catch-error: контракт

`CATCH-ERROR` — настоящий примитив: прогоняет тело и возвращает **одно** значение одной из двух форм:

- **успех** — пара `(T . результат)`: `car` → `T`, `cdr` → результат тела;
- **ошибка** — структура ошибки; у структуры `car` = `nil`, поэтому проверка «была ли ошибка» через `(car e)` работает.

```lisp
(car (catch-error (lambda () 42)))                  ;; → T — успех
(cdr (catch-error (lambda () (+ 1 2))))             ;; → 3 — результат тела
(car (catch-error (lambda () (throw "x"))))         ;; → nil — была ошибка
(type-of (catch-error (lambda () (throw "boom"))))  ;; → STRUCT
```

`try` из модуля `errors` построен поверх `catch-error` и потому сохраняет результат тела (а не только обрабатывает ошибку).

> 🐍 `try`-макрос ≈ `try/except e:`. Внутри `try` оборачивает body в `(lambda () body)` и вызывает через `catch-error`; ошибка связывается с символом из `catch-clause`.

### Цепочка cause

Ошибка, брошенная поверх уже пойманной, неявно тащит старую в `cause` — цепочка собирается сама, без аннотаций в коде:

```lisp
;; err-cause-chain: внутри внешней ошибки уже была поймана внутренняя
(let ((inner nil))
  (let ((outer (catch-error (lambda ()
                 (setq inner (catch-error (lambda () (throw "first"))))
                 (throw "second")))))
    (error-message outer)                   ;; → "second" — внешняя ошибка
    (error-message (error-cause outer))))   ;; → "first" — ошибка, пойманная под ней
```

Цепочка рекурсивна: `(error-cause (error-cause e))` шагает вглубь, пока не дойдёт до корневой ошибки, у которой `cause` = `nil`. Отчёт по ошибке ходит по цепочке `cause` от корня до листа, печатая `message` каждой ошибки.

## 15.8 VISUAL — дашборд обучения

```lisp
(import VISUAL)

(dashboard 8765)            ;; HTTP-сервер (TensorBoard-подобный) на порту 8765
(log-metric "loss" 0.25)    ;; скалярная метрика (авто-инкремент *step-counter*)
(log-tensor w "weights")    ;; heatmap тензора
(plot xs ys "loss-curve")   ;; линейный график
(scatter xs ys "points")    ;; scatter-плот
(imshow matrix "attention") ;; heatmap
(log-loss 0.1 5)            ;; shortcut: log-metric + явный step
```

Внутри — `start-dashboard`, `log-metric`, `log-tensor` (примитивы в `src/visual/dashboard.cpp`).

## 15.9 PKG — установка пакетов из GitHub

```lisp
(import PKG)
(pkg-install "user/repo")            ;; скачать pkg.qlsp из GitHub raw и установить
(pkg-install-branch "user/repo" "dev") ;; другая ветка
```

Внутри — `http-get` (примитив `prim_http_get`), `pkg-install-branch` идёт через `/tmp/qualisp-pkg-<name>/`, скачивает `pkg.qlsp` манифест и указанные в нём файлы через raw.githubusercontent.com. Подробнее — в главе 16.

## 15.10 IMPORT-FROM

Частичный импорт: поднимает только перечисленные имена из модуля, не весь модуль.

```lisp
(import-from (ml knn-predict nb-fit))
;; после этого доступны только KNN-PREDICT и NB-FIT
;; остальное (KNN-FIT, TREE-FIT, RF-FIT, ...) остаётся недоступным
```

> 🐍 Аналог `from sklearn.neighbors import KNeighborsClassifier` в Python — но без скрытого `import sklearn` (модуль целиком не загружается как имя).

Реализация (`Interpreter::do_import_from`, `interpreter.cpp:1493`):

1. Загрузить модуль (вычислить его top-level формы).
2. Из всех `defvar`/`defun` модуля собрать список экспортов.
3. Из переданного списка имён проверить, что каждое определено в модуле (иначе — `runtime_error`).
4. Скопировать эти имена в текущий environment (через `set_global`).

Полезно в больших моделях — ускоряет загрузку и уменьшает namespace pollution. `import-from` видит имена зависимостей модуля после их загрузки (заголовок `MODULE` и рекурсивные зависимости — 15.14).

## 15.11 BOUNDP

```lisp
(boundp 'x)                ; T, если X связана в текущем/глобальном env
```

Полезно в макросах и meta-circular интерпретаторах: «определена ли переменная?» без чтения значения (т.е. без `eval`).

## 15.12 ASYNC — асинхронный слой

Механика слоя живёт в C++ (пул воркеров, интерпретатор на каждом воркере, рефкаунт результата future, очередь канала) — это не выражаемо на Qualisp. Что принадлежит языку — **форма интерфейса**: имена и правила композиции.

```lisp
(async-run form)        ;; одна работа в фоне → ручка (макрос: задаче нужна форма, а не значение)
(async-wait handle)     ;; дождаться и взять значение
(async-collect handles) ;; дождаться список ручек → значения в порядке входа
(async-map fn lst)      ;; батч: задача на элемент, результат в порядке входа

(async-chan)            ;; канал (очередь)
(async-send ch v)       ;; положить значение
(async-recv ch)         ;; взять значение (блокирует воркер-задачу)
(async-close ch)        ;; закрыть канал

(async-pipeline ...)    ;; конвейер: производитель → потребитель через каналы
(async-threads)         ;; размер пула (см. ниже)
```

**Контракт мира задачи:**

- задача исполняется на воркере со **своим** интерпретатором: свои scratch-арена, loop stores, реестры, лента; ничего, что делает задача, не дотягивается до мутабельного состояния вызывающего;
- мир задачи — **снимок**: глобальное окружение общее и только для чтения (запись глобала из задачи — нарушение контракта, интерпретатор её отвергает), а локальные переменные вызывающего **копируются** в задачу при создании; захваченный задачей тензор копируется вместе с ней;
- форма задачи — собственная копия future, поэтому дождаться future можно в любой форме, а не только в создавшей;
- результат принадлежит future до конца программы (значение «заимствованное», как и любой результат вычисления).

**Где живут ядра.** Арифметика Qualisp параллельна по умолчанию — тензорные кёрнелы внутри одной операции занимают ядра (OpenMP в поэлементных и редукционных циклах, OpenBLAS в матричных произведениях). Слой — вторая половина: структурировать работу, которая не является одной большой тензорной операцией (фоновая работа, поток между шагами, батч мелких кусков).

> ⚠️ **Пул по умолчанию последовательный: один воркер.** Задачи идут одна за другой, поток программы свободен. `QUALISP_THREADS` поднимает число воркеров — **на риск оператора**: при трёх и более одновременных воркерах задача, циклящаяся по переменной `let`, может вернуть испорченное значение (кросс-поточная гонка в общем состоянии интерпретатора, не локализована). Ограничение — свойство слоя, а не программы: задача, которая только считает и возвращает значение, его не чувствует. `(async-threads)` считает потоки вместе с потоком программы: без переменной окружения это 2, с `QUALISP_THREADS=4` — 5.

> 🐍 Ближайший Python-аналог — `concurrent.futures` (пул, future'ы) плюс `queue.Queue` для каналов; только мир задачи здесь — снимок в стиле fork, а не общий heap.

## 15.13 Бектрейс ошибки

Поле `traceback` структуры ошибки (15.7) — список **кадров-структур**. Кадр сам структура с фиксированными полями:

| Поле | Тип | Что несёт |
|---|---|---|
| `name` | символ | имя функции кадра |
| `file` | строка | файл-источник |
| `line` | число | номер строки |
| `form` | форма-список | форма, вычислявшаяся в кадре |

Порядок кадров — **oldest → newest**: внешний вызов первым, последний кадр — точка ошибки. Читать кадры — обычным `slot-value`:

```lisp
(defun bt-frames (e) (error-traceback e))

(let* ((e (catch-error (lambda () (bt-top 3))))
       (f (car (bt-frames e))))
  (list (type-of f)            ;; → STRUCT — кадр тоже структура
        (slot-value f 'name)   ;; → BT-TOP — символ
        (slot-value f 'file)   ;; → строка
        (slot-value f 'line)   ;; → число
        (slot-value f 'form))) ;; → (bt-rec n) — форма-список
```

Кадр точки вызова несёт **исходную форму вызова**, а не раскрытую: макро-вызов показан формой **до развёртки** (голова — сам макрос):

```lisp
(defmacro bt-mac (x) (list 'bt-rec x))
(defun bt-mac-user () (bt-mac 1))
;; в бектрейсе найдётся кадр BT-MAC с формой (bt-mac 1) —
;; ровно так, как вызов написан в исходнике
(equal (slot-value f 'form) '(bt-mac 1))   ;; → t
```

Кадр из JIT-кода несёт **мастер-форму вызываемого**: нативная страница работает без unwind, поэтому форма берётся из master-копии функции. Вызов через JIT сохраняет стек — все кадры на месте.

**Хвостовая рекурсия схлопывает кадры.** TCO переиспользует кадр, и хвостовой цикл даёт **один** кадр независимо от глубины — как в Scheme; это контракт, а не потеря информации. Не-хвостовая рекурсия растит кадры по-настоящему:

```lisp
(defun bt-err () (throw "boom"))
(defun bt-rec (n) (if (< n 0) (bt-err) (bt-rec (- n 1))))              ;; хвостовая
(defun bt-rec-nt (n) (if (< n 0) (bt-err) (+ 1 (bt-rec-nt (- n 1)))))  ;; не-хвостовая
(defun bt-top (n) (bt-rec n))
(defun bt-top-nt (n) (bt-rec-nt n))

;; (catch-error (lambda () (bt-top 200))): BT-REC в бектрейсе ровно один раз
;; (catch-error (lambda () (bt-top-nt 6))): кадров больше пяти, BT-REC-NT и BT-ERR оба в стеке
```

`PRINT-BACKTRACE e` печатает бектрейс ошибки (возвращает `nil`). Ошибка внутри построения бектрейса не ломает отчёт. Глубокая рекурсия (сотни кадров) собирается без сбоев.

> 🐍 `traceback.extract_tb(e.__traceback__)` в Python отдаёт `FrameSummary(filename, lineno, name, line)`; Qualisp-кадр несёт не текст строки, а саму форму (`form`), а порядок oldest → newest совпадает с питоновским. Схлопывание хвостовых кадров — семантика Scheme (TCO), в Python такого нет.

## 15.14 Модули: MODULE-заголовок и рекурсивные зависимости

Первая форма модуля — **`MODULE`-заголовок**: `(MODULE name (dep...))` — явная опись зависимостей.

```lisp
(MODULE md-dep-a (md-dep-b ...))   ;; имя модуля + опись зависимостей
```

Импорт грузит зависимости **рекурсивно в порядке описи**, и все они выполняются **раньше тела модуля** — порядок детерминирован и не зависит от того, кто первый импортировал:

```lisp
(import tests.mod.md-dep-a)
;; тела зависимостей выполнились раньше тела модуля, порядок описи сохранён
```

`import-from` видит имена зависимостей после загрузки — не только имена самого модуля:

```lisp
(import-from tests.mod.md-dep-b (md-b-val md-c-val))
(md-b-val)   ;; → 2 — имя из самого модуля
(md-c-val)   ;; → 3 — имя из зависимости видно после загрузки
```

Правила загрузки:

- **цикл зависимостей** — внятная ошибка с цепочкой, называющая участников цикла (не тихий no-op с частично инициализированным модулем);
- **отсутствующая зависимость** — ошибка, называющая виновника;
- `MODULE` не первой формой модуля или второй заголовок — ошибка;
- **повторный импорт** загруженного модуля — no-op.

> 🐍 Как импорт Python, но с явной описью: в Python тело модуля тоже выполняется после его импортов, а цикл даёт `ImportError: ... (most likely due to a circular import)`; здесь порядок зависимостей задан заголовком, а не порядком строк в теле.

## 15.15 Сводка модулей

| Модуль | Назначение | Python-аналог |
|---|---|---|
| `core` | утилиты списков, `let*`, тесты | builtins + pytest |
| `string` | строки, char-коды | `str` (Python 3.6+) |
| `ml` | классический ML: KNN, NB, деревья, ансамбли, KMeans, CV, grid search, метрики, потери | scikit-learn |
| `dl` | слои-как-данные, `linear`/`net-forward`/`train-step`, `attention`, `softmax!`, gather-эмбеддинги, чекпоинт Adam, клип градиента | `torch.nn` (стиль Sequential) |
| `ns` | экспорт нейросимвольного слоя, RL-агент-кодер | (нет аналога) |
| `io` | файлы, CSV (RFC-4180) | `open`, `csv` |
| `regex` | регулярки | `re` |
| `audio` | DSP, окна, FFT, STFT, mel | `scipy.signal`, `librosa` |
| `datetime` | время | `time`, `datetime` |
| `pkg` | пакеты из GitHub | `pip` (мини) |
| `errors` | ошибки, try, assert | exceptions |
| `visual` | дашборд метрик | TensorBoard |
| `tablet` | DataFrame-аналог pandas, детерминированный | `pandas` |
| `async` | фоновые задачи, каналы, batch-выполнение | `concurrent.futures` + `queue.Queue` |
| `collections` | `map`/`filter`/`reduce`, `dict`/`set`/`queue`/`heap` | `functools`, `collections`/`heapq` |
| `argparser` | разбор аргументов командной строки, автосправка | `argparse` |

**Tablet.** DataFrame-модуль на чистом Qualisp поверх примитивов реализован целиком: типизированные колонки, стабильные сортировки, groupby/merge с детерминированным порядком, `tcsv-read`/`tcsv-write`, JSON rows, окна. Подробности — гл. 22.

## 15.16 COLLECTIONS — map/filter/reduce и структуры данных

Модуль `collections` (`stdlib/collections.qlsp`) собирает функции высшего порядка над списками и изменяемые структуры данных для алгоритмизации (A*, Дейкстра, DSU, топологическая сортировка). `map`/`filter`/`reduce` выражены композицией `cons`/`car`/`cdr` на самом Qualisp: stdlib-функция к тому же компилируется JIT — в отличие от примитива, который всегда вызов наружу.

Обходы **итеративные**: `map`/`filter` идут `while`-циклом, `reduce` — хвостовой рекурсией, которую схлопывает TCO, поэтому глубина стека не зависит от длины списка. Список в сто тысяч элементов сворачивается без поднятия `EVAL-MAX-DEPTH`.

### map, filter, reduce: контракт

```lisp
(import COLLECTIONS)

(map f lst...)          ;; применить f к элементам списков   map(...) по многим спискам
(filter pred lst)       ;; оставить по предикату             filter(pred, lst)
(reduce fn seq . init)  ;; свёртка                           functools.reduce
```

- **`map` многолистовый и останавливается на КОРОЧАЙШЕМ списке.** `f` получает по одному элементу каждого списка; обход — слева направо, поэтому побочные эффекты колбэка идут в порядке списка. Пустой список (и `nil` вместо списка) даёт `nil`; вызов без списков вовсе — тоже `nil`.
- **`reduce` зовёт `(fn ЭЛЕМЕНТ АККУМУЛЯТОР)` — элемент первым.** Это не порядок Common Lisp, но именно так работают существующие программы. Без инициализатора первый элемент становится аккумулятором **без вызова `fn`** для него. `nil`-инициализатор неотличим от отсутствующего — засеять свёртку пустым списком через `init` нельзя. Пустой список без начального значения → `nil`, с начальным — само начальное.
- **`filter`** оставляет элементы, где предикат истинен (истина — любое непустое значение).

```lisp
(map (lambda (a b) (list a b)) '(1 2) '(10 20 30))  ;; → ((1 10) (2 20)) — решает короткий
(map (lambda (x) (* x 2)) '(1 2 3))                 ;; → (2 4 6)
(filter (lambda (x) (> x 2)) '(1 2 3 4))            ;; → (3 4)

(reduce (lambda (e acc) (+ e acc)) '(1 2 3) 10)     ;; → 16 — (fn элемент аккумулятор)
(reduce (lambda (e acc) (+ e acc)) '(1 2 3))        ;; → 6 — первый элемент сам аккумулятор
(reduce (lambda (e acc) (list e acc)) '(1 2 3))     ;; → (3 (2 1)) — порядок применения виден
(reduce (lambda (e acc) (+ e acc)) '() 5)           ;; → 5 — пустой список с начальным
(reduce (lambda (e acc) (+ e acc)) '())             ;; → nil — пустой без начального
```

Складываются без обёрток — например, `(reduce (lambda (e acc) (+ e acc)) (map (lambda (x) (* x 2)) '(1 2 3)) 0)` → `12`.

> 🐍 `map` ≈ `map`, но многолистовый и останавливается на коротком (как `zip`); `filter` ≈ `filter`; `reduce` ≈ `functools.reduce`, только колбэк зовёт `(элемент, аккумулятор)`, а не `(аккумулятор, элемент)`.

### dict, set, queue, heap

Изменяемые коллекции нужны там, где алгоритм держит состояние в одном объекте. Изменяемость — через поля struct (`set-slot`): примитивов мутации cons в языке нет, а коллекция обязана жить в одном объекте. Равенство ключей и элементов — через `equal`; коллекции — обычные S-выражения, поэтому `save-image` сериализует их без дополнительной работы. Имя `map` занято функцией высшего порядка, поэтому ассоциативный массив называется `dict`; суффикс `!` не используется — он зарезервирован за tape-операциями.

| Коллекция | Функции | Назначение |
|---|---|---|
| `dict` | `dict`, `dict-find`, `dict-put`, `dict-get`, `dict-has?`, `dict-del`/`dict-remove`, `dict-keys`, `dict-vals`, `dict-size` | ассоциативный массив |
| `set` | `set`, `set-has?`/`set-member`, `set-add`, `set-add-all`, `set-del`/`set-remove`, `set-size`, `set->list` | множество без дублей |
| `queue` | `queue`, `queue-push`, `queue-flip`, `queue-pop`, `queue-peek`, `queue-empty?`, `queue-size` | очередь front/rear, O(1) амортизированно |
| `heap` | `heap`/`heap-with`, `heap-merge`, `heap-push`, `heap-pop`, `heap-peek`, `heap-empty?`, `heap-size` | skew-куча, O(log n) амортизированно |

- **dict.** `dict-put` переписывает значение существующего ключа, иначе добавляет запись, и возвращает сам `dict`; `dict-get` — с необязательным дефолтом для отсутствующего ключа; `dict-find` находит запись; `dict-keys`/`dict-vals` — списки ключей и значений; `dict-size` — число записей; удаление — `dict-del`/`dict-remove`.
- **set.** Добавление — `set-add`, слияние с другим множеством — `set-add-all`; проверка — `set-has?`/`set-member`; удаление — `set-del`/`set-remove`; `set-size` — размер, `set->list` — выгрузка в список. Дубли не добавляются.
- **queue.** `queue-push` кладёт в хвост, `queue-pop` берёт из головы (из пустой — `nil`), `queue-peek` смотрит голову без удаления, `queue-empty?`/`queue-size` — состояние. `queue-flip` перекладывает хвост в голову; `queue-pop`/`queue-peek` делают это сами — O(1) амортизированно.
- **heap.** `heap` — куча наименьшего по `<`; `heap-with` задаёт свой предикат порядка `less`; `heap-push`/`heap-pop`/`heap-peek` (из пустой — `nil`), `heap-merge` сливает две кучи, `heap-empty?`/`heap-size` — состояние. Узел — `(значение левое правое)`, при слиянии дети меняются местами (перекос skew): дерево не балансируется, но держит логарифм амортизированно.

```lisp
(defvar d (dict))
(dict-put d "epochs" 10)
(dict-get d "epochs")              ;; → 10
(dict-get d "missing" 'fallback)   ;; → fallback — дефолт для отсутствующего
(dict-has? d "epochs")             ;; → t
(dict-keys d)                      ;; → ("epochs")

(defvar q (queue))
(queue-push q "a")
(queue-push q "b")
(queue-pop q)                      ;; → "a" — FIFO

(defvar h (heap))                  ;; min-куча по <
(heap-push h 3)
(heap-push h 1)
(heap-peek h)                      ;; → 1
(heap-pop h)                       ;; → 1
(heap-size h)                      ;; → 1
```

> 🐍 `dict` ≈ `dict`, `set` ≈ `set`, `queue` ≈ `collections.deque`, `heap` ≈ `heapq` — только всё живёт в одном lisp-объекте (мутация через `set-slot`), без отдельных модулей и без синтаксиса `[]`.

## 15.17 ARGPARSER — аргументы командной строки

Модуль `argparser` (`stdlib/argparser.qlsp`) даёт функционал Python argparse в идиоматичном лисп-обличье: позиционные и опциональные аргументы, типы, дефолты, обязательность, автогенерируемая справка `--help`. Результат разбора — алис значений, доступ через `arg-get`.

Примитив `(command-line)` отдаёт список аргументов командной строки процесса (список строк); в `parse-args` передают его хвост:

```lisp
(import ARGPARSER)

(let ((p (argparser :description "обучение модели")))
  (add-argument p "dataset")                          ;; позиционный
  (add-argument p "--epochs" :type :int :default 10)  ;; :str :int :float :bool
  (add-argument p "--quiet" :type :bool :default nil)
  (add-argument p "--model" :required t)
  (let ((parsed (parse-args p (cdr (command-line)))))
    (arg-get parsed "epochs")            ;; значение
    (arg-get parsed "missing" 'fallback) ;; fallback для отсутствующего
    ))
```

Контракт:

- **позиционные и опциональные различаются по префиксу `-`/`--`**; имена хранятся без ведущих дефисов: `--epochs` → `"epochs"`. Позиционные обязательны по умолчанию (как в argparse), опции — нет; `:required` переопределяет оба правила;
- форма **`--имя=значение`** работает наравне с парой `--имя значение`; флаг `:bool` без значения = `t`;
- типы конвертируются из строк: `:int` принимает `-3`, `:float` — `2.5`; не-число в `:int` — ошибка с именем опции. Значения `:default` уже нужного типа и не конвертируются;
- **обязательность и неизвестные опции — внятные ошибки**, а не тихий `nil`: пропущенный обязательный аргумент даёт ошибку с «required», неизвестная опция и ошибка типа называют саму опцию; ошибки ловятся обычным каналом (`catch-error`/`error-message`, гл. 15.7);
- **`--help` (и `-h`) генерируется автоматически**: печатает usage, `:description` и `:help`-строки аргументов и останавливает разбор;
- результат `parse-args` — **алис значений**; `arg-get` достаёт значение по имени, третий аргумент — fallback для отсутствующего имени (без него — `nil`).

`add-argument` принимает ключи `:type` (`:str`/`:int`/`:float`/`:bool`), `:default`, `:required`, `:help` (строка для автосправки):

```lisp
(add-argument p "--n" :type :int)
(parse-args p (list "--n" "abc"))    ;; ошибка называет опцию --n
(parse-args p nil)                   ;; --model :required t → ошибка «required»
(parse-args p (list "--nope" "1"))   ;; неизвестная опция — ошибка с именем --nope

(arg-get parsed "epochs")            ;; → значение (уже сконвертированное в :int)
(arg-get parsed "missing")           ;; → nil — имени нет и fallback не задан
(arg-get parsed "missing" 'fallback) ;; → fallback
```

> 🐍 `argparse.ArgumentParser` + `add_argument` + `parse_args`: те же роли и привычные правила (позиционные обязательны, `--help` из коробки, `--имя=значение`), только конфигурация — лисп-вызовы, результат — алис, а не namespace, а `(command-line)` ≈ `sys.argv`.