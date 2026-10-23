# Глава 15. Стандартная библиотека

Все модули стандартной библиотеки **вшиты в бинарник** (`stdlib/*.qlsp` генерируются в заголовки через `xxd -i` и компилируются в бинарник на этапе сборки — `cmake/stdlib_embed.cmake`). **14 модулей** загружаются **при старте** REPL/раннера (итерация по `embedded_modules[]` в `src/main.cpp`), причём порядок важен: `core` грузится первым (через `Interpreter::mark_module_loaded(m->name)`). `(import M)` — идиома для явности (повторный импорт — дешёвый no-op). Поддержан частичный импорт: `(import-from (ml knn-predict nb-fit))`.

| Модуль | Файл | Размер | Содержание |
|---|---|---|---|
| `core` | `stdlib/core.qlsp` | ~124 строки | утилиты списков (итеративные: `length`/`reverse`/`member`/`last`/`butlast`/`nth` не жгут стек), `let*`, тестовый фреймворк |
| `string` | `stdlib/string.qlsp` | ~60 строк | строковые операции, char-коды, примитивы + композитные |
| `ml` | `stdlib/ml.qlsp` | ~378 строк | классический ML: KNN, NB, Tree, RF, GB, KMeans, CV, GridSearch, потери |
| `dl` | `stdlib/dl.qlsp` | ~159 строк | слои-как-данные, `linear`, `net-forward`, `train-step`, `train-epochs` |
| `ns` | `stdlib/ns.qlsp` | ~126 строк | экспорт нейросимвольного слоя: маршрутизация по слотам, RL-агент-кодер (гл. 14.10) |
| `io` | `stdlib/io.qlsp` | ~80 строк | файлы, CSV (RFC-4180), `slurp`, `spit` |
| `regex` | `stdlib/regex.qlsp` | ~17 строк | регулярки: `re-find`, `re-replace`, `re-split` |
| `audio` | `stdlib/audio.qlsp` | ~102 строки | DSP: окна, синус, DFT, mel-шкала, dB |
| `datetime` | `stdlib/datetime.qlsp` | ~90 строк | время: `now-ts`, `sleep`, `elapsed` |
| `pkg` | `stdlib/pkg.qlsp` | ~74 строки | `pkg-install` через `http-get` |
| `errors` | `stdlib/errors.qlsp` | ~66 строк | `make-error`, `throw`, `assert`, `try`, тип-предикаты |
| `visual` | `stdlib/visual.qlsp` | ~33 строки | TensorBoard-подобный HTTP-дашборд |
| `tablet` | `stdlib/tablet.qlsp` | ~1313 строк | DataFrame-аналог pandas — гл. 22 |
| `async` | `stdlib/async.qlsp` | ~100 строк | асинхронный слой: фоновые задачи, каналы, batch-выполнение — гл. 15.12 |

> 🐍 «13 вшитых модулей в одном бинарнике» — это «один fat-пакет в одном бинарнике». Python-аналог: монолитный Docker-образ с предустановленными `numpy/torch/sklearn/pandas/tensorboard` — только без GIL и без 4 ГБ зависимостей.

## 15.1 core — всегда загружен

Утилиты списков (`length`, `reverse`, `member`, `nth`, `second`…), `let*`, тестовый фреймворк:

```lisp
(deftest "два-плюс-два"
  (assert-eq (+ 2 2) 4 "математика сломалась"))
(run-tests)
```

> 🐍 `pytest` в миниатюре: `deftest` ≈ `def test_*`, `run-tests` ≈ запуск pytest, `assert-eq` ≈ `assert a == ==`. Имена тестов склеиваются через `strcat` с префиксом `"test::"`. Сводка прогона печатает счётчик вида `INT/INT passed` — это тоже `strcat` над `type-of`-символами: не-строковые аргументы `strcat` печатаются как `WRITE-TO-STRING`, а не молча пропускаются.

Оговорка о числах: release собирается с `-ffast-math`, поэтому сравнение плавающих в release ненадёжно (для NaN оно истинно). Единственный надёжный способ проверить конечность — примитив `(FINITE? t)` (битовая проверка), а не сравнение; числовые пороги в тестах ставятся выше baseline вырожденного решения.

Для тензоров есть `assert-zero` — проверка, что тензор почти нулевой (стандартная проверка в тестах градиентов):

```lisp
(defun assert-zero (tensor msg)
  (let ((s (tensor-scalar (TSUM (T* tensor tensor)))))
    (if (> (ABS s) 0.1)
      (progn (print "    FAIL:") (print msg) nil)
      t)))
```

Также определены 16 сокращений `car`/`cdr`: `caar`, `cadr`, `cdar`, `cddr`, `caaar`, `caadr`, `cadar`, `caddr`, `cdaar`, `cdadr`, `cddar`, `cdddr`, плюс `second`, `third`, `fourth`, `first`, `rest`.

## 15.2 string — строковые операции

Модуль зарегистрирован в `embedded_modules[]` сразу после `core`. Два слоя — примитивы и композитные функции.

### Примитивы (C++, регистрируются через `register_string_primitives`)

| Примитив | Что делает | Python-аналог |
|---|---|---|
| `STRING-LENGTH s` | длина строки | `len(s)` |
| `STRING-SPLIT s delim` | список подстрок | `s.split(delim)` |
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
(string s)                      ; константная строка-обёртка
```

> 🐍 Аналог модуля `str` Python (Python 3.6+). В Python многое встроено в сам язык (`len`, `+`, `[]`), здесь — отдельные примитивы из-за Lisp-стиля (один prim = одна функция).

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

(throw "что-то не так")            ;; бросить ошибку       raise
(assert condition "сообщение")     ;; упасть, если ложно   assert
(make-error 'type "детали")        ;; ошибочное значение-данные: (list 'error type msg)
(error-type e) (error-msg e)       ;; разбор
(type-check v 'float)              ;; проверка типа (int/float/string/list/tensor/nil)
(type-error? e) (value-error? e) (runtime-error? e)
(try (risky-op)
  (e (print (error-msg e))))       ;; try/catch

(is-fixnum? 5) (is-flonum? 2.5)    ;; точные предикаты типов    isinstance
(is-number? 2.5) (is-string? "x")
(is-cons? '(1)) (is-tensor? (tensor ((1.0))))
(is-nil? nil)
(assert-type v 'fixnum "v должен быть целым")
```

Предикаты — честные (по тегам значений, не по `type-of`-строкам); `type-check` проверяет по имени типа (`int`/`float`/`string`/`list`/`tensor`/`nil`).

`CATCH-ERROR` — настоящий примитив: возвращает `((T result) | (NIL "message"))`, поэтому `try` сохраняет результат тела (а не только обрабатывает ошибку).

> 🐍 `try`-макрос ≈ `try/except e:`. Внутри `try` оборачивает body в `(lambda () body)` и вызывает через `catch-error`; ошибка связывается с символом из `catch-clause`.

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

Полезно в больших моделях — ускоряет загрузку и уменьшает namespace pollution.

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

## 15.13 Сводка модулей

| Модуль | Назначение | Python-аналог |
|---|---|---|
| `core` | утилиты списков, `let*`, тесты | builtins + pytest |
| `string` | строки, char-коды | `str` (Python 3.6+) |
| `ml` | классический ML + потери/softmax (377 строк на самом Qualisp) | scikit-learn |
| `dl` | слои-как-данные, `linear`/`net-forward`/`train-step` | `torch.nn` (стиль Sequential) |
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

**Tablet.** DataFrame-модуль на чистом Qualisp поверх примитивов реализован целиком: типизированные колонки, стабильные сортировки, groupby/merge с детерминированным порядком, `tcsv-read`/`tcsv-write`, JSON rows, окна. Подробности — гл. 22.