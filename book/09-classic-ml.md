# Глава 9. Классический ML (модуль ML)

Модуль `ML` — небольшая типизированная библиотека метрик и подготовки данных. Весь его код лежит в `src/stdlib/ml.qlsp` и написан на самом Qualisp: это удобный материал для чтения — каждая функция видна целиком, без магии и скрытых слоёв.

```lisp
(import ML)
```

Важно понимать границы модуля: `ml` считает **метрики** по спискам и одному 1-D тензору и **готовит данные** (разбиение, масштабирование). Готовых моделей KNN, деревьев решений, случайных лесов, градиентного бустинга, K-Means, а также функций `cross-val-score` / `grid-search` в нём нет: обучение моделей — зона модуля `dl` (гл. 8), а кросс-валидацию и перебор гиперпараметров вы собираете из примитивов сами (§9.10). Экспорты перечислены в конце исходника: `mean`, `mse`, `mae`, `rmse`, `accuracy`, `train-part`, `test-part`, `minmax-scale`, `standardize`, `t-mse`, `precision`, `recall`, `f1`.

В 0.19.20 стандартная библиотека включает 13 модулей: рядом с `ml` работают `error` (загружается в prelude, структурированные ошибки с полем `kind`) и `datetime` (UTC-календарь, ISO-8601 и таймеры, подключается `(import DATETIME)`); их API описаны в гл. 15. Каждая экспортируемая функция каждого модуля — включая все 13 экспортов `ml` — покрыта отдельным тестом; всего в проекте 1095 тестов (`zig build test`).

## 9.1 Утилиты для списков

Базовые обходы списков — примитивы языка, они доступны без импорта и работают с явно типизированными функциями:

```lisp
(map (lambda ((x . :fixnum)) (* x x)) (list 1 2 3))          ;; → (1 4 9)
(filter (lambda ((x . :fixnum)) (> x 1)) (list 1 2 3))       ;; → (2 3)
(reduce (lambda ((a . :fixnum) (b . :fixnum)) (+ a b)) (list 1 2 3) 0) ;; → 6
(take 2 (list 1 2 3))                                        ;; → (1 2)
(drop 2 (list 1 2 3))                                        ;; → (3)
```

У `lambda` параметры аннотируются парой `(имя . :тип)`, а результат выводится — отдельной аннотации результата у `lambda` нет (в отличие от `defun`, где результат — последняя форма тела `(expr . :type)`).

`map`, `filter`, `reduce` — итеративные примитивы: глубина стека не зависит от длины списка, поэтому `(map f xs)` на длинном списке не упирается в кадры вызовов.

Модуль `COLLECTIONS` добавляет поверх примитивов типизированные утилиты; `range` и часть функций возвращают `option`, потому что пустого списка в языке нет:

```lisp
(import COLLECTIONS)

(range 5)          ;; → (0 1 2 3 4)
(range 0)          ;; → nil — пустой диапазон
(sum (list 1 2 3)) ;; → 6
(unique (list 1 1 2 3 2)) ;; → (1 2 3)
(zip-with (lambda ((a . :fixnum) (b . :fixnum)) (+ a b))
          (list 1 2) (list 10 20))          ;; → (11 22)
(find (lambda ((x . :fixnum)) (> x 2)) (list 1 2 3))        ;; → 3
(position (lambda ((x . :fixnum)) (> x 2)) (list 1 2 3))    ;; → 2
```

`range` строит список хвостовой рекурсией по `:fixnum` (`range-from` в `collections.qlsp`), а не обходом cons-цепочки. Полный список экспорта `collections`: `sum`, `keep-positive`, `any`, `all`, `count-if`, `last`, `range`, `find`, `position`, `take-while`, `drop-while`, `zip-with`, `member?`, `unique`.

> 🐍 `range(n)` ≈ `list(range(n))`; `zip-with` ≈ `map(f, xs, ys)`; `sum`/`unique` ≈ `sum(xs)` / `list(dict.fromkeys(xs))`. Разница в типах: пустой результат — `nil`, а не `[]`, поэтому у функций, которые могут вернуть пустоту, тип `(:option …)`.

## 9.2 Соглашения модуля ml

Три решения определяют весь API `ml` (они видны в комментариях в `ml.qlsp`):

| Соглашение | Следствие |
|---|---|
| Метрики определены на непустом `(:list :flonum)` | Пустого списка в языке нет, деления на ноль длины не бывает |
| `:fixnum` расширяется до `:flonum` при унификации элементов | `(mean (list 1 2 3))` типизируется без второго имени и без перегрузки |
| Рекурсии по cons нет (`cdr` даёт option) | Всё считается через `reduce`/`map`/`nth`; рекурсивен только генератор индексов по `:fixnum` |

Индексный хелпер — типичный пример:

```lisp
(defun indices ((i . :fixnum) (n . :fixnum))
  ((if (< (+ i 1) n)
       (cons i (indices (+ i 1) n))
       (list i)) . (:list :fixnum)))

(indices 0 4)   ;; → (0 1 2 3)
```

Внутри `ml` он называется `indices-from`; рядом работают `sum-flonums`, `squared-error-sum`, `check-ratio`, `ratio-count`, `check-labels`, `positive-pairs`, `false-positive-pairs`, `false-negative-pairs` — приватные хелперы, не экспортируемые наружу. Абсолютная величина берётся примитивом `abs`, а не отдельным `abs-flonum`.

> 🐍 В Python вы написали бы `sum(xs)`, `enumerate(xs)` и `zip(...)`; здесь те же идеи выражаются через `reduce` и явный список индексов, потому что статический тип `cdr` не гарантирует существования хвоста.

## 9.3 Метрики регрессии

```lisp
(mean (list 1.0 2.0 3.0))                         ;; → 2.0
(mse  (list 1.0 2.0 3.0) (list 1.5 2.5 3.5))      ;; → 0.25
(mae  (list 1.0 2.0 3.0) (list 1.5 2.5 3.5))      ;; → 0.5
(rmse (list 1.0 2.0 3.0) (list 1.5 2.5 3.5))      ;; → 0.5
```

| Функция | Сигнатура | Смысл |
|---|---|---|
| `mean` | `(:list :flonum) → :flonum` | среднее арифметическое |
| `mse` | `(:list :flonum) (:list :flonum) → :flonum` | средний квадрат ошибки |
| `mae` | `(:list :flonum) (:list :flonum) → :flonum` | средняя абсолютная ошибка |
| `rmse` | `(:list :flonum) (:list :flonum) → :flonum` | `sqrt` от `mse` |

Порядок аргументов у метрик — **(prediction, actual)**, то есть сначала предсказание, затем истинное значение. Разная длина входов — `throw` с сообщением вида `mse: prediction and target lengths differ`. Fixnum-списки принимаются напрямую: `(mean (list 1 2 3))` → `2.0`, `(accuracy (list 1 0 1) (list 1 1 1))` → `0.6666666666666666`.

> 🐍 `statistics.mean`, `sklearn.metrics.mean_squared_error`, `mean_absolute_error`, `root_mean_squared_error`. Заметьте, что в sklearn аргументы тоже `(y_true, y_pred)` — здесь порядок обратный, и это нужно помнить при переносе кода.

## 9.4 Тензорная метрика: t-mse

```lisp
(t-mse (tensor (1.0 2.0)) (tensor (1.5 2.5)))   ;; → 0.25
```

Реализация в `ml.qlsp`:

```lisp
(defun t-mse ((pred . (:tensor :f32 (_))) (actual . (:tensor :f32 (_))))
  ((tmean (* (- pred actual) (- pred actual))) . :flonum))
```

Вычитание и умножение здесь — elementwise broadcast, `tmean` сворачивает в `:flonum`. Ранг зафиксирован единицей: `_` — wildcard **измерения**, а не ранга, поэтому 2-D тензор в `t-mse` не типизируется. Несовпадение длин ловит тензорное ядро в рантайме.

> 🐍 `np.mean((pred - actual) ** 2)` для одномерного массива; для матриц используйте обычный `mse` над списками или свою функцию с `tmean`.

## 9.5 Метрики классификации: accuracy

```lisp
(accuracy (list 'a 'b 'a) (list 'a 'b 'b))   ;; → 0.6666666666666666
(accuracy (list 1 0 1) (list 1 1 1))         ;; → 0.6666666666666666
```

`accuracy` параметризована типом метки: оба аргумента имеют тип `(:list :a)`, возврат — `:flonum`. Сравнение идёт через generic `equal`, поэтому метки могут быть числами, символами, строками или bool — аргумент `pos-label`, привычный по многим библиотекам, здесь не нужен. Разная длина — `throw`.

> 🐍 `sklearn.metrics.accuracy_score(y_pred, y_true)`; поддержка произвольных меток «из коробки» совпадает.

## 9.6 Precision, recall и F1

```lisp
(print (precision (list 1 0 1 0) (list 1 0 0 1)))  ;; → 0.5
(print (recall    (list 1 0 1 0) (list 1 0 0 1)))  ;; → 0.5
(print (f1        (list 1 0 1 0) (list 1 0 0 1)))  ;; → 0.5
```

Все три функции принимают `(:list :fixnum)` с **бинарными метками 1/0** и возвращают `:flonum`. Внутри считаются приватные счётчики `positive-pairs` (tp), `false-positive-pairs` (fp) и `false-negative-pairs` (fn); нулевой знаменатель даёт `0.0`, а не NaN.

| Функция | Формула |
|---|---|
| `precision` | `tp / (tp + fp)` |
| `recall` | `tp / (tp + fn)` |
| `f1` | `2·p·r / (p + r)` |

Длины проверяются хелпером `check-labels`; несовпадение — `throw`.

> 🐍 `precision_score(y_pred, y_true, pos_label=1)`, `recall_score`, `f1_score`. Здесь `pos_label` зафиксирован единицей и не передаётся: для небинарной задачи кодируйте нужный класс в `1` явно.

## 9.7 Разбиение на train / test

```lisp
(train-part (list 1 2 3 4 5) 0.6)   ;; → (1 2 3)
(test-part  (list 1 2 3 4 5) 0.6)   ;; → (4 5)
(train-part (list 1 2 3) 0.0)       ;; → nil
(test-part  (list 1 2 3) 1.0)       ;; → nil
```

Сигнатуры: `train-part : (:list :a) :flonum → (:option (:list :a))`, `test-part : (:list :a) :flonum → (:option (:list :a))`.

`ratio` — доля **обучающей** части, обязательно из `0..1`; вне диапазона — `throw`. Число элементов `ratio·n` округляется к ближайшему целому (0.5 вверх) с допуском `1e-9`, чтобы двоичный шум вроде `0.3·10 = 2.9999999999999996` не терял элемент. Пустая часть честно возвращается как `nil` в типе `(:option (:list :a))`, а не как ошибка.

> 🐍 `train_test_split(X, y, train_size=0.6)`. Отличие API: `train-part`/`test-part` принимают **один** список и число, а не `X, y`; target нужно разбивать отдельным вызовом.

## 9.8 Масштабирование в [0, 1]

```lisp
(minmax-scale (list 1.0 2.0 3.0))       ;; → (0.0 0.5 1.0)
(minmax-scale (list 5.0 5.0 5.0))       ;; → (0.0 0.0 0.0)
```

Функция `(minmax-scale xs (:list :flonum)) (:list :flonum)` ищет минимум и максимум через `reduce` и применяет `(x − lo) / (hi − lo)` через `map`. Вырожденный диапазон (все значения равны) отображается в нули: деление с нулевым знаменателем не определено, а `nil` выглядел бы как успех. Это то же соглашение, что у `MinMaxScaler`.

Вход — **плоский список чисел**, не список строк: для таблицы применяйте функцию к нужной колонке.

> 🐍 `MinMaxScaler().fit_transform(X)`; соглашение для константной колонки совпадает (нули).

## 9.9 Стандартизация

```lisp
(standardize (list 1.0 2.0 3.0))   ;; → (-1.224744871391589 0.0 1.224744871391589)
(standardize (list 5.0 5.0))       ;; → (0.0 0.0)
```

`(standardize xs (:list :flonum)) (:list :flonum)` вычитает среднее и делит на **генеральное** стандартное отклонение: делим на `n`, а не на `n − 1`, потому что метрики считаются по всему набору, а не оценивают выборку. Нулевое СКО снова даёт нули.

> 🐍 `StandardScaler().fit_transform(X)` — у sklearn по умолчанию тоже генеральное СКО (`ddof=0`).

## 9.10 Кросс-валидация и подбор гиперпараметров

Отдельных `cross-val-score` и `grid-search` в `ml` нет. Это не «пропущенная функция»: модуль намеренно содержит только то, что нужно для оценки и подготовки данных, — модели и оптимизаторы живут в `dl` (гл. 8). Кросс-валидацию и перебор вариантов вы пишете на обычных типизированных функциях, используя `indices`-хелпер из §9.2, `reduce`, `nth` и метрики `ml`.

Минимальный k-fold-счёт и перебор двух вариантов предобработки:

```lisp
(import ML)

(defun indices ((i . :fixnum) (n . :fixnum))
  ((if (< (+ i 1) n) (cons i (indices (+ i 1) n)) (list i)) . (:list :fixnum)))

(defun fold-mse ((xs . (:list :flonum)) (ys . (:list :flonum)) (k . :fixnum) (fold . :fixnum))
  ((reduce (lambda ((i . :fixnum) (acc . :flonum))
             (if (= (mod i k) fold) acc
               (let ((d (- (nth xs i) (nth ys i)))) (+ acc (* d d)))))
           (indices 0 (length xs)) 0.0) . :flonum))

(defun cv-mse ((xs . (:list :flonum)) (ys . (:list :flonum)) (k . :fixnum))
  ((/ (reduce (lambda ((f . :fixnum) (acc . :flonum)) (+ acc (fold-mse xs ys k f)))
              (indices 0 k) 0.0)
      (* k 1.0)) . :flonum))

(defun grid ((xs . (:list :flonum)) (ys . (:list :flonum)) (k . :fixnum))
  ((list (cv-mse xs ys k)
         (cv-mse (minmax-scale xs) ys k)) . (:list :flonum)))

(print (indices 0 5))   ;; → (0 1 2 3 4)
(print (cv-mse (list 1.0 2.0 3.0 4.0) (list 1.5 2.0 3.0 4.5) 2)) ;; → 0.25
(print (grid   (list 1.0 2.0 3.0 4.0) (list 1.5 2.0 3.0 4.5) 2))
;; → (0.25 11.361111111111112)
```

`fold-mse` пропускает текущий фолд по остатку от деления индекса на `k`, `cv-mse` усредняет ошибки по фолдам, а `grid` сравнивает варианты. Второй результат больше первого: масштабирование предсказаний меняет их масштаб относительно цели — допустимый, но не лучший вариант, и перебор как раз это показывает.

> 🐍 `cross_val_score(model, X, y, cv=k).mean()` и `GridSearchCV(...)` делают то же самое, но в `ml` таких обёрток нет — вы строите их из примитивов под свою модель.

## 9.11 End-to-end пример

Типизированный пример: считаем метрики, масштабируем и разбиваем выборку.

```lisp
(import ML)
(import COLLECTIONS)

(defun evaluate ((pred . (:list :flonum)) (y . (:list :flonum)))
  ((list (mse pred y) (mae pred y) (rmse pred y)) . (:list :flonum)))

(defun scale-split ((xs . (:list :flonum)) (ratio . :flonum))
  ((train-part (minmax-scale xs) ratio) . (:option (:list :flonum))))

(print (evaluate (list 1.0 2.0 3.0) (list 1.5 2.0 2.5)))
;; → (0.16666666666666666 0.3333333333333333 0.408248290463863)

(print (scale-split (list 1.0 2.0 3.0 4.0) 0.5))
;; → (0.0 0.3333333333333333)

(print (accuracy (list 'a 'b 'a) (list 'a 'b 'b)))
;; → 0.6666666666666666
```

Все определения верхнего уровня аннотированы: типы параметров и возврата обязательны, `defvar` тоже требует типа. Результат `scale-split` — option-список, потому что при `ratio = 0` обучающая часть пуста.

> 🐍 Эквивалент по смыслу:
> ```python
> from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
> from sklearn.preprocessing import MinMaxScaler
> from sklearn.model_selection import train_test_split
>
> X_scaled = MinMaxScaler().fit_transform(X)
> X_train, _ = train_test_split(X_scaled, train_size=0.5)
> (mean_squared_error(pred, y), mean_absolute_error(pred, y))
> ```

## 9.12 Отличия от scikit-learn

| Qualisp | Python | Комментарий |
|---|---|---|
| `(mse pred actual)` | `mean_squared_error(y_true, y_pred)` | порядок аргументов обратный |
| `(accuracy pred actual)` | `accuracy_score(y_true, y_pred)` | метки любого атомарного типа, `pos-label` не нужен |
| `(precision pred actual)` | `precision_score(y_true, y_pred)` | только бинарные метки 1/0 |
| `(train-part xs ratio)` | `train_test_split(X, train_size=ratio)` | один список; target разбивается отдельно |
| `(minmax-scale xs)` | `MinMaxScaler().fit_transform(X)` | плоский список чисел, константа → нули |
| `(standardize xs)` | `StandardScaler().fit_transform(X)` | генеральное СКО |
| `(t-mse p a)` | `np.mean((p - a) ** 2)` | только 1-D `:f32`-тензоры |
| — | `KNeighborsClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `KMeans` | моделей в `ml` нет; обучение — в `dl` (гл. 8) |
| — | `cross_val_score`, `GridSearchCV` | готовых обёрток нет; см. §9.10 |

API функциональный, данные — типизированные списки Qualisp (для нейросетей — тензоры), реализации короткие и целиком видны в `src/stdlib/ml.qlsp`. Это позволяет прочитать каждый алгоритм от начала до конца, но означает, что «всё из sklearn» в модуле не ожидается.
