# Глава 7. Автоградиент

В Qualisp reverse-mode автоградиент построен поверх **HLO-графа**: `defgraph` один раз трассирует тело в граф, `grad!` исполняет forward и проходит по узлам в обратном порядке. Это модель `torch.compile` + `loss.backward()`: отдельной ленты градиентов как значения языка нет. Рядом работает символьный reverse-mode `(grad term var)` (§7.8): он не считает градиент численно, а **строит терм** градиента скалярного лосса для последующей сверки с autograd.

## 7.1 Конвенция: граф, а не лента

- `(defgraph name ((p . :type) …) body… (expr . :ret))` трассирует тело **один раз** на нулевых placeholder-тензорах и записывает узлы; результат аннотируется на последней форме тела. Вызов графа компилируется в `(%GRAPH-RUN gid ret arg…)`.
- `param` помечает тензор `requires_grad`; `grad!` берёт последний запуск графа и идёт по узлам с конца, читая forward-значения из `run.values` — **без replay**.
- Суффикса `!` как признака tape-операции нет: `relu`, `exp`, `matmul`, `+ - * /` — обычные функции, работающие и в eager-коде, и в теле графа.
- Операторы `+ - * /` на тензорах — поэлементные с broadcast (гл. 6); матричное произведение — `matmul`.
- Backward-правила определены для `add`, `sub`, `mul`, n-арного `div` (числитель `1/(b·c…)`, каждый знаменатель `−output/den`; broadcast сворачивается `reduceToShape`), `neg`, batched `matmul`, `reshape`, `transpose` с перестановкой, `relu`, `exp`, `log`, `tanh`, `sin`, `cos`, `sqrt`, `atan`, `abs`, `softmax`, `log-softmax`, `sum`, `mean`, `max`, `t-slice`, `t-concat`, `conv2d`, max/avg pooling, `gather`/`scatter-add`. `sum` размножает градиент на форму входа, `mean` делит на число элементов, `max` отдаёт градиент всем позициям с максимумом. `softmax`/`log-softmax` считаются по последней оси; осевые редукции дифференцируются через `reduceAxisBackward`; `argmax`, `one-hot`, `t-gt`/`t-lt` — дискретные данные без градиента.

> 🐍 **Python-аналогия.** В PyTorch узел появляется во время forward, если вход `requires_grad`; здесь граф задаётся явно через `defgraph`, а `param` лишь помечает, для каких входов заполнять `.grad`. Видно, что считается, в самом определении графа.

## 7.2 Обучаемые параметры: `param`

```lisp
(defvar w1 ((param (tensor ((0.5 -0.4 0.3) (-0.2 0.6 -0.5)))) . (:tensor :f32 (2 3))))
(defvar b1 ((param (tensor (0.1 -0.1 0.0))) . (:tensor :f32 (3))))
```

`(param t)` помечает `:f16`/`:f32`/`:f64`-тензор `requires_grad`; `:u8` и `:i64` отвергаются, потому что целочисленные данные не дифференцируются. Флаг переживает копирование в world (`cloneToWorld`), так что параметр, объявленный через `defvar`, остаётся обучаемым.

**Важно.** Всё, что трассировщик видит в теле графа как глобальную константу (`tensor`-литерал, `zeros`/`ones`/`randn` внутри тела), становится стабильной константой графа и градиента **не получает**. Поэтому веса объявляются `param` и передаются в граф аргументами — как в `examples/xor.qlsp`.

> 🐍 Аналог `torch.nn.Parameter(...)`. Но `param` — обычная функция, а не обёртка модуля: отдельной формы для параметров нет, обучаемый тензор объявляется через `defvar` с аннотацией парного вида `(значение . :тип)`.

## 7.3 Forward + Backward

Проверенный пример: для `p = 2`, `x = 3` градиент `tmean((p·x − x)²)` по `p` равен `2·3·3 = 18`.

```lisp
(defvar p ((param (tensor (2.0))) . (:tensor :f32 (1))))
(defgraph sq ((p . (:tensor :f32 (1))) (x . (:tensor :f32 (1))))
  ((tmean (* (- (* p x) x) (- (* p x) x))) . :flonum))

(grad! (sq p (tensor (3.0))))
(tref (grad-of p) 0)    ;; → 18.0
```

- `(grad! loss)` берёт последний запуск (`Runtime.last_run`: граф, входы, выход), сеет единицы формы выхода и идёт по узлам с конца.
- Forward выполняется с `frame.retain = 1`: все активации остаются в per-run `TensorBufferPool`, поэтому backward их перечитывает — без повторного forward.
- Сложность backward — **O(n)** по числу узлов; память — O(n) активаций до `grad!`/конца формы.
- Градиенты накапливаются в scratch; для `param`-входов копируются в stable и кладутся в `Tensor.grad`.
- `(grad-of t)` возвращает stable-градиент или нули той же формы/dtype.
- Повторный `grad!`, новый вызов графа или конец формы освобождают per-run пул; `last_run` сбрасывается на границе top-level формы.

> 🐍 `grad!` ≈ `loss.backward()`, `grad-of` ≈ `w.grad`. Вызов графа и `grad!` должны быть в одной top-level форме: `last_run` живёт ровно до её конца. Сам градиентный тензор после этого лежит в stable.

## 7.4 Оптимизаторы

Для eager-пути в `dl` есть шаги, меняющие существующий тензор на месте: `(sgd-update! w grad lr)` и `(adam-update! state grad lr)`. Они сохраняют идентичность `param` и не перепривязывают слот структуры к scratch-тензору. В графовом цикле можно обновлять веса явно через `tset!` по `(grad-of w)`, как в `examples/xor.qlsp`:

```lisp
(defvar lr (0.5 . :flonum))
(defvar epoch (0 . :fixnum))
(while (< epoch 5000)
  (grad! (xor-loss x w1 b1 w2 b2 tgt))
  (tset! w1 0 0 (- (tref w1 0 0) (* lr (tref (grad-of w1) 0 0))))
  ;; … остальные элементы весов и смещений
  (setq epoch (+ epoch 1)))
```

Почему не `setq`: результат чистого `sgd-step`/`adam-step` из модуля `dl` имеет wildcard-форму `(_ _)`, а wildcard нельзя присвоить конкретной переменной `(2 1)` (гл. 8). Чистые шаги возвращают новое состояние и удобны в eager-пути с переменными `(:tensor :f32 (_ _))`; `*-update!` меняют уже существующий stable-тензор. В графе проще и дешевле обновить stable-параметр на месте.

> 🐍 `(tset! w i j (- (tref w i j) (* lr (tref (grad-of w) i j))))` — это развёрнутый `optimizer.step()` для одного элемента параметра. `adam-update!` копирует моменты `m`/`v` в **существующие** stable-буферы через `t-copy!`; иначе стабильная структура `adam-state` держала бы ссылку на scratch, которая умирает на границе top-level формы.

## 7.5 Поддерживаемые autograd-операции

| Категория | Операции |
|---|---|
| Арифметика | n-арные `+ - * /` (broadcast), унарный `-` → `neg` |
| Матричные | batched `matmul` (ранг ≥2) |
| Форма | `reshape`, `transpose` с перестановкой, `t-slice`, `t-concat` |
| Активации | `relu`, `exp`, `log`, `tanh`, `sin`, `cos`, `sqrt`, `atan`, `abs`, `softmax`, `log-softmax` |
| Редукции | `tsum`, `tmean`, `tmax` (полные и по одной оси; только float) |
| Индексация | `t-gather`, `t-scatter-add` (градиент по значениям, не по индексам) |
| Свёртки/пулинг | `t-conv2d` (stride/pad/dilation/groups), `t-max-pool2d`, `t-avg-pool2d` |
| Дискретное | `argmax`, `t-one-hot`, `t-gt`, `t-lt` — данные без градиента |
| Сеть | `defgraph` (трассировка + вызов), `defuse` (fused-ядро, инлайнится перед трассировкой) |
| Градиент | `param`, `grad!`, `grad-of` |

`sin`, `cos`, `sqrt`, `atan` имеют и forward-, и backward-ядра: в графе они работают наравне с `exp`/`log`/`tanh`. Пример для `x = 0.5`: градиент `tsum(sin x)` — `cos(0.5) ≈ 0.87758255`, `tsum(cos x)` — `−sin(0.5) ≈ −0.47942555`, `tsum(sqrt x)` — `1/(2√0.5) ≈ 0.70710677`, `tsum(atan x)` — `1/(1+0.5²) = 0.8`.

Градиент конкретного параметра — `(grad-of w)`; для параметра, не участвовавшего в последнем запуске, сохраняется прежний градиент, а для никогда не считанного — нули той же формы.

В графе **нет** `tref`/`tset!` и `pow`; `argmax` теперь есть как `argmax_axis` (последняя ось), `one-hot`/`gt`/`lt` — дискретные узлы без градиента. Редукции в графе работают только с float-тензорами, поэтому `defgraph` с объявленным `:fixnum`-возвратом фактически недостижим. Нет вложенных `defgraph`, рекурсии и data-dependent control flow.

## 7.6 Сравнение с PyTorch

| PyTorch | Qualisp |
|---|---|
| `torch.randn(4, 2)` | `(randn :f32 (4 2))` |
| `w = torch.nn.Parameter(...)` | `(defvar w ((param (randn :f32 (4 2))) . (:tensor :f32 (4 2))))` |
| `model = nn.Linear(4, 1); y = model(x)` | `(defgraph predict ((x . (:tensor :f32 (4 1))) (w . (:tensor :f32 (4 1)))) ((matmul x w) . (:tensor :f32 (4 1))))` |
| `F.mse_loss(pred, y)` | `(let ((d (- pred y))) (tmean (* d d)))` |
| `loss.backward()` | `(grad! (loss w b xs ys))` |
| `w.grad` | `(grad-of w)` |
| `torch.compile(model)` | `defgraph` (ленивая компиляция графа в x86-64) |
| `F.relu(x)` | `(relu x)` |
| `opt.step()` | `tset!` по `(grad-of w)` |
| `opt.zero_grad()` | отдельной формы нет: градиент перезаписывается следующим `grad!` |
| `x.detach()` | вызов eager-функции без `grad!` (граф просто не строится) |

Граф линеаризуется в узлы с арностью ≤ 2: n-арные `+ - * /` разворачиваются в цепочку бинарных. Backward лениво компилируется в обратную последовательность вызовов `ql_bwd_*` (x86-64, W^X); интерпретатор остаётся fallback.

## 7.7 Полный пример: линейная регрессия

Проверенный прогон сходится к `w ≈ 2`, `b ≈ 1` (истина `y = 2x + 1`):

```lisp
(defvar xs ((tensor ((1.0) (2.0) (3.0) (4.0))) . (:tensor :f32 (4 1))))
(defvar ys ((tensor ((3.0) (5.0) (7.0) (9.0))) . (:tensor :f32 (4 1))))
(defvar w ((param (tensor ((0.0)))) . (:tensor :f32 (1 1))))
(defvar b ((param (tensor (0.0))) . (:tensor :f32 (1))))

(defgraph loss ((w . (:tensor :f32 (1 1))) (b . (:tensor :f32 (1)))
                (xs . (:tensor :f32 (4 1))) (ys . (:tensor :f32 (4 1))))
  ((let ((d (- (+ (matmul xs w) b) ys)))
     (tmean (* d d))) . :flonum))

(defvar lr (0.1 . :flonum))
(defvar epoch (0 . :fixnum))
(while (< epoch 200)
  (grad! (loss w b xs ys))
  (tset! w 0 0 (- (tref w 0 0) (* lr (tref (grad-of w) 0 0))))
  (tset! b 0 (- (tref b 0) (* lr (tref (grad-of b) 0))))
  (setq epoch (+ epoch 1)))

(print (tref w 0 0))   ;; → 2.000223159790039
(print (tref b 0))     ;; → 0.9993436336517334
```

`let` в Qualisp — параллельные биндинги, поэтому зависимые биндинги в теле графа вкладываются (`let` → `let`), как в `examples/xor.qlsp`. Полный разбор XOR и MLP — в гл. 8 и 17.

## 7.8 Символьная сверка: `diff` и `term`

Один и тот же символьный терм может служить и обучаемым графом, и источником **аналитического** градиента для сверки с autograd. Это делает связка модулей `neuro` (`term`) и `symbolic` (`diff`, `grad`); подробности — гл. 14.

Проверенный прогон из `examples/term.qlsp` (`c` учится на `y = c·x²`, истинное `c = 2`):

```lisp
(import neuro)
(import symbolic)

(defvar xs ((tensor (0.5 1.0 1.5 2.0)) . (:tensor :f32 (4))))
(defvar ys ((tensor (0.5 2.0 4.5 8.0)) . (:tensor :f32 (4))))
(defvar c ((param (tensor (0.5))) . (:tensor :f32 (1))))
(defvar lr (0.05 . :flonum))
(defvar epoch (0 . :fixnum))

(defgraph loss ((c . (:tensor :f32 (1))) (x . (:tensor :f32 (4))) (y . (:tensor :f32 (4))))
  ((term '(tmean (* (- (* c (* x x)) y) (- (* c (* x x)) y))) '(c x y)) . :flonum))

(while (< epoch 200)
  (grad! (loss c xs ys))
  (tset! c 0 (- (tref c 0) (* lr (tref (grad-of c) 0))))
  (setq epoch (+ epoch 1)))
(print (tref c 0))                       ;; → 2.0
(print (diff '(* c (* x x)) 'c))         ;; → (* X X) — аналитический градиент

(defgraph sum-sq ((c . (:tensor :f32 (1))) (x . (:tensor :f32 (4))))
  ((term '(tsum (* c (* x x))) '(c x)) . :flonum))
(defgraph slope-sq ((x . (:tensor :f32 (4))))
  ((term '(tsum (* x x)) '(x)) . :flonum))
(grad! (sum-sq c xs))
(print (tref (grad-of c) 0))             ;; → 7.5
(print (slope-sq xs))                    ;; → 7.5 — тот же результат термом
```

`diff` строит терм аналитического градиента, `term` компилирует его в тот же набор HLO-узлов; совпадение `7.5` и `7.5` — сверка аналитической производной и обратного прохода графа.

> 🐍 Это `sympy.diff(f, c)` против `autograd.grad`: символьная формула и численный backward дают одно и то же значение.

### Reverse-mode `(grad term var)`

`(grad term var)` — градиент **скалярного лосса `(tsum term)`**, а не поэлементная производная (в отличие от `diff`). Реализация — `(simplify (grad-step term var 1.0))`: стартовый градиент (seed) равен единице, дальше правила применяются рекурсивно по дереву терма.

| Узел | Правило |
|---|---|
| `+` | сумма градиентов аргументов |
| `-` | унарный минус / разность градиентов |
| `*` | `∂a = g*b`, `∂b = g*a`; n-арное сводится к `b = (* …)` |
| `/` | `g/b` и `-g*a/(b*b)` |
| `^` | `g*n*a^(n-1)`, num exponent |
| `tsum` | `g * ones-like(a)` |
| `tmean` | `(g * ones-like(a)) / tsum(ones-like(a))` |
| `tmax` | `g * tmax-mask(a)` |
| `matmul` (оба аргумента) | `matmul(g, transpose(b))` и `matmul(transpose(a), g)` |
| `relu` | `g * relu-mask(a)` |
| `softmax` (1-D) | `s * (g − tsum(g*s))`, где `s = softmax(a)` |
| `transpose` | `transpose(g)` |
| `reshape` | `reshape-like(g, a)` |
| `ones-like`/`tmax-mask`/`relu-mask` | `0` |
| унарные `sin…atan` | `g * unary-derivative-form(op, a)` |

Формы сохраняются отдельными тензорными операциями (`ones-like`, `tmax-mask`, `relu-mask`, `reshape-like`), поэтому `simplify` не склеивает их в арифметику.

```lisp
(import symbolic)
(grad '(tsum x) 'x)                 ;; → (ONES-LIKE X)
(grad '(tmax x) 'x)                 ;; → (TMAX-MASK X)
(grad '(tsum (relu x)) 'x)          ;; → (* (ONES-LIKE (RELU X)) (RELU-MASK X))
(grad '(tsum (reshape x (2 3))) 'x) ;; → (RESHAPE-LIKE (ONES-LIKE (RESHAPE X (2 3))) X)
(grad '(tsum (matmul x y)) 'x)      ;; → (MATMUL (ONES-LIKE (MATMUL X Y)) (TRANSPOSE Y))
(grad '(tsum (matmul x y)) 'y)      ;; → (MATMUL (TRANSPOSE X) (ONES-LIKE (MATMUL X Y)))
(grad '(tsum (transpose x)) 'x)     ;; → (TRANSPOSE (ONES-LIKE (TRANSPOSE X)))
(grad '(tmean x) 'x)                ;; → (/ (ONES-LIKE X) (TSUM (ONES-LIKE X)))
(grad '(tsum (softmax x)) 'x)
;; → (* (SOFTMAX X) (- (ONES-LIKE (SOFTMAX X)) (TSUM (* (ONES-LIKE (SOFTMAX X)) (SOFTMAX X)))))
```

Градиент по обоим аргументам `matmul` и аналитическая формула `tsum(w·softmax(x))` сверены с autograd; `grad` для `matmul` работает там, где `diff` отказывает.

**ANALYTICAL-сверка.** Символьный терм градиента можно скомпилировать через `term` и сравнить с численным `grad-of`. `term` требует литерального `quote`, поэтому готовую форму `grad` подставляют явно:

```lisp
(import symbolic)
(import neuro)
(defvar xs ((tensor (0.5 1.0 1.5 2.0)) . (:tensor :f32 (4))))
(defvar c ((param (tensor (0.5))) . (:tensor :f32 (1))))
(defgraph sq ((c . (:tensor :f32 (1))) (x . (:tensor :f32 (4))))
  ((tsum (* c (* x x))) . :flonum))
(grad! (sq c xs))                          ;; градиент по c = 7.5
(defun analytic ((c . (:tensor :f32 (1))) (x . (:tensor :f32 (4))))
  ((tsum (term '(* (ones-like (* c (* x x))) (* x x)) '(c x))) . :flonum))
(print (tref (grad-of c) 0))   ;; → 7.5 — autograd
(print (analytic c xs))        ;; → 7.5 — ANALYTICAL-терм из (grad '(tsum (* c (* x x))) 'c)
```

### `diff`: унарные производные и n-арная арифметика

`diff` знает производные унарной математики и поддерживает арифметику `term`: n-арные `+`/`*`/`-`, унарный минус, правило произведения для любого числа множителей. `/` и `^` остаются бинарными; `^` над составным основанием связывает подтерм один раз через `let`.

```lisp
(import symbolic)
(diff '(sin x) 'x)          ;; → (COS X)
(diff '(cos x) 'x)          ;; → (* -1 (SIN X))
(diff '(sqrt x) 'x)         ;; → (/ 1 (* 2 (SQRT X)))
(diff '(atan x) 'x)         ;; → (/ 1 (+ 1 (^ X 2)))
(diff '(- x) 'x)            ;; → -1
(diff '(+ x 1 2) 'x)        ;; → 1
(diff '(* 2 3 x) 'x)        ;; → 6
(diff '(^ x 3) 'x)          ;; → (* 3 (^ X 2))
(diff '(^ (+ x 1) 3) 'x)    ;; → (* 3 (^ (+ X 1) 2))
```

Аналитические производные `sin/cos/exp/log/sqrt/tanh/atan` сверены с autograd по каждому оператору. `diff` на `relu` даёт `diff: relu is not differentiable symbolically`, на `matmul` — `diff: not elementwise; use grad`; на `softmax` — отправляет в `grad`. `term` транслирует `reshape`, `reshape-like`, `transpose`, `softmax`, `ones-like`, `tmax-mask`, `relu-mask` и унарную математику (гл. 14).

## 7.9 Контракт автоградиента

### Запуск графа — последний в форме

`ql_graph_run` сохраняет последний запуск (граф, входы, выход) в `Runtime.last_run`; в конце каждой top-level формы он сбрасывается. `grad!` работает только с этим последним запуском и **не проверяет**, что переданный аргумент — действительно его выход.

### Активации живут до `grad!`

Forward держит `frame.retain = 1`, поэтому `ql_hlo_release` не освобождает значения: все активации остаются в per-run `TensorBufferPool`. Отдельного recompute нет; повторный `grad!` или новый вызов графа освобождают пул.

### Stable-градиенты параметров

Градиенты накапливаются в scratch, но для param-входов копируются в stable и кладутся в `Tensor.grad`. Поэтому `(grad-of w)` доступен через именование параметра; у никогда не считанного параметра — нули той же формы/dtype.

### Что происходит на границе формы

| Сценарий | Поведение |
|---|---|
| `grad!` и чтение `(grad-of w)` в одной форме | Канонический паттерн: работает |
| `grad!` в одной форме, чтение `(grad-of w)` или `tset!` в следующей | Работает: градиент лежит в stable `Tensor.grad` |
| Повторный `grad!` без нового вызова графа | Не работает: `last_run` уже сброшен на границе формы |
| Параметр не участвовал в последнем запуске | Сохраняет прежний градиент |
| `clear` имени графа | Снимает callable, но сам граф/JIT-страница живут до teardown |
| Eager-вызов `relu`/`matmul` без `defgraph` | Графа нет, `grad!` нечего считать |

### Как объявлять параметры

Отдельной формы нет: `param` — обычная функция, обучаемый тензор объявляется через `defvar` **до** обучения:

```lisp
(defvar w ((param (randn :f32 (4 2))) . (:tensor :f32 (4 2))))
```

`defvar` кладёт значение в stable, `param` помечает `requires_grad`. Никакого макроса-обёртки не нужно.

## 7.10 Детерминированность и точность

Редукции (`tsum`/`tmean`) считаются восемью независимыми f64-полосами с фиксированным порядком сведения: результат **детерминирован**, но не совпадает с наивным «слева направо». Параллельный проход сводит partial'ы в том же фиксированном порядке, поэтому результат бит-в-бит совпадает с последовательным (гл. 6). Отсюда два практических правила:

1. Точное сравнение плавающих оправдано только для точно представимых значений; иначе сравнивайте с допуском (`np.allclose`-стиль).
2. Числовые пороги в тестах ставятся выше baseline вырожденного решения: если модель может сойтись к «всегда предсказывать самый частый класс», порог должен быть выше доли этого класса.

Специального предиката конечности (`finite?`) в текущей реализации нет: содержимое проверяют через `tref` и обычные сравнения. Поскольку порядок сведения зафиксирован, один и тот же граф на тех же входах даёт бит-в-бит тот же результат.

## 7.11 Известные ограничения

- `grad!` работает только с последним запуском графа в текущей top-level форме и не проверяет, что аргумент — его выход.
- Нет `tref`/`tset!`/`pow` в графе; `argmax`/`one-hot`/`gt`/`lt` — дискретные узлы без градиента. Редукции и matmul в графе — только float. `sin`/`cos`/`sqrt`/`atan`/`abs` в графе есть; `softmax`/`log-softmax` работают по последней оси любого ранга, `conv2d`/пулинг дифференцируются.
- `defgraph` с `:fixnum`-возвратом фактически недостижим.
- Нет вложенных `defgraph`-вызовов, рекурсии и data-dependent control flow; управление в теле должно быть статическим.
- Нет автоматического fusion/CSE/DCE: fusion только через явный `defuse`.
- Параметры, не участвовавшие в последнем запуске, сохраняют старый градиент; `grad-of` никогда не считанного параметра — нули.
- `param` на `:u8` и `:i64` запрещён: целочисленные тензоры — индексы и счётчики, а не обучаемые величины.
- `clear` имени снимает callable, но граф/JIT-страница живут до teardown.
- `defuse` внутри `defgraph` инлайнится перед трассировкой, поэтому его forward/backward идут обычными HLO-узлами; вне графа `defuse` компилирует AVX2-цикл сам.
- `(grad term var)` символичен и считает градиент скалярного лосса; `diff` остаётся поэлементным и отказывает на `relu`, `matmul` и `softmax`.
