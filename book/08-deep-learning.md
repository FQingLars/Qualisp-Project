# Глава 8. Глубокое обучение

Глава про построение нейросетей: слои как чистые функции, прямой проход, цикл обучения, классические метрики ML и состояние оптимизаторов. Модуль `dl` (`src/stdlib/dl.qlsp`) работает поверх eager-тензоров и HLO-графов (гл. 6–7); `ml` отвечает за оценку и подготовку данных; символьный мост `neuro` описан в гл. 14. В 0.19.20 `dl` — не только линейные слои: есть LayerNorm, embedding, scaled dot-product attention, свёртки/пулинг, dropout, кросс-энтропия, инициализация и обновления параметров на месте. Среди доступных из DL-кода возможностей — `softmax`/`log-softmax` по последней оси, `abs`, `gather`/`scatter-add`, батчевый `matmul`, дифференцируемые `conv2d`/pooling, термы градиента (`ones-like`, `tmax-mask`, `relu-mask`, `reshape-like`), символьный `(grad term var)` и параллельные ядра.

## 8.1 Модуль `dl`: чистые функции над тензорами

```lisp
(import DL)
```

`dl` даёт **чистые функции** над тензорами `:f32`: линейные слои и MLP — «batch × features», attention/embedding/conv — свои ранги (`[B,T,d]`, `[N,C,H,W]`). Потери, нормализация и большая часть состояния оптимизаторов допускают ранговый wildcard, слои с фиксированной формой результата — конкретный ранг. Переданные тензоры не мутируются; результат — новый тензор или новая структура. Исходник модуля использует парные аннотации: `(defun linear-forward ((x . (:tensor :f32 (_ _))) …) ((+ (matmul x w) b) . (:tensor :f32 (_ _))))`, поля структур — `(param . (:tensor :f32 :any))`.

Структуры модуля (слоты — wildcard-формы `(_ _)`):

| Структура | Слоты | Смысл |
|---|---|---|
| `linear-model` | `w`, `b` | веса и смещение линейной модели |
| `linear-gradients` | `dw`, `db` | замкнутый градиент линейной регрессии |
| `adam-state` | `param`, `m`, `v`, `step` | состояние Adam |
| `momentum-state` | `param`, `velocity` | состояние моментного SGD |

Полный список экспортов:

| Функция | Что делает | Python-аналог |
|---|---|---|
| `linear-forward x w b` | `xW + b`, логиты | `nn.Linear` + вызов |
| `mlp-forward x w1 b1 w2 b2` | `relu(xW₁+b₁)W₂+b₂`, логиты | `nn.Sequential(Linear, ReLU, Linear)` |
| `mse-loss yhat y` | `tmean((yhat−y)²)` → `:flonum` | `F.mse_loss` |
| `squared-error pred actual` | `(pred−actual)²` → `:flonum` | `(pred - actual)**2` |
| `sgd-step w grad lr` | `w − lr·grad` | часть `optim.SGD.step()` |
| `adam-init w` | нулевые `m`, `v`, шаг 0 | создание `optim.Adam` |
| `adam-step st grad lr` | Adam с bias correction → новый `adam-state` | `optim.Adam.step()` |
| `adam-param st` | параметр из состояния | `param` |
| `adam-steps st` | номер шага → `:fixnum` | `state_dict()['step']` |
| `momentum-init w` | нулевая скорость | создание `optim.SGD(momentum=…)` |
| `momentum-step st grad lr momentum` | `v ← m·v + grad`, `w ← w − lr·v` | `optim.SGD(momentum=…).step()` |
| `momentum-param st` | параметр из состояния | `param` |
| `linear-grads x y w b` | `dW = xᵀ·err`, `db = sum(err)` | ручной `backward()` линейного слоя |
| `linear-regression-step x y w b lr` | один full-batch шаг → `linear-model` | ручной цикл SGD |
| `layernorm x eps` | нормализация по последней оси | `nn.LayerNorm` |
| `embedding-lookup rows table` | строки таблицы по `:i64`-индексам | `nn.Embedding` |
| `scaled-dot-attention q k v scale` | `softmax(QKᵀ·scale)V` для `[B,T,d]` | `F.scaled_dot_product_attention` |
| `conv2d-forward x w b stride pad` | свёртка + broadcast-bias, `b` — `[O,1,1]` | `F.conv2d` |
| `apply-mask x mask` | поэлементное произведение с готовой маской | `x * mask` |
| `dropout x p` | eager dropout, inverted scaling | `F.dropout(training=True)` |
| `mae-loss yhat y` | `tmean(|yhat−y|)` | `F.l1_loss` |
| `cross-entropy logits targets` | softmax + NLL по последней оси | `F.cross_entropy` |
| `cross-entropy-labels logits labels classes` | one-hot на лету, `:i64`-метки | `F.cross_entropy` |
| `accuracy-labels logits labels` | доля верных меток | `accuracy_score` |
| `init-weights rows cols scale` | `randn(list rows cols) * scale` | `nn.init` |
| `init-bias cols` | нулевой bias `(1 cols)` | `nn.init.zeros_` |
| `sgd-update! w grad lr` | обновляет `w` на месте | `optim.SGD.step()` |
| `adam-update! st grad lr` | обновляет параметр и моменты на месте | `optim.Adam.step()` |
| `clip-by-norm w max-norm` | глобальное ограничение нормы | `clip_grad_norm_` |
| `lr-decay base decay step` | расписание learning rate | `lr_scheduler` |
| `glorot-scale fan-in fan-out`, `he-scale fan-in` | множители инициализации | `calculate_gain` |

Проверенный вызов:

```lisp
(import dl)
(linear-forward (tensor ((1.0 2.0))) (tensor ((3.0) (4.0))) (tensor ((0.5))))
(mse-loss (tensor ((1.0 2.0))) (tensor ((0.0 0.0))))   ;; → 2.5
(sgd-step (tensor ((1.0 2.0))) (tensor ((0.5 0.25))) 0.1)
```

**Почему именно такие сигнатуры.**

- Dtype фиксирован: `dl` работает только с `:f32`. Ранг полиморфен не везде: `adam-state`/`momentum-state` и лоссы используют `:any`, а слои с известной формой результата — конкретный ранг. Литерал `tensor` без dtype тоже `:f32`.
- Bias — строка `(1 m)`, а не вектор `(m)`: broadcast `(n m) + (1 m)` даёт per-column bias, и один шаг оптимизатора годится и для матрицы весов, и для bias.
- Формы в основном берутся из типов, а не из значений; `init-weights`/`init-bias` принимают runtime-размеры через `(list n m)`, но ранг остаётся статическим, а форма результата — wildcard `(_ …)`. Несовпадение ловит тензорное ядро; broadcast-совместимые, но разные формы посчитаются как broadcast. Формы — забота вызывающего.
- Имя `mse` занято модулем `ml`, и два экспорта с одним именем из разных модулей дают `ImportConflict`; поэтому тензорная потеря называется `mse-loss`.
- Результат чистого шага имеет wildcard-форму `(_ _)`: `defvar`/`setq`, которые будут его получать, объявляются как `(:tensor :f32 (_ _))`; переменная с формой `(2 1)` такое значение отвергнет. Альтернатива — держать тензоры в структуре (`linear-model`, `adam-state`) или использовать `sgd-update!`/`adam-update!`.
- Чистые функции не мутируют входы: `(setq w (sgd-step w g lr))` заменяет биндинг. `sgd-update!`/`adam-update!` намеренно меняют stable-тензор на месте и сохраняют идентичность `param`; `adam-update!` копирует моменты в существующие буферы, не подменяя слоты структуры.

> 🐍 **Python-аналогия.** `linear-forward` ≈ `torch.nn.functional.linear`, `mlp-forward` ≈ маленький `nn.Sequential`, `adam-init`/`adam-step`/`adam-param` ≈ пара `optimizer = Adam(params)` + `optimizer.step()` + чтение `param`. Отличие: состояние — обычная структура, а не скрытый объект оптимизатора.

### Обучение без графа: замкнутый шаг

У линейной регрессии градиент известен в замкнутой форме, поэтому autograd не нужен: достаточно `matmul`/`transpose`/`tsum`.

```lisp
(import dl)
(defvar xs ((tensor ((1.0) (2.0) (3.0))) . (:tensor :f32 (3 1))))
(defvar ys ((tensor ((2.0) (4.0) (6.0))) . (:tensor :f32 (3 1))))
(defvar m ((linear-regression-step xs ys (tensor ((0.0))) (tensor ((0.0))) 0.1)
           . (:struct linear-model)))
(defvar epoch (0 . :fixnum))
(while (< epoch 100)
  (setq m (linear-regression-step xs ys (slot-value m 'w) (slot-value m 'b) 0.1))
  (setq epoch (+ epoch 1)))
(tref (slot-value m 'w) 0 0)   ;; → 1.9917595386505127 — близко к 2.0
(tref (slot-value m 'b) 0 0)   ;; → 0.01873299665749073 — близко к нулю
```

`linear-grads` считает `err = xW + b − y`, `dW = xᵀ·err`, `db = sum(err)`; `lr` относится к «суммарному» градиенту (при желании среднего по батчу делите `lr` на `n`). Один шаг даёт `w ≈ 2.8` — сходимость видна на последующих шагах.

### Слои за пределами MLP: нормализация, внимание, свёртки, dropout

Новые функции остаются обычными Qualisp-функциями, поэтому их можно вызывать и в eager-коде, и внутри `defgraph` (если операция трассируема):

```lisp
;; LayerNorm по последней оси
(defvar h ((randn :f32 (8 16)) . (:tensor :f32 (8 16))))
(layernorm h 1e-5)

;; embedding: rows — :i64-тензор индексов, table — [V, d]
(defvar ids ((tensor :i64 (1 3 2)) . (:tensor :i64 (3))))
(embedding-lookup ids (randn :f32 (10 4)))       ;; → (3 4)

;; attention для [B,T,d]
(defvar q ((randn :f32 (2 5 8)) . (:tensor :f32 (2 5 8))))
(defvar k ((randn :f32 (2 5 8)) . (:tensor :f32 (2 5 8))))
(defvar v ((randn :f32 (2 5 8)) . (:tensor :f32 (2 5 8))))
(scaled-dot-attention q k v (/ 1.0 (sqrt 8.0))) ;; → (2 5 8)

;; свёртка [N,C,H,W] × [O,C,KH,KW] + bias [O,1,1]
(conv2d-forward (randn :f32 (2 3 32 32)) (randn :f32 (8 3 3 3))
                (zeros :f32 (8 1 1)) 1 1)          ;; → (2 8 30 30)

;; eager dropout: маска строится rand-like; apply-mask трассируется
(dropout h 0.1)
(apply-mask h (t-gt (rand-like h) 0.1))
```

`dropout` — eager-функция с inverted scaling: при обучении активные элементы делятся на `1−p`, при инференсе маска не нужна. Внутри `defgraph` `rand-like` запрещён, поэтому маску генерируют снаружи и передают в `apply-mask`; это делает граф чистым и воспроизводимым при фиксированной маске.

`t-conv2d` поддерживает `stride`, `pad`, `dilation` и `groups`; `conv2d-forward` — удобная обёртка с `stride`/`pad` и broadcast-bias. Пулинг: `t-max-pool2d`/`t-avg-pool2d` с ядром, шагом и необязательным `pad`; внешние ячейки не участвуют (max их игнорирует, avg делит на число валидных). Backward для max-pool идёт по argmax, для avg — равномерно по валидным ячейкам.

### Потери, метки и обновление параметров

```lisp
(cross-entropy logits targets)                    ;; targets — вероятности
(cross-entropy-labels logits labels 10)           ;; labels — :i64, classes — fixnum
(accuracy-labels logits labels)                   ;; → :flonum
(mae-loss yhat y)
```

`cross-entropy` считает softmax по последней оси и отрицательное лог-правдоподобие; `cross-entropy-labels` строит one-hot на лету через `t-one-hot`. Обе функции дифференцируемы и проверены в `defgraph`.

Инициализация и оптимизаторы:

```lisp
(defvar w ((init-weights 128 64 (glorot-scale 128 64)) . (:tensor :f32 (_ _))))
(defvar b ((init-bias 64) . (:tensor :f32 (_ _))))
(sgd-update! w grad lr)
(adam-update! state grad lr)
```

`init-weights` принимает runtime-размеры `(list rows cols)`, но `randn` внутри всё равно требует литеральную форму — поэтому модуль комбинирует `randn` с множителем `glorot-scale`/`he-scale`. Ранг статический, а конкретные размеры результата — неизвестны: аннотация должна быть wildcard `(_ _)`; если нужна конкретная форма, используйте литеральный `(randn (rows cols))`. `adam-update!` обновляет `m`/`v` в существующих stable-тензорах; после этого `adam-steps` растёт, а `param` сохраняет идентичность.

> 🐍 `scaled-dot-attention` ≈ `F.scaled_dot_product_attention`, `layernorm` ≈ `nn.LayerNorm`, `embedding-lookup` ≈ `nn.Embedding`, `dropout`/`apply-mask` ≈ `F.dropout`/`x * mask`, `conv2d-forward` ≈ `F.conv2d`. В отличие от PyTorch, слои — функции, а состояние оптимизатора — явная структура.

## 8.2 Полный пример: XOR (DL)

Источник — `examples/xor.qlsp` (проверенный прогон). Сеть 2→3→1 с `tanh`, полный батч, SGD на месте через `tset!`:

```lisp
(defvar x   ((tensor ((0.0 0.0) (0.0 1.0) (1.0 0.0) (1.0 1.0))) . (:tensor :f32 (4 2))))
(defvar tgt ((tensor ((0.0) (1.0) (1.0) (0.0))) . (:tensor :f32 (4 1))))

(defvar w1 ((param (tensor ((0.5 -0.4 0.3) (-0.2 0.6 -0.5)))) . (:tensor :f32 (2 3))))
(defvar b1 ((param (tensor (0.1 -0.1 0.0))) . (:tensor :f32 (3))))
(defvar w2 ((param (tensor ((0.7) (-0.8) (0.9)))) . (:tensor :f32 (3 1))))
(defvar b2 ((param (tensor (0.0))) . (:tensor :f32 (1))))

(defgraph xor-loss ((x . (:tensor :f32 (4 2))) (w1 . (:tensor :f32 (2 3))) (b1 . (:tensor :f32 (3)))
                    (w2 . (:tensor :f32 (3 1))) (b2 . (:tensor :f32 (1))) (tgt . (:tensor :f32 (4 1))))
  ((let ((h (tanh (+ (matmul x w1) b1))))
     (let ((p (+ (matmul h w2) b2)))
       (let ((d (- p tgt)))
         (tmean (* d d))))) . :flonum))

(defgraph xor-predict ((x . (:tensor :f32 (4 2))) (w1 . (:tensor :f32 (2 3))) (b1 . (:tensor :f32 (3)))
                       (w2 . (:tensor :f32 (3 1))) (b2 . (:tensor :f32 (1))))
  ((+ (matmul (tanh (+ (matmul x w1) b1)) w2) b2) . (:tensor :f32 (4 1))))

(defvar lr (0.5 . :flonum))
(defvar epoch (0 . :fixnum))
(while (< epoch 5000)
  (grad! (xor-loss x w1 b1 w2 b2 tgt))
  (tset! w1 0 0 (- (tref w1 0 0) (* lr (tref (grad-of w1) 0 0))))
  ;; … остальные элементы w1, b1, w2, b2 — по той же схеме
  (tset! b2 0 (- (tref b2 0) (* lr (tref (grad-of b2) 0))))
  (setq epoch (+ epoch 1)))

(defvar y ((xor-predict x w1 b1 w2 b2) . (:tensor :f32 (4 1))))
(print (tref y 0 0))   ;; → ≈ 0
(print (tref y 1 0))   ;; → ≈ 1
```

> Веса — `param` и аргументы графа: всё, что трассировщик видит как константу (`tensor`-литерал в теле), градиента не получит. `let` связывает параллельно, поэтому зависимые биндинги вложены. Матрицы здесь маленькие (4×2, 2×3), поэтому parallel-пул их не трогает: маленькие входы остаются последовательными и результат побитово совпадает с serial (гл. 6).

### Python-эквивалент (PyTorch)

```python
import torch
import torch.nn as nn

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([[0.], [1.], [1.], [0.]])
model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
opt = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(5000):
    pred = model(X)
    loss = nn.functional.mse_loss(pred, Y)
    opt.zero_grad()
    loss.backward()
    opt.step()

print(model(X))  # → [[~0], [~1], [~1], [~0]]
```

### Ключевые различия

| Qualisp | PyTorch | Пояснение |
|---|---|---|
| `(defgraph xor-loss …)` | `forward` + `loss` модели | граф задаётся явно, компилируется лениво |
| `(defvar w1 … (param …))` | `nn.Parameter` | параметры — stable-тензоры с `requires_grad` |
| `(grad! (xor-loss …))` | `loss.backward()` | backward по последнему запуску графа |
| `(tset! w1 0 0 …)` / `(sgd-update! w1 g lr)` | `opt.step()` | XOR обновляет элементы вручную; в eager-коде есть in-place шаги |
| `(xor-predict …)` | `model(X)` | отдельный граф для инференса |

## 8.3 Классические метрики: модуль `ml`

Модуль `ml` (`src/stdlib/ml.qlsp`) не строит моделей — он считает метрики и готовит данные. Метрики определены на непустых `(:list :flonum)`; `:fixnum`-входы расширяются до `:flonum`, результат всегда `:flonum`.

| Функция | Что делает | Python-аналог |
|---|---|---|
| `mean xs` | среднее | `np.mean` |
| `mse pred actual` | средний квадрат по спискам | `sklearn.metrics.mean_squared_error` |
| `mae pred actual` | средняя абсолютная ошибка | `mean_absolute_error` |
| `rmse pred actual` | корень из MSE | `sqrt(mean_squared_error)` |
| `accuracy pred actual` | доля совпадений, метки любого атомарного типа | `accuracy_score` |
| `precision` / `recall` / `f1` | бинарные метки 1/0, нулевой знаменатель → `0.0` | `precision_score` / `recall_score` / `f1_score` |
| `train-part xs ratio` / `test-part xs ratio` | разбиение, пустая часть → `nil` | `train_test_split` |
| `minmax-scale xs` | `[0,1]`, вырожденный диапазон → нули | `MinMaxScaler` |
| `standardize xs` | генеральное СКО (делим на n) | `StandardScaler` |
| `t-mse pred actual` | MSE для 1-D `(:tensor :f32 (_))` | `F.mse_loss` на векторе |

```lisp
(import ml)
(mean (list 1.0 2.0 3.0))                 ;; → 2.0
(mse (list 1.0 2.0) (list 0.0 0.0))       ;; → 2.5
(t-mse (tensor (1.0 2.0)) (tensor (0.0 0.0)))  ;; → 2.5
(train-part (list 1.0 2.0 3.0 4.0) 0.5)   ;; → (1.0 2.0)
(accuracy (list 1 0 1) (list 1 0 0))      ;; → 0.6666666666666666
```

`train-part`/`test-part` возвращают `(:option (:list T))` (пустая часть — `nil`), ratio вне `0..1` — `throw`, округление `ratio·n` с допуском `1e-9`.

В 0.19.20 свёртки, пулинг, LayerNorm, dropout, embedding и attention живут в `dl`/тензорных примитивах, а не в `ml`; `batchnorm` как готового слоя по-прежнему нет, `softmax` работает по последней оси любого ранга. Для построения выражений над тензорами из символьных термов служит модуль `neuro` (гл. 14) — он описан там же, вместе с примером сверки `diff` и autograd.

## 8.4 Контракт autograd для DL

- Параметры создаются через `param` (гл. 7) и передаются в `defgraph` аргументами; константы тела графа градиента не получают.
- `(grad-of w)` — stable-градиент параметра (или нули той же формы). Обновление — `tset!` по `(grad-of w)` в той же top-level форме, где вызван `grad!`.
- Функции `dl` чистые: они не мутируют переданные тензоры и не «потребляют» градиент. Состояние Adam/momentum — отдельная структура (`adam-state`, `momentum-state`), а не поля тензора; `adam-step`/`momentum-step` возвращают **новое** состояние. `sgd-update!`/`adam-update!` — явные исключения: они меняют stable-тензор и состояние на месте.
- `adam-step` возвращает состояние, чьи `m`/`v` — scratch-тензоры; `adam-update!` копирует их в **существующие** stable-буферы через `t-copy!`, поэтому структура не держит ссылку на умерший scratch. Это важно при пересечении границы top-level формы.
- Графа как значения языка нет: `defgraph` — мастер-форма, скомпилированный граф — производный артефакт. `(grad! …)` работает с последним запуском в текущей форме.
- Разреженных шагов по строкам и снимков состояния оптимизатора нет: для чекпоинта используйте `save-weights`/`load-weights` (гл. 6).
- Шаг обучения — пользовательская функция, которая в одной форме делает `grad!`, чтение `grad-of` и `tset!`; результат шага можно вернуть как loss.
- Альтернатива численному backward для скалярного лосса — символьный `(grad term var)` из `symbolic`: он возвращает терм градиента (`ones-like`, `tmax-mask`, `relu-mask`, `reshape-like`, `transpose`, `matmul` по обоим аргументам и т. д.), который можно компилировать через `term` и сверять с `grad-of` (гл. 7).

> 🐍 `optimizer.step()` в PyTorch соответствует либо явному проходу по элементам с `tset!`, либо in-place `(sgd-update! w g lr)` / `(adam-update! st g lr)`; `state_dict()` — структурой `adam-state`, которую можно передавать как обычное значение.

## 8.5 HLO-инференс модели (гл. 10)

Граф задаётся `defgraph`, а его вызов и есть forward. Для тех же весов обычно определяется `xor-predict`-граф, отличный от графа потерь:

```lisp
(defgraph predict ((x . (:tensor :f32 (4 2))) (w1 . (:tensor :f32 (2 3))) (b1 . (:tensor :f32 (3)))
                   (w2 . (:tensor :f32 (3 1))) (b2 . (:tensor :f32 (1))))
  ((+ (matmul (tanh (+ (matmul x w1) b1)) w2) b2) . (:tensor :f32 (4 1))))

(defvar y ((predict x w1 b1 w2 b2) . (:tensor :f32 (4 1))))   ;; нативный инференс
```

Первый вызов графа компилирует его в x86-64 (fused-ядра — call-threaded, W^X-страница), последующие идут без ожидания; фоновая компиляция — `N = min(cpu_count, 8)` воркеров. Для одного elementwise-выражения вместо графа можно определить `defuse`:

```lisp
(defuse affine ((x . (:tensor :f32 (4))) (a . :flonum) (b . :flonum))
  ((relu (+ (* x a) b)) . (:tensor :f32 (4))))
```

Eager-вызов `defuse` компилирует скалярный f32-цикл с AVX2 по 8 чисел + scalar-tail, без промежуточных тензоров. Внутри `defgraph` вызов `defuse` инлайнится в `let` перед трассировкой, поэтому такой forward/backward идёт обычными HLO-узлами.

**Контракты трассировки, которые важны на практике:**

- Константы тела графа (`tensor`, `zeros`/`ones`/`randn`) трассируются один раз и становятся stable-тензорами графа: они переживают границу формы и не требуют `grad!`. `t-copy!` и `rand-like` в трассировке запрещены: граф остаётся чистой функцией, случайность и запись передаются маской/входом.
- Управление статическое: ветвления по значениям тензоров, `tref`/`tset!` внутри графа не поддерживаются.
- Под трассировкой нет рекурсии и вложенных `defgraph`; data-dependent control flow отсутствует. `ns-if` трассируется отдельным узлом, а `ns-band` в v1 остаётся eager-формой.
- В графе доступны `sin`/`cos`/`sqrt`/`atan` с forward- и backward-ядрами; `abs` дифференцируем, `softmax`/`log-softmax` считаются по последней оси, `argmax_axis`/`one_hot`/`gt`/`lt` — дискретные узлы без градиента. Добавлены `slice`, `concat`, `conv2d`, `max_pool2d`, `avg_pool2d`, `gather`, `scatter_add`; `pow` по-прежнему нет.
- Большие `matmul` (включая batched), `elementwise`/`map`, `reshape`, `transpose`, `tsum`/`tmean`, осевые редукции и `softmax`/`log-softmax` (по строкам последней оси) исполняются параллельно через `kernel_pool`; `conv2d`/pooling параллелятся по батчу, backward — отдельными проходами `dx`/`dw`. `max` и `argmax` остаются последовательными. Параллельный проход побайтно совпадает с последовательным, `QUALISP_SERIAL=1` отключает пул (гл. 6).

> 🐍 Аналог `torch.compile(model)` — но граф собирается вручную через `defgraph`/`defuse`, без автоматической трассировки `model.forward`.

## 8.6 Инференс без градиента

Поскольку записи на ленту нет, инференс — это просто вызов eager-функции `dl` или `defgraph` без `grad!`: узлы backward не строятся сами по себе, даже если тензор помечен `param`.

```lisp
(import dl)
(defvar logits ((mlp-forward x w1 b1 w2 b2) . (:tensor :f32 (_ _))))  ;; обычный вызов
(defvar yhat ((predict x w1 b1 w2 b2) . (:tensor :f32 (4 1))))        ;; вызов графа
```

Специальной формы `no_grad` не нужно: `param` лишь разрешает `grad!` заполнить `grad-of`, а не меняет поведение самих операций. Термы `neuro` добавляют к этому символьную проверку градиента (гл. 14).

> 🐍 Это `with torch.no_grad(): model(x)` — но без контекстного менеджера: те же функции работают и на обучении, и на инференсе, разница лишь в том, вызывается ли `grad!`.

## 8.7 Сравнение с PyTorch CPU

В репозитории есть стенд `bench/pytorch/`: четыре шага обучения — `mlp`, `conv`, `attention`, `embedding` — записаны дважды, идиоматичным PyTorch и Qualisp-графом на `dl`. Протокол общий: три прогрева, затем лучшее из трёх серий по десять шагов, серии чередуются между реализациями. Замеры ReleaseFast на CPU с 8 потоками, PyTorch 2.13:

| Шаг | Qualisp, мс/шаг | PyTorch, мс/шаг | Отношение |
|---|---|---|---|
| MLP | 0.495 | 0.354 | ~1.4× |
| Conv | 3.222 (повторный прогон 2.324) | 0.684 (0.645) | ~4.7× (3.6×) |
| Attention | 1.992 (1.910) | 0.409 (0.419) | ~4.9× (4.6×) |
| Embedding | 0.306 | 0.277 | ~1.1× |

Разрыв объясняется не интерпретатором, а ценой обобщённых ядер: до 0.19.19 broadcast-операции считали смещение по одному элементу, свёртка шла скалярным циклом через `get`/`set`, а каждый промежуточный тензор брал свежий буфер у аллокатора. Специализированные f32/f64-ядра свёртки, строчные broadcast/reduce-ядра и пул eager-скретча сократили разрыв; оставшиеся кандидаты — регистровый ABI вызова и более агрессивная специализация. Стенд — не часть stdlib: он показывает, где именно Qualisp платит за универсальность.

## 8.8 Известные ограничения DL

- Модуль работает только с `:f32`; dtype-полиморфизма нет. Ранг полиморфен частично: потери и состояние оптимизатора принимают `(:tensor :f32 :any)`, а слои с фиксированной формой результата требуют конкретный ранг.
- Нет слоёв-как-данных, `nn.Module`, `Sequential` с произвольным порядком и автоматической трассировки `model.forward`: графы собираются вручную.
- `conv2d`, max/avg pooling, LayerNorm, dropout, embedding и scaled-dot attention есть; `batchnorm` как готового слоя по-прежнему нет. `sigmoid` собирается выражением, отдельного примитива нет.
- In-place шаги `sgd-update!`/`adam-update!` существуют и сохраняют идентичность `param`; чистые `sgd-step`/`adam-step` возвращают новое состояние и требуют wildcard-формы у принимающих переменных.
- Нет разреженных шагов по строкам и снимков состояния оптимизатора: для сохранения весов — `save-weights`/`load-weights`.
- Нет прямого доступа к форме тензора из рантайма: несовпадение ловит ядро или broadcast, а не проверка в stdlib; `init-weights`/`init-bias` принимают `(list n m)`, но ранг остаётся статическим, а конкретные размеры — неизвестны (`_`).
- Имена структур `dl` специфичны (теги не мангляются по модулю).
- `ml` считает метрики по спискам и одну метрику по 1-D тензору; 2-D вход в `t-mse` не типизируется.
- `:i64` — данные (индексы, метки), не параметр: `param`/`grad` для него запрещены. `argmax`/`one-hot`/`gt`/`lt` не дифференцируются.
- Быстрые векторные ядра свёртки есть только для f32/f64; f16/u8 идут общим путём. `max`/`argmax` остаются последовательными; маленькие батчи (как XOR) parallel-пул не трогает.
