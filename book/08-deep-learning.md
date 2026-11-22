# Глава 8. Глубокое обучение

Глава про построение нейросетей: слои как чистые функции, прямой проход, цикл обучения, классические метрики ML и состояние оптимизаторов. Модуль `dl` (`src/stdlib/dl.qlsp`, 10 777 байт) работает поверх eager-тензоров и autograd (гл. 6–7); `ml` отвечает за оценку и подготовку данных; символьный мост `neuro` описан в гл. 14. Среди доступных из DL-кода возможностей — унарная математика `sin`/`cos`/`sqrt`/`atan` в графе, термы градиента (`ones-like`, `tmax-mask`, `relu-mask`, `reshape-like`), символьный `(grad term var)` и параллельные ядра matmul/elementwise/reduce.

## 8.1 Модуль `dl`: чистые функции над тензорами

```lisp
(import DL)
```

`dl` даёт **чистые функции** над 2-D тензорами `:f32` формы «batch × features»: слой — это вызов `linear-forward`, сеть — `mlp-forward`, потери — `mse-loss`, шаг — `sgd-step`/`adam-step`. Переданные тензоры не мутируются, результат — новый тензор или новая структура. Исходник модуля использует парные аннотации: `(defun linear-forward ((x . (:tensor :f32 (_ _))) …) ((+ (matmul x w) b) . (:tensor :f32 (_ _))))`, поля структур — `(param . (:tensor :f32 (_ _)))`.

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

Проверенный вызов:

```lisp
(import dl)
(linear-forward (tensor ((1.0 2.0))) (tensor ((3.0) (4.0))) (tensor ((0.5))))
(mse-loss (tensor ((1.0 2.0))) (tensor ((0.0 0.0))))   ;; → 2.5
(sgd-step (tensor ((1.0 2.0))) (tensor ((0.5 0.25))) 0.1)
```

**Почему именно такие сигнатуры.**

- Нет полиморфизма по dtype и рангу: `(:tensor :f32 (_ _))` фиксирует и dtype `:f32`, и ранг 2. Литерал `tensor` тоже всегда `:f32`.
- Bias — строка `(1 m)`, а не вектор `(m)`: broadcast `(n m) + (1 m)` даёт per-column bias, и один шаг оптимизатора годится и для матрицы весов, и для bias.
- В рантайме нет доступа к форме тензора (нет `shape`), поэтому проверять формы внутри модуля нечем: несовпадение ловит тензорное ядро, а broadcast-совместимые, но разные формы посчитаются как broadcast. Формы — забота вызывающего.
- Имя `mse` занято модулем `ml`, и два экспорта с одним именем из разных модулей дают `ImportConflict`; поэтому тензорная потеря называется `mse-loss`.
- Результат шага имеет wildcard-форму `(_ _)`: `defvar`/`setq`, которые будут его получать, объявляются как `(:tensor :f32 (_ _))`; переменная с формой `(2 1)` такое значение отвергнет. Альтернатива — держать тензоры в структуре (`linear-model`, `adam-state`).
- Все функции чистые: `(setq w (sgd-step w g lr))` заменяет биндинг, а не мутирует ячейку `param`. Плата — scratch-тензор на каждый шаг, живущий до конца top-level формы.

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
| `(tset! w1 0 0 …)` | `opt.step()` | обновление элементов вручную, без in-place оптимизатора |
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

Свёрточных, пулинговых и нормализационных примитивов (`conv2d`, `maxpool2d`, `batchnorm`, `layernorm`, `dropout`) в текущей реализации нет; нет и 2-D `softmax`. Для построения выражений над тензорами из символьных термов служит модуль `neuro` (гл. 14) — он описан там же, вместе с примером свёрки `diff` и autograd.

## 8.4 Контракт autograd для DL

- Параметры создаются через `param` (гл. 7) и передаются в `defgraph` аргументами; константы тела графа градиента не получают.
- `(grad-of w)` — stable-градиент параметра (или нули той же формы). Обновление — `tset!` по `(grad-of w)` в той же top-level форме, где вызван `grad!`.
- Функции `dl` чистые: они не мутируют переданные тензоры и не «потребляют» градиент. Состояние Adam/momentum — отдельная структура (`adam-state`, `momentum-state`), а не поля тензора; `adam-step`/`momentum-step` возвращают **новое** состояние.
- Графа как значения языка нет: `defgraph` — мастер-форма, скомпилированный граф — производный артефакт. `(grad! …)` работает с последним запуском в текущей форме.
- Нет in-place оптимизатора, разреженных шагов по строкам и снимков состояния: для чекпоинта используйте `save-weights`/`load-weights` (гл. 6).
- Шаг обучения — пользовательская функция, которая в одной форме делает `grad!`, чтение `grad-of` и `tset!`; результат шага можно вернуть как loss.
- Альтернатива численному backward для скалярного лосса — символьный `(grad term var)` из `symbolic`: он возвращает терм градиента (`ones-like`, `tmax-mask`, `relu-mask`, `reshape-like`, `transpose`, `matmul` по обоим аргументам и т. д.), который можно компилировать через `term` и сверять с `grad-of` (гл. 7).

> 🐍 `optimizer.step()` в PyTorch заменяется явным проходом по элементам с `tset!`; `state_dict()` — структурой `adam-state`, которую можно передавать как обычное значение.

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

- Константы тела графа (`tensor`, `zeros`/`ones`/`randn`) трассируются один раз и становятся stable-тензорами графа: они переживают границу формы и не требуют `grad!`.
- Управление статическое: ветвления по значениям тензоров, `tref`/`tset!` внутри графа не поддерживаются.
- Под трассировкой нет рекурсии и вложенных `defgraph`; data-dependent control flow отсутствует.
- В графе доступны `sin`/`cos`/`sqrt`/`atan` с forward- и backward-ядрами; `pow`/`argmax` по-прежнему нет.
- Большие `matmul`, `elementwise`/`map`, `reshape`, `transpose`, `tsum`/`tmean` исполняются параллельно через `kernel_pool`; `max`, `softmax`, `argmax`, `reduceToShape` остаются последовательными. Параллельный проход побайтно совпадает с последовательным, `QUALISP_SERIAL=1` отключает пул (гл. 6).

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

## 8.7 Известные ограничения DL

- Модуль работает только с `:f32` и только с 2-D тензорами «batch × features»; полиморфизма по dtype и рангу нет.
- Нет слоёв-как-данных, `nn.Module`, `Sequential` с произвольным порядком и автоматической трассировки `model.forward`: графы собираются вручную.
- Нет свёрточных/пулинговых примитивов, нормализаций, dropout и 2-D softmax; символьные термы — в `neuro` (гл. 14). Унарная математика `sin`/`cos`/`sqrt`/`atan`, напротив, в графе есть.
- Нет in-place шага оптимизатора: веса обновляются `tset!` по `(grad-of w)`; шаги `dl` чистые и требуют wildcard-формы `(_ _)` у принимающих переменных.
- Нет разреженных шагов по строкам и снимков состояния оптимизатора: для сохранения весов — `save-weights`/`load-weights`.
- Нет прямого доступа к форме тензора из рантайма: несовпадение ловит ядро или broadcast, а не проверка в stdlib.
- Имена структур `dl` специфичны (теги не мангляются по модулю).
- `ml` считает метрики по спискам и одну метрику по 1-D тензору; 2-D вход в `t-mse` не типизируется.
- Параллельные ядра ускоряют matmul и elementwise на больших входах, но `max`/`softmax`/`argmax`/`reduceToShape` остаются последовательными; маленькие батчи (как XOR) parallel-пул не трогает.
