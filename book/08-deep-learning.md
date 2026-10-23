# Глава 8. Глубокое обучение

Глава про построение нейросетей: слои как данные, прямой проход, цикл обучения, свёрточные примитивы, нормализация, Adam-состояние.

## 8.1 Модуль `dl`: слои как данные

Модуль `dl` (`stdlib/dl.qlsp`) реализует нейросеть **как данные**: модель — это обычный список, в котором слои лежат cons-ячейками, а активации — голые символы. Модель не объект, не замыкание, а значение, которое можно распечатать, сериализовать и передать куда угодно.

```lisp
(import DL)
;;                       in    out
(defvar model (list (linear 1 4)     ;; слой: (list 'linear W b)
                    'relu            ;; активация: голый символ
                    (linear 4 1)))   ;; последний слой
```

Слой — это cons-ячейка формы `(linear W B)`, где `W` и `B` — обучаемые параметры (`PARAM`, requires_grad=true). Активации — голые символы: `'relu`, `'sigmoid`, `'tanh`, `'softmax`. Оптимизаторы — тоже символы: `'sgd`, `'adam`.

Почему именно данные, а не замыкания:

- **HibLin-безопасность между формами.** Замыкание захватывает `w`/`b` в scratch-closure, и параметры переживают границу форм по собственным правилам. Список-структура параметризуется явно через `defvar`/промоут — и переживает границы форм по тем же правилам, что и любое другое значение.
- **QSRD v2-сериализуемость.** Список-структура сериализуется QSRD v2 (гл. 16) — можно сохранять и загружать обученные модели как данные, без специального протокола. Тензоры весов сериализуются на **любой глубине** структуры, так что модель — alist слоёв с весами — сохраняется и загружается целиком с dtype/shape.
- **Контракт tape.** Слои-как-данные не «прячут» узлы ленты в замыканиях — граф полностью видим в scope.

### Создание слоя: `linear`

```lisp
;; Полносвязный слой: (linear in out) → (list 'linear W B)
;; Конвенция батча: x [N in] → y [N out]
(defun linear (in-features out-features)
  (list 'linear
        (param (randn (list in-features out-features)))
        (param (zeros (list 1 out-features)))))
```

> 🐍 Аналог `nn.Linear(in_features, out_features)`. В `Linear` PyTorch смещение — `[out_features]` (broadcast), здесь — `[1, out_features]` (явная форма `[N, out]`, broadcast-friendly). В остальном — прямой аналог.

Доступ к весам и смещению слоя (если нужно — для отладки, для ручного обновления):

```lisp
(layer-w (car model))       ; → W параметр первого слоя
(layer-b (car model))       ; → B параметр первого слоя
```

### Прямой проход: `net-forward`

```lisp
(defun net-forward (model x)
  (if (null model)
      x
      (let ((layer (car model)))
        (if (symbolp layer)
            (net-forward (cdr model) (apply-activation layer x))
            (net-forward (cdr model)
                         (t+! (matmul! x (layer-w layer))
                              (layer-b layer)))))))
;; или просто:
(net-forward model x)
```

> 🐍 Аналог `model(x)` для `nn.Sequential`. Разница: `net-forward` — рекурсия по cons-списку, без `for`/`while`, без Python-циклов. Это согласуется с HibLin-семантикой — каждый шаг рекурсии в новом scope.

`apply-activation` — внутренний helper:

```lisp
;; (В stdlib/dl.qlsp определены макросы/функции для каждой активации)
(relu x)         ; max(x, 0)
(sigmoid x)      ; 1 / (1 + exp(-x))
(tanh x)         ; стандартный tanh
(softmax x)      ; softmax по последней оси
```

> 🐍 Аналог `F.relu(x)`, `F.sigmoid(x)`, `torch.tanh(x)`, `F.softmax(x, dim=-1)`.

### Шаг обучения: `train-step`

```lisp
(train-step model x y-true optimizer lr)
;;   model       : список-структура
;;   x, y-true   : батч [N, in] и ground truth [N, out]
;;   optimizer   : 'sgd или 'adam
;;   lr          : learning rate
;; → последний loss (mse!)
```

Реализация (из `stdlib/dl.qlsp`):

```lisp
(defun train-step (model x y-true optimizer lr)
  (let ((loss (mse! (net-forward model x) y-true)))
    (GRAD! loss)
    (opt-apply model optimizer lr)
    loss))
```

> 🐍 Аналог одного шага `loss.backward(); opt.step(); return loss` в PyTorch. Отличие: `GRAD!` объединяет `backward` + `tape.clear()`.

### `opt-apply` — шаг оптимизатора

```lisp
(defun opt-apply (model optimizer lr)
  (if (null model) nil
    (progn
      (let ((layer (car model)))
        (if (equal (car layer) 'linear)
            (if (equal optimizer 'adam)
                (progn (ADAM-STEP (layer-w layer) lr)
                       (ADAM-STEP (layer-b layer) lr))
              (progn (SGD-STEP (layer-w layer) lr)
                     (SGD-STEP (layer-b layer) lr)))))
      (opt-apply-next (cdr model) optimizer lr))))   ;; мост, НЕ self-вызов
```

> 🐍 Аналог цикла по `model.parameters()` в PyTorch — здесь рекурсия по cons-списку.

**Почему через мост, а не прямо в хвост.** Это не стилистика, а контракт владения.

Обход модели **не должен перебиндинживать список**. Если список перебиндить, копия будет глубокой, а вместе с ней — копии весов; оптимизатор тогда работает с копией, которая умирает вместе со scope, и обучение не идёт. Два пути к этому — self-хвостовая рекурсия (TCO пере-биндит кадр, то есть клонирует каждый слой) и перебор через `while` + `setq` (`(setq lst (cdr lst))` первым пере-биндингом клонирует весь хвост вместе с `requires_grad`-тензорами).

Мост `opt-apply-next` делает вызов **не-self**, поэтому TCO не срабатывает, клон не нужен, а `(car model)` читает оригинальные стабильные ячейки. Глубина рекурсии = число слоёв. `net-forward` устроен так же (`net-forward-tail`).

### `train-epochs` — N эпох

```lisp
(defun train-epochs (model x y-true optimizer lr epochs)
  (let ((loss nil) (i 0))
    (while (< i epochs)
      (setq loss (train-step model x y-true optimizer lr))
      (setq i (+ i 1)))
    loss))
```

> 🐍 Аналог `for epoch in range(epochs): ...` в Python. В Qualisp — `while`; рекурсия осталась для `net-forward` и `opt-apply` ради согласованности с HibLin-семантикой (внутри `while` откат итерации рвёт узлы ленты, поэтому forward не делают циклом).

## 8.2 Полный пример: XOR (DL)

```lisp
(import DL)

;; XOR-датасет
(defvar X (tensor ((0.0 0.0) (0.0 1.0) (1.0 0.0) (1.0 1.0))))
(defvar Y (tensor ((0.0) (1.0) (1.0) (0.0))))

;; Модель: 2 → 4 (ReLU) → 1
(defvar model (list (linear 2 4)
                    'relu
                    (linear 4 1)))

;; Обучение: 500 эпох, SGD, lr=0.5
(train-epochs model X Y 'sgd 0.5 500)

;; Проверка
(net-forward model X)
;; → тензор, близкий к [[0], [1], [1], [0]]
```

### Python-эквивалент (PyTorch)

```python
import torch
import torch.nn as nn

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([[0.], [1.], [1.], [0.]])

model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
opt = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(500):
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
| `(defvar model (list ...))` | `nn.Sequential(...)` | модель — данные, а не объект |
| `(net-forward model x)` | `model(x)` | прямой проход по cons-списку (рекурсия) |
| `(train-step model x y 'sgd lr)` | `loss.backward(); opt.step()` | одна функция = forward + loss + backward + step |
| `(train-epochs model x y 'sgd lr 500)` | `for epoch in range(500): ...` | N эпох, `while` |
| `'sgd`, `'adam` (символы) | `optim.SGD`, `optim.Adam` (объекты) | оптимизатор — выбираемый символ |

## 8.3 Свёрточные примитивы

Вне модуля `dl` (в ядре) есть autograd-операции для свёрточных сетей:

| Примитив | Что делает | Python-аналог |
|---|---|---|
| `(conv2d! x k s p)` | 2D-свёртка | `F.conv2d(x, k, stride=s, padding=p)` |
| `(maxpool2d! x k s)` | 2D max-pooling | `F.max_pool2d(x, kernel_size=k, stride=s)` |
| `(batchnorm! h)` | batch normalization | `nn.BatchNorm1d`/`BatchNorm2d` |
| `(layernorm! h)` | layer normalization | `nn.LayerNorm` |
| `(dropout! h 0.5)` | dropout (p=0.5) | `nn.Dropout(0.5)` |
| `(weighted-lookup probs v1 v2)` | взвешенная сумма (скаляр) | ручное `probs[0]*v1 + probs[1]*v2` |
| `(argmax t)` / `(softmax! t)` | argmax / softmax | `t.argmax()` / `F.softmax(t, dim=-1)` |

Все эти примитивы **записывают на ленту** (суффикс `!`) и участвуют в `(grad! loss)`.

Арность и типы аргументов проверяются: неверное число аргументов или нечисловой `eps` — внятная ошибка, а не чтение мусора.

> 🐍 Эти операции не имеют «удобного» Python-аналога, объединяющего в один prim — в PyTorch каждая идёт через `torch.nn.functional` или `torch.Tensor`. В Qualisp всё встроено.

## 8.4 Контракт tape для DL

- Параметры создаются через `PARAM` (requires_grad=true), узел ленты живёт на самом тензоре и **переживает HibLin-promote** в stable-скоп.
- Adam-состояние (`m`/`v`/`step`) аллоцируется в stable-скоп **на самом параметре** — состояние оптимизатора не умирает с итерацией и не переаллоцируется: шаг оптимизатора читает оригинальные стабильные ячейки модели (см. `opt-apply` выше), а header стабильного тензора освобождается вместе с буфером, а не остаётся в `stable-bytes` навсегда.
- Все `!`-операции внутри `net-forward` и `train-step` пишут на tape. После `GRAD!` tape пуст; `SGD-STEP`/`ADAM-STEP` засчитывают градиент прочитанным (гл. 7).
- Свою дифференцируемую функцию можно описать прямо в программе — `deftape!`: прямая часть на `!`-операциях, обратная — лямбда со списком градиентов (гл. 7.8).
- Градиент не переживает границу формы: `GRAD!`, чтение градиента и шаг оптимизатора должны происходить в одной форме. `train-step` делает всё три в одной форме, поэтому обучение через `train-epochs` работает.
- `(train-step ...)` возвращает loss — можно накапливать в список для логов.

## 8.5 HLO-инференс модели (гл. 10)

После обучения можно скомпилировать прямой проход в нативный код:

```lisp
(start-trace)
(defvar dx (graph-param X))
;; собрать граф заново (трейс на стабильных входах)
(defvar h1 (relu (t+ (matmul dx (layer-w (car model))) (layer-b (car model)))))
(defvar out (t+ (matmul h1 (layer-w (car (cdr (cdr model)))))
                (layer-b (car (cdr (cdr model))))))
(defvar hlo (hlo-compile out))
(defvar result (hlo-run hlo X))   ; нативный инференс
```

> 🐍 Аналог `torch.compile(model)` — но требует пересборки графа вручную (нет автоматической трассировки `model.forward`). Это явный trade-off.

Два контракта трассировки, которые важны на практике:

- **Константы графа владеют своими байтами (record-time seal).** Константа, записанная трейсом, клонируется в хранилище, живущее ровно столько же, сколько идёт сессия трассировки (`trace_store_`), и обнуляется на новом `START-TRACE`. `seal_graph_constants` копирует константы из хранилища, которое заведомо переживает граф, — поэтому кросс-форменная компиляция (трассировка в одной форме, `HLO-COMPILE` в следующей) безопасна.
- **Под трассировкой рекурсия не хвостовая.** Граф накапливается **сквозь** итерации self-хвостового вызова, а откат scratch на итерации освобождал прокси и тензоры-операнды, на которые граф ещё ссылается. Поэтому трассируемая рекурсия всегда идёт обычным вызовом, удерживая все кадры (и их узлы графа).

Подробности — в гл. 10 (HLO-компиляция).

## 8.6 Инференс без ленты и REINFORCE

### `net-apply`, `net-infer`, `net-infer-apply` — слой и tape-free инференс

Обучающие `!`-операции внутри `net-forward` пишут на ленту, а в циклах tape-опы без `GRAD!` рвут граф. Модуль `dl` разделяет шаг слоя и проход по модели на две пары:

```lisp
(net-apply layer x)          ;; шаг слоя с записью на ленту: linear/relu/sigmoid/tanh/softmax
(net-forward model x)         ;; проход по модели (использует net-apply)

(net-infer-apply layer x)     ;; шаг слоя без ленты: linear и relu
(net-infer model x)           ;; проход по модели без записи на ленту
```

Tape-free вариант поддерживает только слои `linear` и `relu` — для остальных kinds он бросает внятную ошибку (`net-infer: unsupported layer kind for tape-free inference`).

> 🐍 Аналог `with torch.no_grad(): model(x)` — но без контекста: это просто другие функции, и в горячем `while` они не создают нод ленты вовсе.

### REINFORCE: `SAMPLE` и `reinforce-loss!`

В ядре есть примитив **`SAMPLE`** (категориальный softmax-сэмплинг one-hot действий, детерминированный сид как у `RANDN`; невыбранные слоты заполняются нулями явно). В `stdlib/dl.qlsp` на нём построено REINFORCE-ядро:

```lisp
(defvar action (sample logits))               ;; one-hot [batch, N]
(reinforce-loss! logits action-index reward)  ;; reward × CROSS-ENTROPY!(logits, index)
```

- `reinforce-loss!` = reward × `CROSS-ENTROPY!(logits, action-index)` с градиентом `reward·(p − onehot)`; таргеты — целочисленные индексы `[batch]`, upstream-градиент (множитель reward) протаскивается в backward.
- Полный REINFORCE с baseline — в stdlib (`tests/nsrl_t1.qlsp`: ручные значения CE, знаки градиентов при ±reward, детерминированный 2-рукий бандит сходится).
- RL-надстройка (макро-слоты, `EVAL-SANDBOXED`, роутинг) — в гл. 12 и 14.

## 8.7 Известные ограничения DL

- **Нет автоматической трассировки `model.forward`.** Чтобы скомпилировать в HLO, нужно вручную собрать cons-граф из параметров — `(graph-param w)`, `(graph-param b)`, и т.д. Удобство в том, что это явное.
- **Нет `nn.Module`-подобной интроспекции.** Чтобы получить список параметров модели, нужно рекурсивно обойти cons-список — `opt-apply` делает это внутри.
- **Нет `Sequential` с произвольным порядком.** Только `linear` → activation → `linear` → … (без `BatchNorm`/`Dropout` как слоёв). Свёрточные примитивы (`conv2d!`, `maxpool2d!`) — autograd-примитивы, а не слои; их можно вставить в HLO-граф, но не в `net-forward` напрямую.
- **`opt-apply` рекурсивный** — для очень глубоких серий может быть стек-переполнение. На практике не проблема (модели редко >100 слоёв).
