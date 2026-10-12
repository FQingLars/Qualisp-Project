# Глава 17. Разбор примеров

Два канонических примера из `examples/` исходного репозитория, разобранных построчно, с Python-эквивалентами. NB: имена примитивов и функций в Qualisp после `intern` хранятся в верхнем регистре (`T+`, `MATMUL!`, `MSE!`, `PARAM`, `GRAD!`), но из-за регистронезависимости вы можете писать их как угодно — книга использует стиль «как есть», для соответствия Lisp-идиому. Все вызовы тензорных функций возвращают тензоры; `!`-операции записывают узлы на autograd-ленту.

## 17.1 Линейная регрессия: обучение + HLO-инференс

**Задача.** Синтетические данные `y = 2x + 1 + noise`. Обучаем `w`, `b` методом MSE + SGD, затем компилируем forward в нативный код.

### Полный код

```lisp
;; ----- данные -----
(defvar x-data (randn (list 100 1)))
(defvar noise (t* (randn (list 100 1)) (tensor ((0.2)))))
(defvar y-data (t+ (t+ (t* x-data (tensor ((2.0)))) (tensor ((1.0)))) noise))

;; ----- параметры -----
(defvar w (param (randn (list 1 1))))
(defvar b (param (zeros (list 1 1))))
(defvar lr 0.02)

;; ----- forward -----
(defun predict (x) (t+! (matmul! x w) b))

;; ----- обучение -----
(defun train-loop (iter max-iter)
  (if (> iter max-iter) nil
    (progn
      (let ((yp (predict x-data))
            (loss (mse! yp y-data)))
        (grad! loss)
        (sgd-step w lr)
        (sgd-step b lr)
        (if (equal (mod iter 40) 0)
          (progn (print "epoch") (print iter) (print loss)))
        (train-loop (+ iter 1) max-iter)))))

(train-loop 1 200)

;; ----- компиляция инференса в HLO -----
(defvar x-test (randn (list 10 1)))   ;; свежие тестовые данные
(start-trace)
(defvar dx (graph-param x-test))
(defvar dw (graph-param w))
(defvar db (graph-param b))
(defvar dp (t+ (matmul dx dw) db))

(defvar hlo (hlo-compile dp))
(defvar hlo-result (hlo-run hlo x-test w b))
```

### Построчный разбор

| Строки | Что происходит | Python |
|---|---|---|
| `x-data`, `noise`, `y-data` | Датасет 100×1: `y = 2x + 1 + 0.2·ε` | `x = np.random.randn(100,1); y = 2*x + 1 + 0.2*np.random.randn(100,1)` |
| `w`, `b` | Обучаемые параметры `[1,1]` | `w = nn.Parameter(torch.randn(1,1))` |
| `predict` | Forward: `x·w + b`. `!`-операции пишут граф | `def predict(x): return x @ w + b` |
| `train-loop` | Рекурсия-цикл: forward, loss, backward, шаг SGD | `for i in range(200): loss.backward(); opt.step()` |
| `(grad! loss)` | Backward по ленте | `loss.backward()` |
| `(sgd-step w lr)` | Обновление весов на месте | `w -= lr * w.grad` |
| `(start-trace)…(hlo-compile)` | Трассировка forward в граф и стенсил-компиляция | `compiled = torch.compile(lambda x: x @ w + b)` |
| `(hlo-run hlo ...)` | Нативный инференс | `compiled(x_test)` |

Обратите внимание: тензоры-константы записываются как `(tensor ((2.0)))` — скаляр в форме `[1,1]`, чтобы избежать сюрпризов broadcasting'а.

## 17.2 XOR: двухслойный MLP

**Задача.** Обучить сеть `2 → 4 (ReLU) → 1` решать XOR.

```lisp
;; ----- датасет -----
(defvar X (tensor ((0.0 0.0) (0.0 1.0) (1.0 0.0) (1.0 1.0))))
(defvar Y (tensor ((0.0) (1.0) (1.0) (0.0))))

;; ----- параметры -----
(defvar w1 (param (randn (list 2 4))))
(defvar b1 (param (zeros (list 4))))
(defvar w2 (param (randn (list 4 1))))
(defvar b2 (param (zeros (list 1))))
(defvar lr 0.5)

;; ----- forward -----
(defun forward (x)
  (let ((h (relu! (t+! (matmul! x w1) b1))))
    (matmul! h w2)))

;; ----- обучение -----
(defun train-epoch (iter max-iter)
  (if (> iter max-iter) nil
    (progn
      (let ((loss (mse! (forward X) Y)))
        (grad! loss)
        (sgd-step w1 lr) (sgd-step b1 lr)
        (sgd-step w2 lr) (sgd-step b2 lr)
        (if (equal (mod iter 100) 0) (progn (print "epoch") (print iter) (print loss)))
        (train-epoch (+ iter 1) max-iter)))))

(train-epoch 1 500)

;; ----- HLO-компиляция инференса -----
(start-trace)
(defvar dx (graph-param X))
(defvar w1p (graph-param w1))
(defvar b1p (graph-param b1))
(defvar w2p (graph-param w2))
(defvar dh (relu (t+ (matmul dx w1p) b1p)))
(defvar out (matmul dh w2p))
(defvar hlo (hlo-compile out))
(defvar hlo-pred (hlo-run hlo X w1 b1 w2))
```

### Python-эквивалент (PyTorch)

```python
import torch, torch.nn.functional as F

X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
Y = torch.tensor([[0.],[1.],[1.],[0.]])

w1 = torch.randn(2, 4, requires_grad=True)
b1 = torch.zeros(4, requires_grad=True)
w2 = torch.randn(4, 1, requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)
lr = 0.5

def forward(x):
    return (F.relu(x @ w1 + b1)) @ w2

for epoch in range(1, 501):
    loss = F.mse_loss(forward(X), Y)
    loss.backward()
    with torch.no_grad():
        w1 -= lr * w1.grad; b1 -= lr * b1.grad
        w2 -= lr * w2.grad; b2 -= lr * b2.grad
        w1.grad = b1.grad = w2.grad = b2.grad = None
    if epoch % 100 == 0:
        print(epoch, loss.item())
```

### Ключевые различия стилей

1. **Нет `optimizer.zero_grad()`** — лента Qualisp очищается в `GRAD!` автоматически (после backward).
2. **Нет `torch.no_grad()`** — разграничение train/inference-операций делается выбором `!`/без-`!` (`matmul!` vs `matmul`).
3. **Рекурсия вместо `for`** — идиоматично для Лиспа, `while` тоже доступен (через `while …`).
4. **HLO-инференс** — отдельный этап: трассировка → стенсил-компиляция (`compile_hlo_stencil`, W^X-страница, без LLVM) → нативный вызов. В PyTorch это `torch.compile`, в Qualisp — язык первого класса.

## 17.3 Идеи для упражнений

1. Добавьте Adam вместо SGD: `(adam-step w lr)` и сравните сходимость.
2. Замените MSE на `cross-entropy!` и добавьте `softmax!` — превратите регрессор в классификатор.
3. Оберните `predict` в `defuse!` и проверьте, что HLO соберёт всё в один fused-кирнел.
4. Прогоните датасет через `(cross-val-score ...)` из модуля ML (гл. 9).
5. Сохраните обученные веса через `save-npy w "w.npy"` и проверьте их в NumPy.

## 17.4 Расширения: нейросимвольный классификатор (NS-IF)

```lisp
;; Маршрутизация по символьному правилу: cat/dog
(defvar router-w (param (randn (list 4 2))))    ;; логиты cat/dog

(defvar router-output (matmul! x router-w))

(defvar pred (ns-if router-output
  (("cat") (matmul! x cat-net))
  (("dog") (matmul! x dog-net))))

(defvar loss (mse! pred y-true))
(ns-grad! loss 'cat)
```

`NS-IF` выбирает ветвь per-sample, `NS-GRAD!` направляет градиент в виновный сегмент (гл. 14).