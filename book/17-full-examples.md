# Глава 17. Разбор примеров

В этой главе — рабочие примеры из `examples/` и `bench/` репозитория 0.19.16, разобранные построчно, с Python-эквивалентами. Все они проверены бинарём `qualisp-0.19.16-linux-x86_64`: файлы запускаются как `qualisp examples/<file>.qlsp`, фрагменты — через REPL. Имена в Qualisp регистронезависимы и интернируются в верхний регистр (`MATMUL`, `PARAM`, `GRAD!`, `TSET!`), но книга пишет их «как в исходнике», в Lisp-идиоме.

Два контракта, которые видны в примерах и часто спотыкают новичков:

- **Статическая типизация обязательна.** `defun`/`defgraph` требуют парных аннотаций параметров и результата — `(x . :fixnum)` и `(expr . :fixnum)` (результат — **последняя форма тела**), `defvar` требует `(имя (значение . :тип))`: `(defvar w ((param …) . (:tensor :f32 (2 2))))`. У `lambda` аннотируются только параметры. Вывод типов локальный, `:any` запрещён; старый позиционный вид полностью удалён, совместимости нет. Исключение — `defmacro`, где параметры остаются позиционными `:syntax`-состояниями.
- **`print` принимает ровно один аргумент.** Чтобы напечатать несколько значений, склеивайте их сами: `(print (strcat "epoch " (to-string iter)))`. Файловый раннер не печатает результат последней формы.

Третий контракт — из устройства autograd: `grad!` заполняет `grad-of` для `param` и **возвращает `nil`**; веса обновляются на месте `tset!`, а не переприсваиванием результата шага оптимизатора (у `sgd-step`/`adam-step` wildcard-форма `(_ _)`).

## 17.1 Линейная регрессия: обучение + defgraph-инференс

**Задача.** Синтетические данные `y = 2x + 1 + noise`. Обучаем `w`, `b` через MSE + градиентный спуск по графу, затем вызываем отдельный `defgraph` для инференса.

### Полный код (проверен бинарём)

```lisp
;; ----- данные: 8 точек, форма (8 1) -----
(defvar xs ((tensor ((1.0) (2.0) (3.0) (4.0)
                     (5.0) (6.0) (7.0) (8.0))) . (:tensor :f32 (8 1))))
(defvar ys ((tensor ((3.1) (4.9) (7.2) (9.1)
                     (10.8) (13.2) (14.9) (17.1))) . (:tensor :f32 (8 1))))

;; ----- параметры: всё, что трассировщик видит как глобал, константа без градиента -----
(defvar w ((param (tensor ((0.0)))) . (:tensor :f32 (1 1))))
(defvar b ((param (tensor ((0.0)))) . (:tensor :f32 (1 1))))

;; ----- loss-граф: скаляр :flonum -----
(defgraph loss ((x . (:tensor :f32 (8 1))) (y . (:tensor :f32 (8 1)))
                (w . (:tensor :f32 (1 1))) (b . (:tensor :f32 (1 1))))
  ((let ((d (- (+ (matmul x w) b) y)))
     (tmean (* d d))) . :flonum))

;; ----- отдельный forward-граф для инференса -----
(defgraph predict ((x . (:tensor :f32 (8 1))) (w . (:tensor :f32 (1 1)))
                   (b . (:tensor :f32 (1 1))))
  ((+ (matmul x w) b) . (:tensor :f32 (8 1))))

;; ----- обучение: while + in-place tset! -----
(defvar lr (0.02 . :flonum))
(defvar epoch (0 . :fixnum))
(while (< epoch 200)
  (grad! (loss xs ys w b))
  (tset! w 0 0 (- (tref w 0 0) (* lr (tref (grad-of w) 0 0))))
  (tset! b 0 0 (- (tref b 0 0) (* lr (tref (grad-of b) 0 0))))
  (setq epoch (+ epoch 1)))

(print (loss xs ys w b))             ; → 0.023599…
(print (tref (predict xs w b) 0 0))  ; → 2.93799…
(print (tref (predict xs w b) 7 0))  ; → 17.0826…
```

### Построчный разбор

| Строки | Что происходит | Python |
|---|---|---|
| `xs`, `ys` | датасет 8×1: `y ≈ 2x + 1` | `x = torch.tensor([...]); y = 2*x + 1 + noise` |
| `w`, `b` | обучаемые параметры `[1,1]`, помечены `param` | `w = nn.Parameter(torch.zeros(1,1))` |
| `defgraph loss` | граф трассируется один раз, компилируется в native-код | `def loss(x,y,w,b): return ((x@w+b-y)**2).mean()` |
| `(grad! (loss …))` | forward + backward; `grad-of` заполнен, само вызов даёт `nil` | `loss_value.backward()` |
| `(tset! w 0 0 …)` | in-place шаг по `(grad-of w)` | `with torch.no_grad(): w -= lr*w.grad` |
| `defgraph predict` | отдельный граф без параметра-цели | `compiled = torch.compile(lambda x: x@w+b)` |
| `(predict xs w b)` | нативный инференс по графу | `compiled(xs)` |

Тензоры-константы записаны как `(tensor ((2.0)))` — скаляр в форме `[1,1]`: broadcast в 0.19.16 считается elementwise, и форма здесь фиксирована заранее. `tref`/`tset!` **не поддерживаются внутри `defgraph`** — они остаются в теле цикла над графом.

### Python-эквивалент (PyTorch)

```python
import torch

xs = torch.arange(1, 9, dtype=torch.float32).reshape(8, 1)
ys = torch.tensor([[3.1],[4.9],[7.2],[9.1],[10.8],[13.2],[14.9],[17.1]])
w = torch.nn.Parameter(torch.zeros(1, 1))
b = torch.nn.Parameter(torch.zeros(1, 1))
lr = 0.02

for epoch in range(200):
    d = xs @ w + b - ys
    loss = (d * d).mean()
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad = b.grad = None

print(((xs @ w + b - ys) ** 2).mean().item())
print((xs @ w + b)[0].item(), (xs @ w + b)[7].item())
```

> 🐍 Ключевое различие: в PyTorch `backward()` накапливает градиенты и требует `zero_grad()`; в Qualisp `grad!` работает с последним запуском графа в текущей top-level форме и перезаписывает `grad-of`.

## 17.2 XOR: двухслойный MLP

**Задача.** Обучить сеть `2 → 3 (tanh) → 1` решать XOR. Это дословный `examples/xor.qlsp`, проверенный бинарём (5000 эпох, full-batch SGD, `loss ≈ 2e-11`).

```lisp
; XOR 2-3-1 trained with full-batch SGD.
;
; The network is a defgraph: weights are *parameters*, because everything the
; tracer sees as a global is recorded as a constant and gets no gradient.
; `let` binds in parallel, so dependent bindings have to be nested.
;
; There is no in-place optimizer yet, so each update is an explicit `tset!`
; on the stable parameter tensor.

(defvar x   ((tensor ((0.0 0.0) (0.0 1.0) (1.0 0.0) (1.0 1.0))) . (:tensor :f32 (4 2))))
(defvar tgt ((tensor ((0.0) (1.0) (1.0) (0.0))) . (:tensor :f32 (4 1))))

(defvar w1 ((param (tensor ((0.5 -0.4 0.3) (-0.2 0.6 -0.5)))) . (:tensor :f32 (2 3))))
(defvar b1 ((param (tensor (0.1 -0.1 0.0))) . (:tensor :f32 (3))))
(defvar w2 ((param (tensor ((0.7) (-0.8) (0.9)))) . (:tensor :f32 (3 1))))
(defvar b2 ((param (tensor (0.0))) . (:tensor :f32 (1))))

(defgraph xor-loss ((x . (:tensor :f32 (4 2))) (w1 . (:tensor :f32 (2 3))) (b1 . (:tensor :f32 (3))) (w2 . (:tensor :f32 (3 1))) (b2 . (:tensor :f32 (1))) (tgt . (:tensor :f32 (4 1))))
  ((let ((h (tanh (+ (matmul x w1) b1))))
     (let ((p (+ (matmul h w2) b2)))
       (let ((d (- p tgt)))
         (tmean (* d d))))) . :flonum))

(defgraph xor-predict ((x . (:tensor :f32 (4 2))) (w1 . (:tensor :f32 (2 3))) (b1 . (:tensor :f32 (3))) (w2 . (:tensor :f32 (3 1))) (b2 . (:tensor :f32 (1))))
  ((+ (matmul (tanh (+ (matmul x w1) b1)) w2) b2) . (:tensor :f32 (4 1))))

(defvar lr (0.5 . :flonum))
(defvar epoch (0 . :fixnum))

(while (< epoch 5000)
  (grad! (xor-loss x w1 b1 w2 b2 tgt))
  (tset! w1 0 0 (- (tref w1 0 0) (* lr (tref (grad-of w1) 0 0))))
  (tset! w1 0 1 (- (tref w1 0 1) (* lr (tref (grad-of w1) 0 1))))
  (tset! w1 0 2 (- (tref w1 0 2) (* lr (tref (grad-of w1) 0 2))))
  (tset! w1 1 0 (- (tref w1 1 0) (* lr (tref (grad-of w1) 1 0))))
  (tset! w1 1 1 (- (tref w1 1 1) (* lr (tref (grad-of w1) 1 1))))
  (tset! w1 1 2 (- (tref w1 1 2) (* lr (tref (grad-of w1) 1 2))))
  (tset! b1 0 (- (tref b1 0) (* lr (tref (grad-of b1) 0))))
  (tset! b1 1 (- (tref b1 1) (* lr (tref (grad-of b1) 1))))
  (tset! b1 2 (- (tref b1 2) (* lr (tref (grad-of b1) 2))))
  (tset! w2 0 0 (- (tref w2 0 0) (* lr (tref (grad-of w2) 0 0))))
  (tset! w2 1 0 (- (tref w2 1 0) (* lr (tref (grad-of w2) 1 0))))
  (tset! w2 2 0 (- (tref w2 2 0) (* lr (tref (grad-of w2) 2 0))))
  (tset! b2 0 (- (tref b2 0) (* lr (tref (grad-of b2) 0))))
  (setq epoch (+ epoch 1)))

(print (xor-loss x w1 b1 w2 b2 tgt))

(defvar y ((xor-predict x w1 b1 w2 b2) . (:tensor :f32 (4 1))))
(print (tref y 0 0))
(print (tref y 1 0))
(print (tref y 2 0))
(print (tref y 3 0))

(print (tref w1 0 0))   ; далее — веса, как в examples/xor.qlsp
(print (tref w1 0 1))
(print (tref w1 0 2))
(print (tref w1 1 0))
(print (tref w1 1 1))
(print (tref w1 1 2))
(print (tref b1 0))
(print (tref b1 1))
(print (tref b1 2))
(print (tref w2 0 0))
(print (tref w2 1 0))
(print (tref w2 2 0))
(print (tref b2 0))
```

Вывод (проверен бинарём 0.19.16): `1.9630519432212168e-11` — loss; предсказания `4.35e-6`, `1.0000051`, `1.0000033`, `4.71e-6` — XOR выучен; дальше идут веса (`1.6469`, `-1.7026`, `0.1419`, …).

### Что здесь важно

| Приём | Почему так | Python |
|---|---|---|
| `param` на весах | трассировщик записывает глобалы как константы без градиента | `requires_grad=True` |
| вложенный `let` | `let` **параллельный**: `h` и `p` зависят друг от друга | последовательные присваивания |
| `defgraph` для loss и predict | граф трассируется один раз и компилируется; `tref`/`tset!` внутри графа запрещены | `torch.compile` для forward |
| `tset!` по `(grad-of w)` | результат `adam-step`/`sgd-step` имеет форму `(_ _)` и не присваивается конкретной `(2 1)` | in-place `w -= lr*w.grad` |
| `while` вместо рекурсии | счётчик в `defvar` + `setq` того же типа | `for epoch in range(5000)` |

### Python-эквивалент (PyTorch)

```python
import torch

x = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
tgt = torch.tensor([[0.],[1.],[1.],[0.]])
w1 = torch.nn.Parameter(torch.tensor([[0.5,-0.4,0.3],[-0.2,0.6,-0.5]]))
b1 = torch.nn.Parameter(torch.tensor([0.1,-0.1,0.0]))
w2 = torch.nn.Parameter(torch.tensor([[0.7],[-0.8],[0.9]]))
b2 = torch.nn.Parameter(torch.tensor([0.0]))
lr = 0.5

def loss_fn():
    h = torch.tanh(x @ w1 + b1)
    return ((h @ w2 + b2 - tgt) ** 2).mean()

for epoch in range(5000):
    loss = loss_fn()
    loss.backward()
    with torch.no_grad():
        for p in (w1, b1, w2, b2):
            p -= lr * p.grad
            p.grad = None

print(loss_fn().item())
print((torch.tanh(x @ w1 + b1) @ w2 + b2).flatten().tolist())
```

## 17.3 Идеи для упражнений

1. **Adam вместо SGD.** Модуль `dl` даёт `(adam-init w)`, `(adam-step st grad lr)`, `(adam-param st)`, `(adam-steps st)`; состояние — структура `adam-state`. В графе обновление по-прежнему in-place через `grad-of`.
2. **Другая потеря.** `(mse-loss yhat y)` из `dl` или `(t-mse pred actual)` из `ml` для 1-D. Softmax и cross-entropy как готовых примитивов нет — соберите их из `exp`/`tsum`/`log`.
3. **defuse.** Явный fused-кернел собирается в один x86-64 цикл:

   ```lisp
   (defuse affine ((x . (:tensor :f32 (4))) (a . :flonum) (b . :flonum))
     ((relu (+ (* x a) b)) . (:tensor :f32 (4))))

   (affine (tensor (1.0 -1.0 2.0 -2.0)) 2.0 1.0)
   ```

   Тело — одно elementwise-выражение (`+ - * /`, унарный `-`, `relu`, числа, параметры); первого eager-вызова достаточно для компиляции. Внутри `defgraph` вызов `defuse` инлайнится в `let` перед трассировкой, так что forward/backward идут обычными HLO-узлами.

4. **Чекпоинт весов.** `(save-weights path [state])` и `(load-weights path)` — секция `model` QSRD; состояние возвращается как `:syntax`:

   ```lisp
   (save-weights "/tmp/ckpt.qsrd" (list 7 8))   ; 7 — эпоха, 8 — seed
   (write-to-string (load-weights "/tmp/ckpt.qsrd"))  ; → "(7 8)"
   ```

5. **Сверка аналитического градиента.** Возьмите `symbolic/diff` и `neuro/term` — как в `examples/term.qlsp` ниже.

6. **Tablet вместо ручного среза данных.** `(import tablet)`, `(make-tablet names data)`, методы `where`/`sort-by`/`head` (гл. 15.10).

## 17.4 Расширения: NS-IF и бенчмарки

NS-логика — не булевы предикаты, а непрерывные level-функции, знак которых равен истинности. Примитивы `ns-grt`/`ns-geq`/`ns-lss`/`ns-leq`/`ns-eq`/`ns-neq`/`ns-and`/`ns-or`/`ns-not` возвращают `:flonum`. `ns-if` — per-sample hard-роутер `[B,K]`: guard'ы покрывают ячейки разбиения ровно один раз, одна общая boundary-ветка по `|atom_j| < eps`, никакого blending.

### ns-if в скомпилированном графе (уменьшенный вариант bench/nsif_graph.qlsp, проверен)

```lisp
(defgraph route ((p . (:tensor :f32 (16384 8))) (a0 . (:tensor :f32 (16384)))
                 (a1 . (:tensor :f32 (16384))) (a2 . (:tensor :f32 (16384))))
  ((ns-if (:payload p) (:eps 0.02)
    ((ns-grt a0 0) (lambda ((v . (:tensor :f32 (16384 8)))) (* v v)))
    ((ns-and (ns-lss a0 0) (ns-grt a1 0))
     (lambda ((v . (:tensor :f32 (16384 8)))) (* v 3.0)))
    ((ns-and (ns-lss a0 0) (ns-lss a1 0) (ns-grt a2 0))
     (lambda ((v . (:tensor :f32 (16384 8)))) (* v 5.0)))
    ((ns-and (ns-lss a0 0) (ns-lss a1 0) (ns-lss a2 0))
     (lambda ((v . (:tensor :f32 (16384 8)))) (* v 7.0)))
    (else (lambda ((v . (:tensor :f32 (16384 8)))) (* v 11.0)))) . (:tensor :f32 (16384 8))))

(defvar pp ((param (randn :f32 (16384 8))) . (:tensor :f32 (16384 8))))
(defvar a0 ((randn :f32 (16384)) . (:tensor :f32 (16384))))
(defvar a1 ((randn :f32 (16384)) . (:tensor :f32 (16384))))
(defvar a2 ((randn :f32 (16384)) . (:tensor :f32 (16384))))

(print (tsum (route pp a0 a1 a2)))
(grad! (route pp a0 a1 a2))
(print (tsum (grad-of pp)))
```

В графе `ns-if` трассируется в узел с подграфами ветвей; исполняются только непустые ветки. Backward идёт **только по исполненной ветке**, градиента по решению роутера не существует (`∂H/∂atom = 0` вне поверхности).

Eager-вариант из `bench/nsif_eager.qlsp` отличается только тем, что `ns-if` стоит в обычном `defun` над `(payload x)`; он тоже проверен бинарём.

### Остальные бенчмарки

| Файл | Нагрузка | Что показывает |
|---|---|---|
| `bench/fib.qlsp` | `fib(27)` × 3 | путь вызова/возврата, TCO не нужен; вывод `589254` |
| `bench/reduce.qlsp` | 200 × `tsum` по `randn :f32 (16384 8)` | редукции и TensorBufferPool |
| `bench/matmul.qlsp` | 100 × `matmul` 128×128 | i-k-j + AVX2, пул компиляции графа |
| `bench/nsif_eager.qlsp` | 65536×8, 4 ведра + boundary, 20 проходов | eager-роутер |
| `bench/nsif_graph.qlsp` | 16384×8, forward + backward в графе | `Op.ns_if` и backward по исполненной ветке |

В 0.19.16 часть ядер идёт через `kernel_pool` — отдельный пул вычислительных потоков, не путать с пулом компиляции HLO. Потоки создаются лениво на первом достаточно большом вызове (`max(1, min(cpu-1, 8))`), у воркеров нет `Runtime`: вызывающий выделяет буферы и раздаёт непересекающиеся диапазоны. Параллелятся `elementwise`/`map`, `matmul` (по строкам результата), `reshape`/`transpose`, `tsum`/`tmean` (8 полос с фиксированным порядком сведения — результат побайтово совпадает с последовательным), а в NS — `fillBand`, выбор ветки, counting-sort, `gather`/`scatter`. `max`, `softmax`, `argmax` и `reduceToShape` остаются последовательными. Переменная `QUALISP_SERIAL=1` отключает пул полностью.

Замер A/B (ReleaseFast, 16 ядер, min из 11): matmul 15.9 → 7.6 мс (**2.10×**), ns-if eager 84.9 → 79.0 мс, graph 59.9 → 60.4 мс, reduce 5.1 → 5.7 мс, fib без изменений. Остаток одной нити в ns-if — сами branch-лямбды (4 ветки). Тесты `src/parallel_test.zig` сверяют parallel/serial побайтово.

> 🐍 `ns-if` ≈ `torch.where`, но с жёстким выбором ветви на строку и без смешивания: это скорее `np.select` с проверкой полноты покрытия (ROBDD), у которого есть настоящий backward по выбранной ветке.

## 17.5 examples/args.qlsp — аргументы командной строки

```lisp
;; The program's own command line: everything after the script path.
;;
;;   qualisp examples/args.qlsp --verbose --name lena input.txt
(import argparse)

(print (command-line-args))
(print (flag? (command-line-args) "verbose"))
(print (option-value (command-line-args) "name"))
(print (positionals (command-line-args) (list "name")))
```

Проверенный вывод:

```text
$ qualisp examples/args.qlsp --verbose --name lena input.txt
(examples/args.qlsp --verbose --name lena input.txt)
t
lena
(input.txt)
```

`(command-line-args)` всегда непуст: первый элемент — путь скрипта (argv[0]). Имя опции передаётся без дефисов; `positionals` требует явного списка опций со значениями — иначе `--name lena` был бы принят за позиционный аргумент.

> 🐍 `(command-line-args)` ≈ `sys.argv`; `flag?`/`option-value`/`positionals` — аккуратные хелперы поверх него, а не `argparse.ArgumentParser` с автогенерацией `--help`.

## 17.6 examples/term.qlsp — нейросимволический мост

Терм — данные: `(term '(...) '(c x y))` раскрывается в код над тензорами. Одно выражение служит трём целям: обучаемому графу, eager-функции и сверке аналитического `diff` с autograd. Проверено бинарём (200 эпох, `c` сходится к `2.0`).

```lisp
(import neuro)
(import symbolic)

(defvar xs ((tensor (0.5 1.0 1.5 2.0)) . (:tensor :f32 (4))))
(defvar ys ((tensor (0.5 2.0 4.5 8.0)) . (:tensor :f32 (4))))

;; `param` — тензор, для которого `grad!` заполнит `grad-of`.
(defvar c ((param (tensor (0.5))) . (:tensor :f32 (1))))
(defvar lr (0.05 . :flonum))

;; Потеря — тот же терм, что и модель: (c*x^2 - y)^2, среднее по точкам.
(defgraph loss ((c . (:tensor :f32 (1))) (x . (:tensor :f32 (4))) (y . (:tensor :f32 (4))))
  ((term '(tmean (* (- (* c (* x x)) y) (- (* c (* x x)) y))) '(c x y)) . :flonum))

(defvar epoch (0 . :fixnum))
(while (< epoch 200)
  (grad! (loss c xs ys))
  (tset! c 0 (- (tref c 0) (* lr (tref (grad-of c) 0))))
  (setq epoch (+ epoch 1)))

(print (tref c 0)) ; 2.0

;; Аналитический градиент c*x^2 по c — это x^2; `diff` строит его как терм.
(print (diff '(* c (* x x)) 'c)) ; (* X X)

;; Тот же градиент двумя путями: autograd и скомпилированный терм.
(defgraph sum-sq ((c . (:tensor :f32 (1))) (x . (:tensor :f32 (4))))
  ((term '(tsum (* c (* x x))) '(c x)) . :flonum))
(defgraph slope-sq ((x . (:tensor :f32 (4))))
  ((term '(tsum (* x x)) '(x)) . :flonum))

(grad! (sum-sq c xs))
(print (tref (grad-of c) 0)) ; 7.5
(print (slope-sq xs))        ; 7.5
```

Вывод (проверен бинарём 0.19.16): `2.0`, `(* X X)`, `7.5`, `7.5` — аналитический градиент и autograd совпадают.

Терм умеет и новое: `term->code` транслирует `reshape`/`reshape-like`/`transpose`/`ones-like`/`tmax-mask`/`relu-mask`/`softmax`, а `^ n` над составным основанием связывает подтерм один раз через `let`:

```lisp
(import neuro)
(import symbolic)
(grad '(tsum x) 'x)                      ; → (ONES-LIKE X)
(grad '(tsum (relu x)) 'x)               ; → (* (ONES-LIKE (RELU X)) (RELU-MASK X))
(write-to-string (term->code '(^ (+ x 1) 3) '(x)))
;; → "(LET ((POW-0 (+ X 1))) (* POW-0 POW-0 POW-0))"
```

`grad` даёт градиент скалярного лосса `(tsum term)` — в отличие от поэлементного `diff`; результат проходит `simplify`, но маски и `ones-like` остаются отдельными операциями.

| Приём | Что происходит | Python |
|---|---|---|
| `(term '(…) '(c x y))` | макрос раскрывает терм в код над тензорами до типизации | `sympy.lambdify(expr, ...)` |
| `defgraph` с термом | `defgraph` раскрывает макросы в теле и трассирует результат | `torch.compile(lambdify(...))` |
| `diff` → `term` | символьная производная становится тензорной функцией | `sympy.diff` + `lambdify` |
| `grad!`/`grad-of` | автодифференцирование графа | `loss.backward()` + `.grad` |

> 🐍 Это мост `sympy → torch` без текстогенерации на стороне пользователя: терм остаётся S-выражением, а раскрытие происходит на этапе макроexpand, до проверки типов.

## 17.7 examples/error-demo.qlsp — структурированные ошибки

Демо показывает сообщение, категорию, причину и трейс пойманной ошибки. `error` — prelude, поэтому `(import error)` здесь не нужен. Файл дословно из `examples/error-demo.qlsp` (проверен бинарём 0.19.16):

```lisp
; Демо структурированных ошибок и трассировки.
;
;   ./zig-out/bin/qualisp examples/error-demo.qlsp
;
; Хвостовые вызовы сворачиваются в один кадр (см. P5v), поэтому middle/outer
; ниже намеренно не хвостовые — так в трейсе видно несколько кадров.
; Закомментируй catch-error и вызови (outer) напрямую, чтобы увидеть
; позицию ошибки в terminal: file:line:col: boom.

(defun inner ((x . :fixnum))
  ((throw 42) . :never))
(defun middle ((x . :fixnum))
  ((let ((r (inner x))) r) . :never))
(defun outer ()
  ((let ((r (middle 1))) r) . :never))

(defvar e ((result-error (catch-error (lambda () (outer)))) . (:struct error)))

(print "=== сообщение ===")
(print (error-message e))
(print "=== категория ===")
(print (error-kind e))
(print "=== причина (cause) ===")
(print (error-cause e))
(print "=== трейс ===")
(print (format-error e))
```

Вывод (проверен: `qualisp examples/error-demo.qlsp`, exit 0):

```text
=== сообщение ===
42
=== категория ===
USER
=== причина (cause) ===
42
=== трейс ===
Error: 42 (user)
  at OUTER
  at MIDDLE  examples/error-demo.qlsp:15  in (MIDDLE 1)
  at INNER  examples/error-demo.qlsp:13  in (INNER X)
```

`(throw 42)` даёт `kind = 'user`, `message = "42"` и `cause = 42`; `format-error` печатает кадры от oldest к newest, а тег ошибки `ERROR::ERROR` в отчёт не попадает. У кадра `OUTER` нет места в source map, поэтому он напечатан без `<unknown>:0` — только имя. Если убрать `catch-error` и вызвать `(outer)` напрямую, раннер покажет позицию броска в формате `file:line:col: 42` (в пробе — `unhandled.qlsp:2:31: 42`) и `error: RuntimeError` с exit 1.

## 17.8 examples/lsp-demo.qlsp — демо для редактора

Демо для VSCodium/VS Code: оно не печатает ничего интересного (`12`), зато показывает навигацию и диагностику. Файл дословно из `examples/lsp-demo.qlsp`:

```lisp
; Демо LSP. Откройте файл в VSCodium после `bash tools/lsp/build.sh`.
;
;  1. Наведите курсор на add-pair в последней форме — hover покажет тип.
;  2. Ctrl+Click (F12) по add-pair — переход к определению.
;  3. Откройте панель Outline — видны defun и defvar.
;  4. Замените (x . :fixnum) на (x :fixnum) — появится красная диагностика,
;     потому что старый позиционный вид удалён полностью.

(defun add-pair ((x . :fixnum) (y . :fixnum))
  ((+ x y) . :fixnum))

(defvar limit (10 . :fixnum))

(print (add-pair 2 limit))
```

Проверенный вывод раннера: `12`. hover показывает статический тип и исходную форму определения с комментарием над ней, definition по `defun`/`defvar` работает в пределах файла и по зарегистрированным `.qlsp` воркспейса, Outline строится по верхним формам, completion включает локальные имена и экспорты stdlib. Замена `(x . :fixnum)` на `(x :fixnum)` даёт диагностику `defun expects parameters, a result and a body`.

Сервер и расширение собираются **локально**: `bash tools/lsp/build.sh`, затем установка `python3 editors/vscode/package-vsix.py` и `codium --install-extension zig-out/qualisp.vsix --force`. Готового `qualisp-lsp` и VSIX в релизных ассетах 0.19.16 нет (гл. 16.7).
