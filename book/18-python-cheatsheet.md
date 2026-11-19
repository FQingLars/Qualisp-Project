# Глава 18. Шпаргалка: Qualisp ↔ Python

Быстрый перевод между Qualisp и Python-стеком (NumPy / PyTorch / scikit-learn) применительно к релизу 0.19.15 (реализация на Zig). Имена примитивов в Qualisp интернируются в верхний регистр (`MATMUL`, `TSET!`, `PARAM`, `GRAD!`, `NS-IF`); писать их можно в любом регистре. Типы обязательны: `defun`/`lambda` требуют аннотаций параметров и возврата, `defvar` — типа. Ламбда-литерал не вызывается напрямую — только через `funcall`.

## 18.0 Гомоиконность

| Qualisp | Python |
|---|---|
| `(read-from-string "(+ 1 2)")` | `ast.parse("1 + 2")` — текст → код как данные (`:syntax`) |
| `(write-to-string expr)` | `repr(expr)` / `ast.unparse(...)` — читаемая форма, round-trip для данных |
| `(eval 'form)` | `eval(...)` — но над S-выражением; `:syntax → :a`, тип выводится из контекста |
| `(quote x)` / `'x` | данные, а не вызов; `` `(f ,x) `` — квазиквота |
| `(macroexpand-1 form)` / `(macroexpand form)` / `(macroexpand-all form)` | раскрытие макросов до проверки типов; `macroexpand-all` — рекурсивно, не заходя в `quote` |
| `(gensym "fd")` | свежее имя для гигиены макроса (`G-0`, `G-1`, …) |
| `(defgraph linear (x (:tensor :f32 (2 3))) (:tensor :f32 (2 3)) body)` | мастер-граф как форма: близко к `torch.fx.GraphModule`, но исходник — S-выражение |
| `(defuse affine (x (:tensor :f32 (4)) a :flonum b :flonum) (:tensor :f32 (4)) body)` | `@torch.compile` над одним elementwise-выражением: один цикл без промежуточных тензоров |
| `(save-image "w.qimg")` / `(load-image "w.qimg")` | `pickle` + `torch.save` в одном файле QSRD (гл. 21) |
| `(save-weights "w.qsrd")` / `(load-weights "w.qsrd")` | `torch.save(model.state_dict())` / `load_state_dict` |
| `(import ml)` / `(import-from ml (mean mse))` | `import ml` / `from ml import mean, mse` |
| `(defvar w (:tensor :f32 (2)) (param (tensor (1.0 2.0))))` | `w = torch.tensor([1., 2.], requires_grad=True)` |
| `qualisp --trace tr.json script.qlsp` + `qprof tr.json` | `cProfile` + `pstats` — внешний инструмент, не форма языка |

> 🐍 В Qualisp нет отдельного AST: код и данные — одни и те же cons-ячейки. `:syntax` — не `:any`, а полноценный тип: с ним работают `car`/`cdr`/`cons`/`equal`/печать/`eval`.

## 18.1 Базовый синтаксис

| Qualisp | Python |
|---|---|
| `(+ 1 2)` | `1 + 2` |
| `(f x y)` | `f(x, y)` |
| `(defun sq (x :fixnum) :fixnum (* x x))` | `def sq(x: int) -> int: return x * x` |
| `(\ (x :fixnum) :fixnum (* x x))` | `lambda x: x * x` — `\` — сахар для `lambda` |
| `(defvar y :fixnum 20)` | `y: int = 20` — тип фиксируется |
| `(setq y 21)` | `y = 21` — только значение того же типа |
| `(let ((a 1) (b 2)) (+ a b))` | `a, b = 1, 2` — **параллельные** биндинги; `let*` отсутствует |
| `(if (null x) a b)` | `a if x is None else b` — `if` принимает только `:bool`; `(if x …)` — ошибка |
| `(cond (c1 r1) (c2 r2) (t r))` | `if / elif / else` |
| `(while c body)` | `while c:` |
| `(progn a b c)` | тело функции / блок |
| `(and a b)` / `(or a b)` | `a and b` / `a or b` — булевы короткозамкнутые формы, возвращают `t`/`nil` |
| `; комментарий` | `# комментарий` |
| `'(1 2 3)` | `[1, 2, 3]` (литерал данных) |
| `(= 0 0.0)` | `0 == 0.0` → `t` — числа сравниваются по значению |
| `(equal a b)` | `a == b` — cons/строки по содержимому; тензоры и функции — по идентичности (`is`) |
| `(eq 'a 'a)` | `a is a` — интернированные символы, один id |
| `:f32`, `:linux` | литералы вроде `enum.Enum`: keyword-символы самовычисляемы |
| `(print (to-string 42))` | `print(str(42))` — `print` и `display` принимают **ровно один** аргумент |
| `(type-of 42)` | `type(42).__name__` → `INT` (в Qualisp — верхний регистр) |

Числа: `42`, `-7`, `3.14`, `1e3` → `1000.0`; `/` всегда даёт float (`(/ 7 2)` → `3.5`). `0xFF` и `1/2` числами не являются. `sqrt` есть; `abs`, `min`, `max`, `expt` — нет. `nil` — false, «None» и пустой список одновременно; других ложных значений нет, `0` истинен. `t` и `nil` — булевы константы.

Проверка типов — статическая: `(defun sq (x :fixnum) :fixnum (* x x))` печатает `SQ`, `(sq 7)` → `49`. Несовпадение типа — отказ с позицией `error at 1:1: …`, а не runtime-значение.

### 18.1.1 Структуры, методы и трейты

| Qualisp | Python |
|---|---|
| `(defstruct pt ((x :fixnum) (y :fixnum)))` | `@dataclass` с аннотированными полями |
| `(pt 1 2)` | `pt(1, 2)` — конструктор-функция |
| `(slot-value p 'x)` / `(set-slot p 'x 10)` | `p.x` / `p.x = v` — статический доступ; неизвестное поле — ошибка компиляции |
| `(type-of p)` → `STRUCT` | `type(p).__name__` |
| `(deftrait display (show (self :self) :string))` | `typing.Protocol` — набор сигнатур без тел |
| `(defstruct box ((v :fixnum)) (display))` | класс, объявляющий реализацию протокола |
| `(impl box (defun show (self) :string …))` | `def show(self) -> str: …` в теле класса |
| `(show obj)` | `obj.show()` — диспетч статически по типу первого аргумента, без vtable |

Структура — **атом** с тегом и типизированными слотами, а не S-выражение; гомоиконность относится к коду. Трейт — не значение. Метод можно объявить и через `impl` без трейта.

## 18.2 Списки

| Qualisp | Python |
|---|---|
| `(car xs)` / `(cdr xs)` | `xs[0]` / `xs[1:]` — `cdr` возвращает `(:option (:list T))` |
| `(cons x xs)` | `[x] + xs` |
| `(nth xs 1)` | `xs[1]` — **список первым, индекс вторым** |
| `(length xs)` | `len(xs)` |
| `(null xs)` | `xs is None` — `null : :a → :bool` |
| `(list 1 2 3)` | `[1, 2, 3]` |
| `(map (lambda (x :fixnum) :fixnum (* x x)) xs)` | `list(map(lambda x: x*x, xs))` |
| `(filter (lambda (x :fixnum) :bool (> x 1)) xs)` | `list(filter(..., xs))` |
| `(reduce (lambda (a :fixnum b :fixnum) :fixnum (+ a b)) xs 0)` | `functools.reduce(lambda a, b: a+b, xs, 0)` |
| `(append xs ys)` / `(reverse xs)` | `xs + ys` / `xs[::-1]` |

`reduce` принимает `(элемент аккумулятор)`; без init первым аккумулятором становится первый элемент. `and`/`or` — не функции-значения, а спецформы; `(apply and xs)` не работает.

### 18.2.1 Коллекции: утилиты списков (модуль `collections`)

| Qualisp | Python |
|---|---|
| `(sum xs)` | `sum(xs)` |
| `(range 5)` | `range(5)` — только один аргумент, результат — option-список |
| `(any p xs)` / `(all p xs)` | `any(map(p, xs))` / `all(map(p, xs))` — предикат первым |
| `(count-if p xs)` | `sum(1 for x in xs if p(x))` |
| `(find p xs)` / `(position p xs)` | `next(x for x in xs if p(x))` / `next(i for i, x in enumerate(xs) if p(x))` — option |
| `(take-while p xs)` / `(drop-while p xs)` | `itertools.takewhile(p, xs)` / `dropwhile(p, xs)` |
| `(zip-with f xs ys)` | `[f(a, b) for a, b in zip(xs, ys)]` |
| `(member? x xs)` | `x in xs` — элемент первым |
| `(unique xs)` | `list(dict.fromkeys(xs))` — порядок и первый элемент сохраняются |
| `(last xs)` / `(keep-positive xs)` | `xs[-1]` / `[x for x in xs if x > 0]` |

`find`/`position`/`take-while`/`drop-while`/`zip-with` рекурсируют по индексу: `cdr` даёт option, поэтому проход «по хвосту» требует распаковки. `keep-positive` возвращает `(:option (:list :fixnum))`.

## 18.3 Тензоры (NumPy)

| Qualisp | NumPy |
|---|---|
| `(tensor ((1.0 2.0) (3.0 4.0)))` | `np.array([[1., 2.], [3., 4.]])` — литерал всегда `:f32` |
| `(zeros :f32 (2 3))` | `np.zeros((2, 3), dtype=np.float32)` |
| `(ones :f32 (3))` | `np.ones(3, dtype=np.float32)` |
| `(randn :f32 (2 2))` | `np.random.randn(2, 2)` — xorshift64*, не криптостойкий |
| `(+ a b)`, `(- a b)`, `(* a b)`, `(/ a b)` | `a + b`, `a - b`, `a * b`, `a / b` — elementwise + broadcast, dtypes должны совпадать |
| `(matmul a b)` | `a @ b` — только 2-D |
| `(reshape t (2 2))` | `t.reshape(2, 2)` — форма — литеральный список целых; копия |
| `(transpose t)` | `t.T` — только 2-D; копия |
| `(tsum t)` / `(tmean t)` / `(tmax t)` | `t.sum()` / `t.mean()` / `t.max()` — оси нет, свёртка в скаляр |
| `(argmax t)` | `t.argmax()` — только 1-D |
| `(relu t)` / `(exp t)` / `(log t)` / `(tanh t)` | `np.maximum(t, 0)` / `np.exp(t)` / … — elementwise |
| `(softmax t)` | `scipy.special.softmax(t)` — только 1-D |
| `(tref t i j)` | `t[i, j]` — индексы `:fixnum`, их число равно рангу |
| `(tset! t i j v)` | `t[i, j] = v` — мутация на месте, возвращает тот же тензор |
| `(t-shape t)` | `t.shape` — список измерений |
| `(t-column t i)` | `t[:, i]` — копия |
| `(t-rows t start count)` | `t[start:start+count]` — копия |
| `(t-select-rows t ids)` | `t[ids]` — копия строк по индексам |

> ⚠️ `tensor-shape`, `tensor-dtype`, `t+`, `t-`, `t*`, `t/`, `sigmoid`, `elementwise` как имена отсутствуют: elementwise делают обычные `+ - * /`, а `sigmoid` можно собрать Lisp-выражением. `zeros`/`ones`/`randn` требуют dtype-ключ и **литеральную** форму `(2 3)`; `(list 2 3)` и `'(2 2)` не подходят. Печать тензора — `#<tensor>`, `type-of` → `TENSOR`.

### Выборки и копии

Выборки всегда материализуются: `reshape`, `transpose`, `t-rows`, `t-select-rows`, `t-column` возвращают **новый** тензор, алиасов над буфером и записи через них нет. `t-slice` в API отсутствует. Изменить элемент можно только через `tset!` на исходном тензоре.

```lisp
(defvar m (:tensor :f32 (2 2)) (tensor ((1.0 2.0) (3.0 4.0))))
(print (tref m 1 0))          ;; 3.0
(tset! m 1 0 9.0)
(print (tref m 1 0))          ;; 9.0
(print (t-rows m 1 1))        ;; #<tensor> — копия строки
(print (matmul m (transpose m)))
```

> 🐍 В NumPy срез `a[1:2]` — алиас, и запись в него видна в исходнике. Здесь `t-rows` — копия: изменение исходника не повлияет на результат, и наоборот. Это то же правило, что у `reshape`/`transpose`.

## 18.4 Автоград (PyTorch)

| Qualisp | PyTorch |
|---|---|
| `(defvar w (:tensor :f32 (2)) (param (tensor (1.0 2.0))))` | `w = torch.tensor([1., 2.], requires_grad=True)` |
| `(defgraph loss (x (:tensor :f32 (2))) :flonum (tmean (* x x)))` | `def loss(x): return (x*x).mean()` — граф трассируется один раз |
| `(grad! (loss w))` | `loss(w).backward()` |
| `(grad-of w)` | `w.grad` — stable-градиент той же формы/dtype; для невычисленного параметра — нули |
| `(tset! w 0 (- (tref w 0) (* lr (tref (grad-of w) 0))))` | `w.data[0] -= lr * w.grad[0]` — in-place обновление |
| `(relu t)` / `(tanh t)` / `(exp t)` в теле графа | `F.relu` / `torch.tanh` / `torch.exp` — узлы backward есть |
| `(linear-forward x w b)` | `torch.nn.functional.linear(x, w, b)` |
| `(mlp-forward x w1 b1 w2 b2)` | `nn.Sequential(nn.Linear(...), nn.ReLU(), nn.Linear(...))` |
| `(mse-loss pred y)` | `F.mse_loss(pred, y)` — скаляр |
| `(sgd-step w grad lr)` | `w - lr * grad` — чистая функция, возвращает новый тензор |
| `(adam-init w)` / `(adam-step st grad lr)` / `(adam-param st)` | `torch.optim.Adam`: состояние в `defstruct adam-state` (m, v, step) |
| `(save-weights "w.qsrd")` / `(load-weights "w.qsrd")` | `torch.save(state_dict)` / `load_state_dict` |

### Граф, запуск и состояние

`defgraph` — master-форма. Тело исполняется один раз в trace-режиме на нулевых placeholder-тензорах; управление статическое, `tref`/`tset!` в графе не поддерживаются. Вызов компилируется в запуск графа; `grad!` сеет единицы формы выхода и идёт по узлам с конца, читая forward-активации из per-run `TensorBufferPool` — **без replay**, за O(n). Градиенты накапливаются в scratch, для `param`-входов копируются в stable и кладутся в `grad`.

```lisp
(defvar w (:tensor :f32 (2 2)) (param (tensor ((1.0 2.0) (3.0 4.0)))))
(defgraph sq (x (:tensor :f32 (2 2))) :flonum (tmean (* x x)))
(print (sq w))                    ;; 7.5
(grad! (sq w))
(print (tref (grad-of w) 0 0))    ;; 0.5
```

> ⚠️ `grad!` работает только с **последним запуском графа в текущей top-level форме** и не проверяет, что аргумент — именно его выход. Отдельного `tape-form`, `grad-accum!`, `zero-grad!`, `clip-grad!`, `sigmoid!` и `!`-суффиксов операций нет. Полный обучающий цикл — в `examples/xor.qlsp`: `grad!` → `tset!` по `grad-of` → следующая итерация.

## 18.5 Слои (DL-модель ↔ torch.nn)

| Qualisp (`dl`) | PyTorch |
|---|---|
| `(linear-forward x w b)` | `F.linear(x, w, b)` — логиты, активация — выбор вызывающего |
| `(mlp-forward x w1 b1 w2 b2)` | `nn.Sequential(Linear, ReLU, Linear)` |
| `(mse-loss yhat y)` / `(squared-error pred actual)` | `F.mse_loss` / `nn.MSELoss` на скалярах |
| `(sgd-step w grad lr)` | шаг SGD — чистая замена биндинга |
| `(adam-init w)` / `(adam-step st grad lr)` / `(adam-param st)` / `(adam-steps st)` | `torch.optim.Adam` + `state_dict` |
| `(momentum-init w)` / `(momentum-step st grad lr momentum)` / `(momentum-param st)` | SGD с momentum |
| `(linear-grads x y w b)` / `(linear-regression-step x y w b lr)` | замкнутая форма градиента линейной регрессии — без графа |
| `(defgraph net (x …) … (relu (+ (matmul x w) b)))` | `nn.Module.forward` как S-выражение |

Bias — строка `(1 m)`, broadcast `(n m) + (1 m)` даёт per-column bias. Все функции `dl` чистые, работают только с `:f32` и 2-D `batch × features`; результат шага имеет wildcard-форму `(_ _)` и не присваивается переменной с конкретной формой. Слои `batchnorm`/`dropout`/`attention`/`conv2d`/`maxpool` как готовые API отсутствуют — они выражаются композицией `defgraph`/`defuse`/`neuro`.

### 18.5.1 Строки (модули `string` и глобальные примитивы ↔ `str`)

| Qualisp | Python |
|---|---|
| `(string-length s)` | `len(s)` — индексы — кодовые точки |
| `(string-split "a,b,c" ",")` | `"a,b,c".split(",")` — **строка первым, разделитель вторым** |
| `(string-join (list "a" "b") ",")` | `",".join(["a", "b"])` |
| `(string-upcase s)` / `(string-downcase s)` | `s.upper()` / `s.lower()` |
| `(string-trim s)` / `(string-replace s "b" "X")` | `s.strip()` / `s.replace("b", "X")` |
| `(string-index s "b")` | `s.find("b")` |
| `(strcat a b)` / `(substr s i n)` | `a + b` / `s[i:i+n]` |
| `(char-code "A")` → `65` / `(code-char 65)` → `"A"` | `ord("A")` / `chr(65)` |
| `(string-contains s "sub")` / `(string-starts-with s "pre")` / `(string-ends-with s "suf")` | `in` / `s.startswith` / `s.endswith` — экспорты `string` |
| `(string-lines s)` / `(string-words s)` | `s.splitlines()` / `s.split()` — option-список |
| `(string-repeat s 3)` / `(string-capitalize s)` / `(string-to-fixnum "42")` | `s * 3` / `s.capitalize()` / `int("42")` — экспорты `string` |
| `(string-blank? s)` / `(string-pad-left s n c)` | `not s.strip()` / `s.rjust(n, c)` — экспорты `string` |
| `(to-string 42)` / `(number-to-string 42)` / `(string-to-number "3.14")` | `str(42)` / `str(42)` / `float("3.14")` |
| `(write-to-string '(1 "a"))` | `repr([1, "a"])` → `"(1 \"a\")"` |

`string-length`, `string-index`, `string-split`, `string-join`, `string-upcase`, `string-downcase`, `string-trim`, `string-replace` — глобальные примитивы; функции из таблицы «экспорты string» требуют `(import string)`. `string-repeat` удваивает блоки (работа пропорциональна результату), `string-lines`/`string-words` возвращают option-список.

## 18.6 Классический ML (sklearn)

| Qualisp (`ml`) | scikit-learn |
|---|---|
| `(mean xs)` / `(mse pred actual)` / `(mae pred actual)` / `(rmse pred actual)` | `np.mean` / `mean_squared_error` / `mean_absolute_error` / `root_mean_squared_error` |
| `(accuracy pred actual)` | `accuracy_score` |
| `(precision pred actual)` / `(recall pred actual)` / `(f1 pred actual)` | `precision_score` / `recall_score` / `f1_score` — бинарные метки 1/0 |
| `(train-part xs ratio)` / `(test-part xs ratio)` | `train_test_split` — option-список, пустая часть — `nil` |
| `(minmax-scale xs)` / `(standardize xs)` | `MinMaxScaler` / `StandardScaler` — `standardize` делит на n |
| `(t-mse pred actual)` | `mean_squared_error` для 1-D `(:tensor :f32 (_))` |
| `(range 5)` / `(unique xs)` / `(find xs x)` | `range(5)` / `np.unique` / `next(...)` — утилиты из `collections` |
| `(import-from ml (mean mse))` | `from sklearn.metrics import mean_squared_error` |

Метрики принимают непустые `(:list :flonum)`; fixnum-входы расширяются до flonum. `ratio` вне `0..1` — `throw`; нулевой знаменатель в `precision`/`recall`/`f1` даёт `0.0`. KNN, naive Bayes, деревья, леса, градиентный бустинг, k-means, `cross_val_score`, `GridSearchCV` и ядра в релизе отсутствуют — для обучения служит `defgraph` + `grad!` и модуль `dl`.

## 18.7 Символьный и нейросимвольный слой

| Qualisp | Python-аналог |
|---|---|
| `(match pattern term)` → `(:option :syntax)` | сопоставление с образцом: `match pattern:` / `unification` |
| `(subst var value term)` / `(rewrite term)` / `(rewrite-all term)` | замена и переписывание до неподвижной точки |
| `(simplify term)` / `(diff term var)` | упрощение (`x+0`, `x*1`, `x*0`) и аналитическая производная |
| `(unify a b env)` | `unification.unify` — с occurs-check; пустое окружение — `t` |
| `(collect term var)` / `(diff-canonical term var)` | нормальная форма «сумма мономов»; `d(x*x)/dx` → `(* 2 x)` |
| `(poly? term var)` / `(poly-coeffs term var)` / `(poly-eval term var x)` | `sympy.Poly` / coeffs / eval |
| `(term '(tmean (* (- yhat y) (- yhat y))) '(yhat y))` | символьный терм → тензорный код (`neuro`) |
| `(term->code term vars)` | то же над значением `:syntax`; `term-problem` возвращает первую ошибку, не бросая |
| `(ns-if (:payload x) ((ns-grt score 0) branch-a) (else branch-b))` | жёсткая per-sample маршрутизация — аналога нет (похоже на mixture-of-experts без blending) |
| `(ns-grt a b)` / `(ns-lss a b)` / `(ns-and p q)` / `(ns-or p q)` / `(ns-not p)` / `(ns-eq a b d)` | непрерывные level-функции: знак = истинность, значение — `:flonum`-данные |
| `(ns-band …)` | — в API 0.19.15 отсутствует |

`term` — макрос из `neuro`: оба аргумента должны быть литералами; переменные обязаны быть перечислены в `vars`, глобал в графе — константа без градиента. Один и тот же терм годится для eager-функции, обучаемого графа и ветки `ns-if`. `diff` упрощает, но **не приводит подобные**: `(diff '(* x x) 'x)` → `(+ X X)`, тогда как `diff-canonical` → `(* 2 X)`.

`ns-if` — per-sample hard-роутер `[B, K]`: `(:payload e)` обязательна, `(:eps e)` опциональна; guard-клауз ≥ 2, ровно одна `(else body)` — boundary-ветка, а не default. Тело ветви — функция одного аргумента над компактифицированными строками. Покрытие проверяется через ROBDD: непокрытие, пересечение и пустая ветка — ошибки компиляции. Градиент идёт только по исполненной ветке; градиента по решению роутера нет.

```lisp
(defvar a0 (:tensor :f32 (4)) (tensor (1.0 -1.0 -1.0 -1.0)))
(defvar x  (:tensor :f32 (4 2)) (tensor ((1.0 2.0) (3.0 4.0) (5.0 6.0) (7.0 8.0))))
(ns-if (:payload x)
  ((ns-grt a0 0) (lambda (v (:tensor :f32 (4 2))) (* v 2.0)))
  ((ns-lss a0 0) (lambda (v (:tensor :f32 (4 2))) (* v 3.0)))
  (else          (lambda (v (:tensor :f32 (4 2))) (* v 11.0))))
```

## 18.8 Строки, файлы, прочее

| Qualisp | Python |
|---|---|
| `(strcat a b)` / `(substr s i n)` | `a + b` / `s[i:i+n]` |
| `(number-to-string 42)` / `(to-string x)` | `str(42)` / `str(x)` |
| `(read-file path)` / `(write-file path text)` | `open(path).read()` / `open(path, "w").write(text)` |
| `(append-file path text)` / `(file-exists? path)` | `open(path, "a").write(text)` / `os.path.exists(path)` |
| `(read-lines path)` / `(write-lines path xs)` (модуль `io`) | `open(path).readlines()` / `writelines` — `read-lines` → option-список |
| `(open-file path mode)` → `:fixnum`-хэндл; `(read-line h)` → option | `open(...)` / `f.readline()` — режимы `#\r`, `#\w`, `#\a` |
| `(read-all h)` / `(write-string h s)` / `(write-line h s)` / `(close-file h)` | `f.read()` / `f.write(s)` / `f.write(s + "\n")` / `f.close()` |
| `(with-file (f path mode) body…)` | `with open(path, mode) as f:` — закрывает и на успехе, и на `throw` |
| `(each-line path f)` | `for line in open(path): f(line)` |
| `(to-fixnum s)` / `(to-flonum s)` / `(to-string x)` → `(:option T)` | `int(s)` / `float(s)` / `str(x)` — `or-else` для распаковки |
| `(or-else (to-fixnum s) 0)` | `int(s) if ... else 0` — санкционированный способ распаковать option |

## 18.8.1 QSRD — бинарная сериализация

| Qualisp | Python |
|---|---|
| `(save-image "w.qimg")` / `(load-image "w.qimg")` | `pickle.dump(globals())` / `pickle.load` — образ пользовательского мира |
| `(save-weights "w.qsrd")` / `(load-weights "w.qsrd")` | `torch.save(state_dict)` / `load_state_dict` |
| `(save-weights "ckpt.qsrd" (list 7 8))` | состояние оптимизатора (шаг, lr, seed) едет вместе с весами |
| `(load-weights path)` → `:syntax` | возвращает сохранённое состояние как данные |

QSRD v1/Q2 — бинарный формат с magic `"QSRD"`, u16 major/minor, таблицей секций с crc32 и секциями `meta`, `symbols`, `forms`, `data`, `model`. Символы хранятся один раз и ссылаются по `u32`-индексу. `save-weights` пишет только `meta`/`symbols`/`model`: параметры — стабильные тензоры с `requires_grad` в порядке `SymbolId`. `load-weights` сначала валидирует **все** параметры (неизвестное имя или несовпадение dtype/формы — ошибка без частичного применения).

```lisp
(defvar w (:tensor :f32 (2 2)) (param (tensor ((1.0 2.0) (3.0 4.0)))))
(defvar b (:tensor :f32 (1))   (param (tensor (0.5))))
(save-weights "w.qsrd" (list 7 8))
(load-weights "w.qsrd")   ;; → (7 8)
```

> 🐍 `pickle` не сохраняет dtype/shape тензоров — QSRD хранит S-выражения, тензоры с dtype и формой, скаляры и состояние в одном файле. Не сериализуются JIT-страницы, скомпилированные графы, планы `ns-if`, `future` и raw-указатели. `qsrd-save`/`qsrd-load` как имен API в 0.19.15 нет: рабочий путь — `save-image`/`save-weights`.

## 18.8.2 Аргументы командной строки: `argparse`

| Qualisp | Python |
|---|---|
| `(command-line-args)` → `(:list :string)` | `sys.argv` — первый элемент — путь скрипта (argv[0]) |
| `(flag? args "verbose")` | `"--verbose" in args or "-verbose" in args` |
| `(option-value args "name")` | `args[args.index("--name")+1]` — понимает `--name value` и `--name=value` |
| `(require-option args "name")` | обязательный параметр: отсутствие — ошибка |
| `(positionals args (list "name"))` | позиционные аргументы; опции со значениями объявляются явно |

Имена опций пишутся без дефисов. Флаги интерпретатора разбираются только до имени скрипта; всё после него — аргументы программы, включая `--foo`.

```lisp
(import argparse)
(print (command-line-args))
(print (flag? (command-line-args) "verbose"))
(print (option-value (command-line-args) "name"))
```

## 18.9 Компиляция

| Qualisp | Python-мир |
|---|---|
| `(defun f (x :fixnum) :fixnum …)` + `(f 1)` | `def f(x): …` — тело компилируется в JIT-страницу **при определении** |
| `(defgraph g (x (:tensor :f32 (2))) … body)` | `torch.compile(fn)` — граф трассируется один раз и компилируется лениво |
| `(defuse f (x (:tensor :f32 (4)) a :flonum) …)` | fused-ядро: один x86-64 цикл с AVX2 (8 f32) + scalar-tail |
| `qualisp --trace tr.json script.qlsp` | `python -m cProfile -o tr.prof script.py` |
| `qprof tr.json --top 10` | `pstats.Stats("tr.prof").sort_stats("cumulative").print_stats(10)` |

Явного `(jit f)`/`(discard-jit f)` в языке нет: компиляция прозрачна, а страница — выбрасываемый кэш, который пересобирается из мастер-формы. Ошибки рантайма несут исходную позицию благодаря source maps (`file:line:col: message`).

`--trace` пишет Chrome Trace JSON: события `B`/`E`, микросекунды, `pid=1,tid=1`; инструментированы границы вызовов функций, запуск HLO-графа и фаза `ns-if`. Компиляция JIT-страницы входит в inclusive-время первого вызова. Трасса пишется и при ошибке.

```text
$ qualisp --trace /tmp/fib.json fib.qlsp
2584
$ qprof /tmp/fib.json
trace: /tmp/fib.json (16722 events)
name                          calls       total_us        self_us   self%
FIB                            8361       255806.0        21248.0  100.0%

total self time: 21248.0 us
```

## 18.10 FFI

| Qualisp | Python-мир |
|---|---|
| `(ffi-load "libm.so.6")` → `:fixnum` | `ctypes.CDLL("libm.so.6")` |
| `(ffi-defun "pow" c-pow f64 (x f64 y f64))` | `ctypes.CFUNCTYPE` — C-имя, лисп-имя, тип возврата, типы аргументов |
| `(c-pow 2.0 3.0)` → `8.0` | `pow(2.0, 3.0)` — вызов как обычной функции, в том числе из JIT-кода |

Типы: `f64 f32 i64 u64 i32 u32 i16 u16 u8 ptr string void`; `void` — только в возврате. Символ ищется в уже загруженных библиотеках. Трамплин — собственный x86-64 генератор, без libffi; на Windows FFI использует `LoadLibraryA`/`GetProcAddress` и Win64-трамплин (регистры выбираются **по позиции** аргумента, аргументы с позиции ≥ 4 идут на стек после 32 байт shadow space). Нет variadic-функций, структур по значению и колбэков C→Lisp. `ffi-load`/`ffi-defun` — top-level формы.

### 18.10.1 Встраивание

В релизе поставляются два бинарника — `qualisp` (REPL/раннер с JIT) и `qprof` (агрегатор трасс). Встраиваемого C API (`qualisp_*`) в 0.19.15 нет: Lisp вызывается как программа через `qualisp script.qlsp [args…]` либо через REPL; внешний код обменивается данными через QSRD и потоковый вывод.

## 18.11 Параллелизм

| Qualisp | Python-мир |
|---|---|
| `(future f)` — `f` — нульарная функция | `executor.submit(fn)` — форма исполняется в фоне |
| `(await f)` / тип `:future` | `future.result()` / `concurrent.futures.Future` |
| `(type-of (future (lambda () 1)))` → `FUTURE` | `type(Future()).__name__` |

Глобалы в задаче доступны только для чтения; результат глубоко клонируется в stable при `await`. У задачи свой thread-local `Runtime` (свой scratch/world, копия глобалов). Возвращать future из future и ждать его в другом потоке нельзя.

```lisp
(defvar f (:future :fixnum) (future (lambda () (+ 1 2))))
(print (await f))   ;; 3
```

> 🐍 Мир задачи — снимок, ближе к `multiprocessing`, чем к общему heap потоков Python. Отдельного модуля `async` и примитивов `pmap`/`chan`/`thread-count` в 0.19.15 нет.

## 18.12 Ошибки

| Qualisp | Python |
|---|---|
| `(throw "boom")` | `raise ValueError("boom")` — тип `:never` |
| `(catch-error (lambda () body))` → `#<result>` | `try/except` как значение |
| `(result-ok? r)` | `isinstance(r, Ok)` |
| `(result-value r)` | `r.value` — полезная нагрузка `ok` |
| `(result-message r)` | `str(e)` — сообщение ошибки |
| `(type-of r)` → `RESULT` | `type(r).__name__` |

`Error` — структура с полями `message :string`, `cause :(:option (:struct Error))`, `traceback :(:list CallFrame)`; кадр — `(name file line form)`. Ошибки компиляции и типов — не `Error`-значения, а отказ с диагностикой со спаном (`error at 1:1: …` в REPL, `file:line:col: message` в файле). Хвостовые вызовы переименовывают верхний кадр, не растя стека.

```lisp
(print (result-ok? (catch-error (lambda () 42))))   ;; t
(print (result-value (catch-error (lambda () 42)))) ;; 42
(print (result-message (catch-error (lambda () (throw "boom"))))) ;; "boom"
```

> 🐍 `try`, `ok?`, `err?`, `ok`, `err`, `error-message` как функции в 0.19.15 отсутствуют: `try` описан как макрос, но в REPL-контексте не разрешается; рабочий путь — `catch-error` + `result-*`.
