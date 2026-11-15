# Глава 18. Шпаргалка: Qualisp ↔ Python

Быстрый перевод между Qualisp и Python-стеком (NumPy / PyTorch / scikit-learn). Имена примитивов в Qualisp после `SymbolTable::intern` хранятся в верхнем регистре (`T+`, `MATMUL!`, `MSE!`, `PARAM`, `GRAD!`); пользователь может писать их в любом регистре.

## 18.0 Гомоиконность

| Qualisp | Python |
|---|---|
| `(read-from-string "(+ 1 2)")` | `ast.literal_eval("(1 + 2)")` / `eval("(1+2)", {})` |
| `(write-to-string expr)` | `repr(expr)` / `ast.unparse(ast.parse(...))` |
| `(eval expr)` | `eval(...)` (но нативно над S-выражениями) |
| `(functionp f)` | `callable(f)` — предикат значения; `(functionp 'f)` → `nil`, символ — данные |
| `(function-params f)` / `(function-body f)` | `inspect.signature(f).parameters` / `inspect.getsource(f)` (но структурно) |
| `(function-env f)` | `f.__closure__` |
| `(symbol-name 'x)` | `str(x)` (но для имён) |
| `(boundp 'x)` | `'x' in globals()` |
| `(jit f)` / `(setq f (jit f))` | CPython 3.13 JIT (copy-and-patch, тоже без IR в рантайме) — гл. 20 |
| `(jit-sig f)` / `(jit-stale-p f)` | (нет аналога) — наблюдаемые типы параметров JIT: `(UNKNOWN)` до прогрева, `(FIXNUM)` после |
| `(jit-stats)` | (нет аналога) — алист учёта JIT: компиляции, кэш-хиты, интерпретированные вызовы по категориям, именованные отказы |
| `(save-image "w.qimg")` / `(load-image ...)` | `pickle(globals())` + `torch.save` в одном файле — гл. 21 |
| `(discard-jit)` | (нет аналога) — сброс всех нативных страниц, поведение не меняется |
| `(set-reader-macro '!' f)` | (нет аналога) |
| `(randint 1 6)` | `random.randrange(1, 6)` — верхняя граница исключается |
| `(import-from ml (a b))` | `from ml import a, b` |
| `(param (randn (list 4 2)))` | `w = nn.Parameter(torch.randn(...))` |

## 18.1 Базовый синтаксис

| Qualisp | Python |
|---|---|
| `(+ 1 2)` | `1 + 2` |
| `(f x y)` | `f(x, y)` |
| `(defun f (x) body)` | `def f(x): return body` |
| `(\ (x) (* x x))` | `lambda x: x * x` |
| `(defvar x 1)` | `x = 1` (модуль) |
| `(setq x 2)` | `x = 2` |
| `(setq a 1 b 2)` | `a, b = 1, 2` — все пары слева направо, каждая видит предыдущую (нечётное число аргументов — ошибка) |
| `(let ((x 1) (y 2)) ...)` | `x, y = 1, 2` (в области) |
| `(if c a b)` | `a if c else b` — истина всё, кроме `nil`; `0` — истина |
| `(cond (c1 r1) (c2 r2) (t r))` | `if/elif/else` |
| `(while c body)` | `while c:` |
| `(progn a b c)` | тело функции / блок |
| `(and a b c)` / `(or a b c)` | `a and b and c` / `a or b or c` — в позиции головы ленивые спецформы |
| `(apply and xs)` / `(map or xs ys)` | `functools.reduce(operator.and_, map(bool, xs))` — как значение `AND`/`OR` — n-арные функции: `(and)` → `T`, `(or)` → `nil` |
| `(xor a b c)` | `functools.reduce(operator.xor, map(bool, xs))` — нечётное число истинных; `(xor)` → `nil` |
| `; комментарий` | `# комментарий` |
| `'(1 2 3)` | `[1, 2, 3]` (литерал) |
| `(= a b)` | `a == b` — числа сравниваются **по значению**: `(= 0 0.0)` → `t` |
| `(equal a b)` | `a == b` (cons/строки/символы — структурно; замыкания, HLO/JIT-программы, структуры, тензоры — по идентичности, как `a is b`) |
| `(eq a b)` | `a is b` — идентичность символа (один интернированный `Symbol*`) |
| `(EQ a b)` | `a is b` — идентичность объекта (тот же объект: замыкание, структура, тензор) |
| `:linux`, `:avx2` (keyword-символы) | литералы, как `enum`-константы: самовычисляемы, `(equal :linux :linux)` → `t` |
| `(print (strcat "x " (to-string 42)))` | `print("x", 42)` — `print`/`display` принимают **ровно один аргумент** |
| `(type-of x)` | `type(x).__name__` (в Qualisp — uppercase) |

## 18.1.1 Структуры, методы и трейты

| Qualisp | Python |
|---|---|
| `(defstruct pt (x y))` / `(pt 1 2)` | `@dataclass` / `class pt: ...` + `pt(1, 2)` — конструктор-функция |
| `(slot-value p 'x)` / `(set-slot p 'x v)` | `p.x` / `p.x = v` — чтение и запись поля |
| `(defstruct p (x: byte y: i64))` | типизированные поля: `byte` (raw 0..255), `i16`/`u16`/`i32`/`u32`/`i64`/`u64`/`f32`/`f64`, `ptr` — как ctypes-раскладка |
| `(type-of p)` → `STRUCT` | `type(p).__name__` |
| `(IMPL name (defun m (self) ...))` | методы класса: `def m(self): ...` |
| `(m obj)` — например `(len2 (v2 3 4))` | `obj.m()` — диспетч по первому аргументу |
| `(TRAIT name (sig ...))` + `(trait-complete-p 'S 'T)` | `typing.Protocol` + проверка полноты реализации |

## 18.2 Списки

| Qualisp | Python |
|---|---|
| `(car xs)` / `(cdr xs)` | `xs[0]` / `xs[1:]` |
| `(cons x xs)` | `[x] + xs` |
| `(nth i xs)` | `xs[i]` |
| `(length xs)` | `len(xs)` |
| `(map f xs)` / `(map f xs ys)` | `list(map(f, xs))` / `list(map(f, xs, ys))` — многолистовый, останавливается на корочайшем (модуль `collections`) |
| `(filter p xs)` | `list(filter(p, xs))` (модуль `collections`) |
| `(reduce fn xs init)` | `functools.reduce(fn, xs, init)` — но `(fn элемент аккумулятор)`: элемент первым (не порядок CL); без `init` первый элемент — аккумулятор, `nil`-init = без инициализатора |
| `(zip-with f xs ys)` | `[f(a,b) for a,b in zip(xs,ys)]` |
| `(member x xs)` | `x in xs` |
| `(range 5)` | `range(5)` / `list(range(5))` |
| `(take n xs)` / `(drop n xs)` | `xs[:n]` / `xs[n:]` |
| `(reverse xs)` | `xs[::-1]` |
| `(append xs ys)` | `xs + ys` |

## 18.2.1 Коллекции: `dict`, `set`, `queue`, `heap` (модуль `collections`)

| Qualisp | Python |
|---|---|
| `(dict)` / `(dict-put d k v)` / `(dict-get d k)` | `dict()` / `d[k] = v` / `d[k]` |
| `(dict-get d k 'fallback)` | `d.get(k, fallback)` |
| `(dict-has? d k)` / `(dict-del d k)` | `k in d` / `del d[k]` |
| `(dict-keys d)` / `(dict-vals d)` / `(dict-size d)` | `list(d)` / `list(d.values())` / `len(d)` |
| `(set)` / `(set-add s x)` / `(set-has? s x)` | `set()` / `s.add(x)` / `x in s` |
| `(set-add-all s other)` / `(set-del s x)` | `s.update(other)` / `s.remove(x)` |
| `(set-size s)` / `(set->list s)` | `len(s)` / `list(s)` |
| `(queue)` / `(queue-push q x)` / `(queue-pop q)` | `deque()` / `q.append(x)` / `q.popleft()` |
| `(queue-peek q)` / `(queue-empty? q)` / `(queue-size q)` | `q[0]` / `not q` / `len(q)` |
| `(heap)` / `(heap-with less)` | `heapq` (min-куча) / куча со своей функцией порядка |
| `(heap-push h x)` / `(heap-pop h)` / `(heap-peek h)` | `heappush(h, x)` / `heappop(h)` / `h[0]` |
| `(heap-empty? h)` / `(heap-size h)` | `not h` / `len(h)` |

Изменяемость — через поля struct (`set-slot`); ключи сравниваются через `equal`. Коллекции — обычные S-выражения, поэтому сериализуются без дополнительной работы. `queue` — амортизированно O(1) (front/rear), `heap` — skew-куча с `heap-merge` для слияния.

## 18.3 Тензоры (NumPy)

| Qualisp | NumPy |
|---|---|
| `(tensor ((1.0 2.0) (3.0 4.0)))` | `np.array([[1.,2.],[3.,4.]])` |
| `(list->tensor '(1.0 2.0))` | `np.array([1., 2.])` — плоский список → форма `[N]` |
| `(list->tensor '((1.0 2.0)))` | `np.array([[1., 2.]])` — одна строка → форма `[1,2]` |
| `(zeros (list 2 3))` | `np.zeros((2,3))` |
| `(ones (list 4 2))` | `np.ones((4,2))` |
| `(randn (list 100 1))` | `np.random.randn(100,1)` |
| `(t+ a b)` / `(t* a b)` | `a + b` / `a * b` |
| `(matmul a b)` | `a @ b` |
| `(reshape t (list 4 1))` | `t.reshape(4,1)` |
| `(transpose t)` | `t.T` |
| `(tsum t)` / `(tsum t (list 0))` | `t.sum()` / `t.sum(axis=0)` |
| `(tmean t)` | `t.mean()` |
| `(tmax t)` / `(tmax t (list 1))` | `t.max()` / `t.max(axis=1)` |
| `(argmax t)` | `t.argmax()` |
| `(tensor-shape t)` | `t.shape` |
| `(tensor-dtype t)` | `t.dtype` (`F32`/`F16`/`F64`/`I32`/`I64`/`U8`) |
| `(tsin t)` / `(tcos t)` / `(t-exp t)` | `np.sin(t)` / `np.cos(t)` / `np.exp(t)` |
| `(concat a b (list 0))` / `(stack a b (list 0))` | `np.concatenate([a,b], axis=0)` / `np.stack([a,b], axis=0)` |
| `(tensor-f16 t)` / `(tensor-f32 t)` / `(tensor-u8 t)` | `t.astype(np.float16/float32/uint8)` |
| `(tref t i)` | `t.flatten()[i]` — один индекс = плоский номер элемента |
| `(tref t i j)` | `t[i, j]` — N индексов = координаты ровно ранга N; выход за границы — ошибка |
| `(tensor-item t i)` | `t.flatten()[i]` — элемент по индексу (индекс обязателен) |
| `(tset! t i v)` | `t.flat[i] = v` — запись элемента на месте по плоскому индексу (float — любое число; целые — fixnum строго в диапазоне типа), возвращает `t`; выход за границу и отрицательный индекс — ошибка |
| `(tensor->list t)` | `t.ravel().tolist()` |
| `(FINITE? t)` | `np.isfinite(t).all()` — битовая проверка конечности (NaN/Inf) |
| `(save-npy t "f.npy")` / `(load-npy "f.npy")` | `np.save` / `np.load` — тензор первым, путь вторым |

### Окна и выборки

| Qualisp | NumPy |
|---|---|
| `(t-rows t from len)` | `t[from:from+len]` — **окно только для чтения** (в NumPy срез записываемый) |
| `(t-slice t from len)` | `t.reshape(-1)[from:from+len]` — плоский индекс, тоже read-only |
| `(t-gather-rows table ids)` | `table[ids]` — копия строк по индексам (`ids` — одномерный f32-тензор или список целых) |
| `(t-scatter-add dst ids src)` | `np.add.at(dst, ids, src)` — `dst[ids[i],:] += src[i,:]`, дубликаты складываются |
| `(t+= dst src)` / `(t*= dst src)` | `dst += src` / `dst *= src` — in-place, тензоры обязательны |
| `(tset! t i v)` | `t.flat[i] = v` — единственная запись в тензор по индексу |

> ⚠️ **Окна read-only.** `(t-rows t 0 1)` — алиас над буфером владельца, а не копия: `TSET!`/`T+=`/`T-RELU`/`SGD-STEP` в него отказывают («destination is a read-only view»). Окно читается после выхода владельца из scope и после переписи имени; `TSET!` в источник виден через окно, обратное — нет; окно от окна материализуется в копию. `(defvar v (t-rows t 0 1))` на верхнем уровне даёт **копию** (escape в stable), и она writable. Устройство — гл. 5 и 6.

```lisp
(defvar m (tensor ((1.0 2.0) (3.0 4.0))))
(print (t-rows m 1 1))          ;; → tensor([3, 4], shape=[1, 2]) — форма [V,d] → [len,d]
(print (t-slice m 1 2))         ;; → tensor([2, 3], shape=[2]) — с плоского индекса 1
(print (t-gather-rows m (list 0)))  ;; → tensor([1, 2], shape=[1, 2])
(print (t-scatter-add m (list 0) (tensor ((10.0 20.0)))))  ;; → m, дубликаты складываются
(print m))                      ;; → tensor([11, 22, 3, 4], shape=[2, 2])
```

> 🐍 В NumPy `a[0:1]` — тоже валидный `ndarray`, и в него можно писать (`a[0:1] = 0`). В Qualisp окно — read-only алиас: изменить буфер можно только через источник (`TSET!`, `T+=`, `T*=`) или `T-SCATTER-ADD` в целевой тензор. Для обучения строк таблицы — `EMB-LOOKUP`/`T-GATHER!` плюс `ADAM-SPARSE-STEP`, а не окно (гл. 6, 8).

## 18.4 Автоград (PyTorch)

| Qualisp | PyTorch |
|---|---|
| `(param (randn (list 4 2)))` | `torch.randn(4,2, requires_grad=True)` |
| `(matmul! x w)` | `x @ w` (в графе) |
| `(relu! h)` | `F.relu(h)` — в backward пропускает только `x > 0` (в нуле 0) |
| `(sigmoid! x)` | `torch.sigmoid(x)` — Lisp-композиция `0.5 * (1 + tanh(x/2))` |
| `(tanh! x)` | `torch.tanh(x)` |
| `(mse! pred y)` | `F.mse_loss(pred, y)` — скаляр |
| `(cross-entropy! logits y)` | `F.cross_entropy(logits, y)` — `y` — целые индексы `[batch]`, по одному на строку |
| `(grad! loss)` | `loss.backward()` |
| `(grad-of w)` | `w.grad` |
| `(sgd-step w lr)` | SGD `opt.step()` (на одном парам.) |
| `(adam-step w lr)` | Adam `opt.step()` |
| обычные `matmul`, `t+` (без `!`) | `with torch.no_grad():` |
| `(sample logits)` | `torch.distributions.Categorical(logits).sample()` (one-hot) |
| `(reinforce-loss! logits idx reward)` | REINFORCE: `reward * F.cross_entropy(logits, idx)` |

### Лента, накопление и состояние

| Qualisp | PyTorch |
|---|---|
| `(tape-form)` | `loss.grad_fn` — но формой: журнал `(OP результат (входы...) payload)`, новые записи первыми |
| `(grad! loss)` | `loss.backward()` — дренирует ленту глобально |
| `(grad-accum! loss)` | `loss.backward()` внутри цикла: накапливает в stable-градиенты листьев |
| `(zero-grad! w)` | `optimizer.zero_grad()` — обнуляет/создаёт нулевой градиент |
| `(set-grad! w g)` | `w.grad = g` — пишет градиент; без градиента no-op |
| `(t-detach x)` | `x.detach()` — копия без `requires_grad`/узла |
| `(adam-state w)` | `optimizer.state_dict()['state'][id(w)]` — `(m v step)` |
| `(adam-state! w m v step)` | восстановление состояния из чекпоинта (`nil` сбрасывает) |
| `(adams-snapshot params)` / `(adams-restore! snap params)` | `deepcopy(state_dict)` / `load_state_dict` — списки параметров ↔ списки состояний |
| `(grad-norm w ...)` | `torch.nn.utils.clip_grad_norm_` — только норма, тензор `[1]` |
| `(clip-grad! max w ...)` | `torch.nn.utils.clip_grad_norm_(params, max)` — глобальный клип, возвращает норму **до** клипа |
| `(adam-sparse-step w ids lr)` | `optimizer.step()` только по строкам `ids` (sparse Adam) — `O(|ids|·d)` вместо `O(V·d)` |

```lisp
(defvar w (param (randn (list 4 2))))
(while (< i steps)
  (grad-accum! (mse! (matmul! x w) y))
  (setq i (+ i 1)))
(clip-grad! 1.0 w)
(sgd-step w lr)
```

`grad!`/`grad-accum!` требуют **скалярный loss** и применяются к графу, построенному `!`-операциями; градиент, не прочитанный до границы формы, умирает с предупреждением (`grad!: 1 gradient(s) left unread at the form boundary ...`) — обучение «через форму» не учит. `sgd-step`/`adam-step` (без `!`) — операции над одним параметром; шаг в цикле делает `train-step` (гл. 8).

- **`tape-form`** — сама лента как данные: рёбра — те же SExpr-ячейки тензоров, `equal` по тензору — по идентичности. У `SUM!`/`MEAN!` payload — `(оси keepdims)`, у `deftape!` — список с backward-функцией, у встроенных — `nil`.
- **`deftape!`** определён в `core.qlsp` — доступен и `dl`, и `ml`. Не-тензорные аргументы (`ids`/`from`/`len`) не рёбра градиента: они сохраняются в списке входов узла (видны в `tape-form`) и передаются в Lisp-backward как есть; тензорные — borrowed-виды.
- **`grad-accum!`** копит градиент, если граф живёт между оборотами `while`; лента всё равно живёт один оборот, поэтому «повторным `GRAD!`» батч не накопить — только `GRAD-ACCUM!` + `ZERO-GRAD!`.
- **`clip-grad!`**/`grad-norm` ставятся между `GRAD!`/`GRAD-ACCUM!` и шагом. Нефинитная норма — громкая ошибка, `max-norm <= 0` — ошибка; сначала проверяются все градиенты, потом запись. `GRAD-NORM` возвращает **тензор формы `[1]`**, а не скаляр.
- **`adam-state`** отдаёт `(m v step)`, где `m`/`v` — копии значений, а не view; `(nil nil 0)` до первого шага. Lisp-обёртки `adams-snapshot`/`adams-restore!` переносят веса и оптимизатор через QSRD в другой процесс (гл. 7, 16).
- **`adam-sparse-step`** — шаг Adam только по строкам `ids`; состояние `m`/`v` общее с `ADAM-STEP` и живёт на тензоре, bias-correction — на шаг, дубликаты обновляют строку один раз. Шаг гасит `param->grad` (повторный шаг без backward — no-op). Цена — `O(|ids|·d)` вместо `O(V·d)`.
- **Операторы пишут ленту сами**: `(+ param 1)` и `(* param x)` обучаются, `(+ обычный-тензор 1)` — нет; у входа `requires_grad` узел появляется без `!`-суффикса. `*` на тензорах — Hadamard, матричное — `MATMUL`.

> 🐍 В PyTorch граф невидим: `loss.grad_fn` — объект, который нельзя напечатать как форму. В Qualisp лента — обычные S-выражения, и её можно инспектировать, сравнивать и сериализовать.

## 18.5 Слои (DL-модель ↔ torch.nn)

| Qualisp | PyTorch |
|---|---|
| `(linear 4 2)` (из DL) | `nn.Linear(4, 2)` (но слой — cons, а не объект) |
| `(list (linear 1 4) 'relu (linear 4 1))` | `nn.Sequential(nn.Linear(1,4), nn.ReLU(), nn.Linear(4,1))` |
| `(net-forward model x)` | `model(x)` |
| `(train-step model x y 'sgd lr)` | `loss.backward(); opt.step()` |
| `(train-epochs model x y 'adam lr 100)` | `for _ in range(100): train_step(...)` |
| `(layer-w layer)` / `(layer-b layer)` | `layer.weight` / `layer.bias` |
| `(batchnorm! x g b eps)` / `(layernorm! x g b)` | `nn.BatchNorm1d` / `nn.LayerNorm` — композиции на Lisp; `batchnorm!` берёт `[N,C,H,W]` |
| `(dropout h 0.5)` | `nn.Dropout(0.5)` — tape-free копия через `T-DETACH` |
| `(softmax! x)` / `(softmax x)` | `F.softmax(x, dim)` / `F.softmax` в `no_grad` |
| `(attention q k v)` / `(attention q k v mask)` | `F.scaled_dot_product_attention(q, k, v)` (одно-головное, аддитивная маска) |
| `(emb-lookup table ids)` | `F.embedding(ids, table)` — синоним `t-gather!` |
| `(conv2d! x k s p)` | `F.conv2d(x, k, stride=s, padding=p)` — `x` `[N,C,H,W]`, ядро `[Cout,Cin,KH,KW]` |
| `(maxpool2d! x k s)` | `F.max_pool2d(x, k, s)` — `x` `[N,C,H,W]`, окно `k`, шаг `s` |
| `(weighted-lookup probs v1 v2)` | взвешенная сумма (скаляр) |
| `(net-infer model x)` | `with torch.no_grad(): model(x)` (tape-free, для `while`) |

> 🐍 Ключевое отличие: Qualisp DL — это **данные** (cons-список), а не объект. Модель сериализуется QSRD как обычная структура. PyTorch-модель — объект с `state_dict`, требует отдельного протокола сериализации.

`softmax!`/`softmax`, `layernorm!`, `batchnorm!`, `sigmoid!`, `dropout` — композиции на Lisp поверх примитивов, а не отдельные C++-ядра: `softmax!` — `TMAX` (устойчивость) → `EXP!` → `SUM!` → `/`, `softmax` — та же композиция на tape-free `TMAX`/`T-EXP`/`T-LOG`/`TSUM`. Поэтому форма ленты у них видна в `TAPE-FORM` (гл. 7, 8). `net-infer` понимает слои `'softmax` и `(layernorm gamma beta)` и не требует `GRAD!`-пустышки.

## 18.5.1 Строки (модуль `string` ↔ `str`)

| Qualisp | Python |
|---|---|
| `(string-length s)` | `len(s)` |
| `(string-split sep s)` | `s.split(sep)` — **разделитель первым** |
| `(string-join sep list)` | `sep.join(list)` |
| `(string-upcase s)` | `s.upper()` |
| `(string-downcase s)` | `s.lower()` |
| `(string-trim s)` | `s.strip()` |
| `(string-replace s "a" "b")` | `s.replace("a", "b")` |
| `(string-index s "x")` | `s.find("x")` |
| `(string-reverse s)` | `s[::-1]` |
| `(char-code "A")` | `ord("A")` — принимает строку |
| `(code-char 65)` | `chr(65)` — возвращает строку |
| `(string-to-number "3.14")` | `float("3.14")` |
| `(string-contains s "sub")` | `"sub" in s` |
| `(string-starts-with s "pre")` | `s.startswith("pre")` |
| `(string-ends-with s "suf")` | `s.endswith("suf")` |
| `(string-slice s 1 5)` | `s[1:5]` |
| `(string-lines s)` | `s.splitlines()` |
| `(string-words s)` | `s.split()` |
| `(string-repeat s 3)` | `s * 3` |
| `(to-string 42)` | `str(42)` |
| `(type-of s)` → `STRING` | `type(s).__name__` → `str` |
| `(string-ref s i)` | `ord(s[i])` — байт 0..255 без аллокации |
| `(string-view s from len)` | `s[from:from+len]` — **окно**, копия не делается |

> 🐍 **Строковые окна.** `(STRING-VIEW s from len)` и куски от `STRING-SPLIT` — окна над stable-строкой, а не копии. Они прозрачны: `TYPE-OF` → `STRING`, `equal` по содержимому, `STRING-LENGTH`/`STRCAT`/`SUBSTR`/`STRING-SPLIT`/`STRING-CONTAINS` работают одинаково. Над скретч-строкой (литерал, результат `SUBSTR`) и над окном (`view-of-view`) операция даёт копию; `(defvar v (STRING-VIEW ...))` материализует копию. Границы — ошибка с именем операции (`string-view: range [1,6) out of [0,3)`). Устройство — гл. 6.

```lisp
(STRING-SPLIT " " "a b c")   ; → ("a" "b" "c")
(STRING-SPLIT "," "a,b,c")   ; → ("a" "b" "c")
(STRING-SPLIT "a,b,c" ",")   ; → (",")  — неверный порядок
```

## 18.6 Классический ML (sklearn)

| Qualisp | scikit-learn |
|---|---|
| `(range 5)` / `(range 5 10 2)` | `range(5)` / `range(5, 10, 2)` |
| `(take n xs)` / `(drop n xs)` | `xs[:n]` / `xs[n:]` — итеративные обходы |
| `(train-test-split data target ratio)` | `train_test_split(train_size=ratio)` — четыре списка: train/test × data/target |
| `(standard-scale-lst xs)` | `StandardScaler` |
| `(minmax-scale-lst xs)` | `MinMaxScaler` |
| `(knn-predict x X y k)` | `KNeighborsClassifier(k)` |
| `(nb-fit X y)` / `(nb-predict x m)` | `GaussianNB` |
| `(build-tree X y depth)` | `DecisionTreeClassifier(max_depth=...)` |
| `(random-forest-fit X y n d)` | `RandomForestClassifier(n, max_depth=d)` |
| `(gb-fit X y n lr d)` | `GradientBoostingClassifier(n, lr, d)` |
| `(kmeans-fit X k iters)` | `KMeans(k, max_iter=iters)` |
| `(accuracy-score yp yt)` | `accuracy_score` |
| `(cross-val-score X y k fit score)` | `cross_val_score(cv=k)` |
| `(grid-search X y params fit score)` | `GridSearchCV` |
| `(linear-kernel a b)` / `(rbf-kernel a b g)` / `(poly-kernel a b d c)` | kernels из `sklearn.metrics.pairwise` |
| `(softmax-list xs)` / `(log-softmax-list xs)` | `torch.softmax` / `log_softmax` (стабильный max-shift) |
| `(mse-loss pred target)` / `(mae-loss ...)` / `(cross-entropy-list ...)` | `nn.MSELoss` / `nn.L1Loss` / `nn.CrossEntropyLoss` |
| `(one-hot k n)` | `one_hot` / `np.eye(n)[k]` |
| `(is-fixnum? v)` / `(is-flonum? v)` / `(is-string? v)` / `(is-tensor? v)` | `isinstance(v, ...)` |
| `(sigmoid-kernel a b alpha c)` | `1 / (1 + exp(-(α·⟨a,b⟩+c)))` — **не** `tanh`-версия |

## 18.7 Символьный и нейросимвольный слой

| Qualisp | Python-аналог |
|---|---|
| `(match v (p1 r1) (p2 r2) (_ def))` | `match v: case p1: r1; case p2: r2; case _: def` |
| `(unify a b)` | `unification.unify(a, b)` |
| `(put 'cat 'legs 4)` / `(get 'cat 'legs)` | атрибуты у enum'ов / `obj.attr = 4` |
| `(assert-fact (cat ?x))` + `(defrule ...)` + `(run-rules)` + `(query ...)` | pyDatalog / CLIPS / Prolog |
| `(amb-let ...)` + `(require ...)` + `(amb-collect ...)` | `itertools.product` + backtracking (`require` — только `T`/`nil`, `amb-collect` перечисляет **все** решения) |
| `(ns-if logits (("a") branch-a) (("b") branch-b))` | tree-of-experts (нет аналога) |
| `(ns-grad! loss 'cat)` | per-sample STE + guilt detector (нет аналога) |
| `(second-best scores)` | второй максимум (ties → соседнее вхождение; откат маршрута) |
| `(slot-set 'name form)` / `(eval-sandboxed form b)` | макро-слот + безопасный eval (нет аналога) |
| `(ns-route logits slots)` | маршрутизация по макро-слотам (нет аналога) |
| `(ns-train-coder! ...)` | RL-агент-кодер: 2-шаговый REINFORCE с baseline (нет аналога) |
| `(ns-config :psi 4 :buffer 1048576)` | (нет аналога) — **глобальная** установка: `:psi` — во сколько раз медленнее учится символьный слой; `:buffer` — бюджет пути графа в байтах; читается `(ns-config)` → алист `((:PSI . v) (:BUFFER . v) (:VERIFY-GATE . v))`. Действие ограничено телом `HLO-START`, если конфиг задан там (гл. 10, 14) |
| `(:proves (row) body...)` в ветке `ns-if` | (нет аналога) — верификация ответа ветви: тело проверяется на строке ответа, `T` = доказано, `nil` = не доказано |

## 18.8 Строки, файлы, прочее

| Qualisp | Python |
|---|---|
| `(strcat a b)` | `a + b` / `f"{a}{b}"` |
| `(substr s i n)` | `s[i:i+n]` |
| `(number-to-string 42)` | `str(42)` |
| `(to-string x)` (общий) | `str(x)` |
| `(slurp "f")` / `(spit "f" s)` | `open("f").read()` / `.write(s)` |
| `(read-lines "f")` | `open("f").readlines()` |
| `(read-csv "f.csv")` | `csv.reader(open("f.csv"))` |
| `(re-find p s)` / `(re-replace p r s)` | `re.search` / `re.sub` |
| `(re-split p s)` | `re.split(p, s)` |
| `(now-ts)` / `(sleep s)` | `time.time()` / `time.sleep(s)` |
| `(now-iso-str)` | `datetime.now().isoformat()` |
| `(elapsed expr)` | contextmanager `timeit`/декоратор |
| `(http-get url)` | `requests.get(url).text` |
| `(dashboard port)` + `(log-metric "loss" v)` | TensorBoard |

## 18.8.1 QSRD — бинарная сериализация

| Qualisp | Python |
|---|---|
| `(qsrd-save path (k1 v1 k2 v2 ...))` | `pickle.dump(obj, f)` — но плоский список: имя, значение, имя, значение |
| `(qsrd-load path)` | `obj = pickle.load(f)` — алист `(("имя" . значение) ...)` |
| `(qsrd-get model "имя")` | `obj.имя` (если QSRD — структура) |

```lisp
(qsrd-save "m.qsrd" (list "w" w "b" b))   ;; → "m.qsrd"
(qsrd-load "m.qsrd")                      ;; → (("w" tensor(...)) ("b" tensor(...)))
```

> 🐍 `pickle` теряет dtype/shape тензоров (нужен отдельный `torch.save` + `np.save`). QSRD хранит всё вместе: S-выражения, тензоры с dtype и шейпом (на **любой глубине** структуры), скаляры — в одном файле с magic `"QSRD"` и u16-версией. Замыкания, примитивы и HLOPROG не сериализуются — это не данные. Второй аргумент `qsrd-save` — именно список `(k1 v1 k2 v2 ...)`, а не отдельные аргументы; имя — строка или символ.

**Чекпоинт с Adam.** Состояние оптимизатора — обычные данные, поэтому веса и `m`/`v` едут в одном файле:

```lisp
(qsrd-save "ckpt.qsrd" (list "w" w "state" (adams-snapshot (list w))))
;; ... в другом процессе:
(defvar ck (qsrd-load "ckpt.qsrd"))
(adams-restore! (qsrd-get ck "state") (list w))
```

Продолжение из чекпоинта совпадает с непрерывным прогоном — состояние `m`/`v` переносится вместе с весами (гл. 7, 16).

## 18.8.2 Аргументы командной строки: `argparser` + `command-line`

| Qualisp | Python |
|---|---|
| `(command-line)` | `sys.argv` — список аргументов командной строки |
| `(argparser :description "обучение")` | `argparse.ArgumentParser(description="обучение")` |
| `(add-argument p "dataset")` | `add_argument("dataset")` — позиционный (обязателен по умолчанию) |
| `(add-argument p "--epochs" :type :int :default 10)` | `add_argument("--epochs", type=int, default=10)` |
| `(parse-args p (cdr (command-line)))` | `parser.parse_args()` |
| `(arg-get parsed "epochs")` | `args.epochs` |
| `(arg-get parsed "missing" 'fallback)` | `getattr(args, "missing", fallback)` |
| `--help` генерируется автоматически | `argparse` help |

Типы — `:str`/`:int`/`:float`/`:bool`; форма `--имя=значение` работает, флаг `:bool` без значения = `t`; имена в результате хранятся без ведущих дефисов (`--epochs` → `"epochs"`). Результат `parse-args` — алис значений.

## 18.9 Компиляция

| Qualisp | Python-мир |
|---|---|
| `(setq f (jit f))` | CPython 3.13 JIT — copy-and-patch нативной страницы из S-выражения (без IR) |
| `(HLO-START (:psi 3 :buffer_size N) (HLO-COMPILE ...))` | `torch.compile` / `jax.jit` — сессия трассы на тело |
| `(HLO-COMPILE form)` | `torch.compile(fn)` — возвращает программу; вне `HLO-START` требует `(START-TRACE)` в той же форме |
| `(HLO-RUN prog x w)` | `compiled(x, w)` — исполнение программы |
| `(graph->program prog)` | `torch.fx.GraphModule.graph` — мастер дословно, имена сохранены |
| `(program->graph form)` | обратное понижение формы в граф |
| `(graph-data prog)` / `(graph-from-data d)` | `torch.fx` граф как данные + pickle |
| `(dump-graph)` | `print(graph)` — расписание живой сессии трассы (без аргументов; вне трассы — ошибка) |
| `(graph-run-passes prog '(DCE CSE FUSION))` | прогон оптимизационных проходов (и на нативной программе) |
| `(defuse f (x) ...)` / `(defuse! f (x) ...)` | `@torch.compile(fullgraph=True)` / с fallback — составная операция из тела |
| `(deftape! name (p...) :forward f :backward b)` | `torch.autograd.Function` — обратный ход для `defuse`-операции (определён в `core`) |
| `(hlo-route-grad! prog labels)` | backward обучаемой ROUTE-ноды без ленты |
| `(hlo-stencil-p h)` / `(hlo-has-range h)` | интроспекция скомпилированного кёрнела |
| `(jit-stats)` | (нет аналога) — счётчики JIT: компиляции, кэш-хиты, интерпретированные вызовы по категориям, отказы |
| `(target)` | `sys.platform` / `platform.machine()` — дескриптор платформы (алист с `"os"`, `"arch"`, `"features"`) |
| `(target-feature :avx2)` | проверка фичи платформы (детерминированная) |
| `(cfg (:and :linux (:not :avx512)) form...)` | условная компиляция (`#ifdef`): несовпавшая ветка отсекается на раскрытии (фаза компиляции) и не выполняется; условия `:and`/`:or`/`:not` |

```lisp
(defvar x (tensor ((1.0 2.0))))
(defvar w (tensor ((1.0) (1.0))))
(defvar prog
  (HLO-START (:psi 3 :buffer_size 1048576)
    (HLO-COMPILE (MATMUL! (GRAPH-PARAM x) (GRAPH-PARAM w)))))
(print (HLO-RUN prog x w))       ;; → tensor([3], shape=[1, 1], dtype=f32)
(print (GRAPH->PROGRAM prog))
;; → (PROGRAM (:PARAMS ((P0 (1 2) F32) (P1 (2 1) F32))) (MATMUL P0 P1))
(print (graph-data prog))
;; → (GRAPH (ENTRY 2) ((PARAM 0 (1 2) nil nil F32) (PARAM 1 (2 1) nil nil F32)
;;                     (DOT 2 (1 1) (0 1) nil F32)))
(print (PROGRAM->GRAPH (GRAPH->PROGRAM prog)))   ;; → #<hlo-prog>
```

- **Мастер — программа.** `(PROGRAM (:PARAMS ...) тело)`: входы, операции по порядку, `LET*` для fanout, `ROUTE` для ветвления, `FUSED` для слитого блока. `(GRAPH->PROGRAM prog)` отдаёт мастер **дословно**, `(PROGRAM->GRAPH form)` понижает обратно, round-trip не меняет чисел. `(GRAPH ...)` (ENTRY/id/шейпы/рёбра) — производное расписание, а не мастер; шейпы промежуточных значений в программу не входят.
- **`HLO-START`** — special form: конфиг читается одним парсером (`NS-CONFIG`), действие конфига и трассы ограничено телом (save/restore, exception-safe), вложенные `HLO-START` изолированы. Ключи: `:psi` (> 0), `:buffer_size`/`:buffer` (байты, бюджет пути графа), `:verify-gate` (`[0,1]`, NS-слой, гл. 14). Неизвестный ключ — ошибка с именем ключа. `HLO-START` не оставляет сессию открытой после тела.
- **`!`-операции под трассой:** с графовым двойником (`T+!`, `T*!`, `MATMUL!`, …) становятся обычным узлом графа; без двойника (`LOG!` и т.п.) — внятная ошибка с именем операции. Деление двойника не имеет: `T/!` под трассой отвечает `T/! under tracing: division has no graph node (use explicit ops)`; оператор `(/ ...)` — то же ядро, вне трассы оба работают как tape-операции.
- **`TRACE-COMPILE`/`TRACE-RUN`** не существуют: трассировку и исполнение дают `HLO-COMPILE`/`HLO-RUN`, а `START-TRACE` — ручной вход в сессию (открывает её в той же форме, где вызывается `HLO-COMPILE`).
- **`(jit-stats)`** — алист: `compiles`, `cache-hits`, `interpreted` с категориями `sandbox`, `no-page`, `stale-page`, `page-bail`, `dest-store`, и `refusals` с именованными причинами (`special-form`, `setq-captured`, …). Сумма категорий = числу интерпретированных вызовов; `refusals` — список, а не счётчик.

```lisp
(jit-stats)
;; (("compiles" . 19) ("cache-hits" . 0)
;;  ("interpreted" ("sandbox" . 0) ("no-page" . 0) ("stale-page" . 6)
;;                 ("page-bail" . 0) ("dest-store" . 0))
;;  ("refusals"))
```

> 🐍 В PyTorch граф — объект `GraphModule` с отдельным API доступа. В Qualisp программа — обычное S-выражение: её печатают, сравнивают, правят и прогоняют через `graph-run-passes` как форму, а не как чёрный ящик.

> ⚠️ **`hlo-compile` требует сессии.** Вне `HLO-START` — ошибка `hlo-compile: no active trace session (START-TRACE / HLO-START first)`; `graph-data`/`graph->program`/`hlo-stencil-p`/`hlo-has-range` требуют именно программу или форму графа. `dump-graph` вне трассы отвечает `dump-graph: no active trace session (give a program)`.

## 18.10 FFI

| Qualisp | Python-мир |
|---|---|
| `(ffi-load "lib.so")` | `ctypes.CDLL("lib.so")` |
| `(FFI-DEFUN "pow" c-pow "f64" (x: f64 y: f64))` | `ctypes.CFUNCTYPE`: C-имя, лисп-имя, возврат, аргументы — строгие типы в обе стороны |
| `(c-pow 2.0 3.0)` | `pow(2.0, 3.0)` — вызов как обычной функции (работает и из JIT-кода) |
| `(FFI-DEFSTRUCT fs (a: byte b: f64))` | `class fs(ctypes.Structure): _fields_ = [...]` — C-layout: выравнивание и padding как в C |
| `(FFI-ADDR st 'field)` | `ctypes.byref(st.field)` — адрес блока или поля (для out-параметров) |
| `(FFI-OFFSET 'fs 'b)` | `fs.b.offset` → 8 (byte-смещение поля) |
| `(FFI-SIZE 'fs)` | `ctypes.sizeof(fs)` → 16 |
| `(ffi-import "libfoo.so" 'fn)` | `cdll.libfoo.fn` |

> ⚠️ Динамической цепочки `FFI-CALL` нет: рабочий путь FFI — `FFI-LOAD`, `FFI-DEFUN` (объявленные сигнатуры), `FFI-DEFSTRUCT`, `FFI-ADDR`/`FFI-OFFSET`/`FFI-SIZE`, `FFI-IMPORT`. Колбэков C→Lisp во FFI нет — для них есть `qualisp_register_prim` в embedding API. На Windows FFI-трамплин строится по SysV, поэтому `FFI-DEFUN` и вызов падают `EXCEPTION_ACCESS_VIOLATION`, а `FFI-LOAD` работает (гл. 16). Из FFI-примитивов в книге описаны `FFI-LOAD`, `FFI-DEFUN`, `FFI-DEFSTRUCT`, `FFI-ADDR`, `FFI-OFFSET`, `FFI-SIZE`, `FFI-IMPORT`.

> ⚠️ **`.pdata` и исключения.** У JIT-страниц и HLO-ядер нет `.pdata`: исключение не должно пересекать нативный кадр — фасады `call_value`/`rest_list` ловят и возвращают bail-токен. Это инвариант, а не проверка: добавление бросающего фасада уронит Windows, но не Linux.

## 18.10.1 Embedding C API: `qualisp_*` (C зовёт Lisp)

| Qualisp C API (гл. 16) | Python-аналог |
|---|---|
| `qualisp_open` / `qualisp_load_stdlib` / `qualisp_close` | жизненный цикл интерпретатора (встраивание, а не `ctypes`-вызов) |
| `qualisp_eval` / `qualisp_call` | `eval(...)` / вызов функции из C |
| `qualisp_nil` / `qualisp_int` / `qualisp_float` / `qualisp_string` / `qualisp_list` | конструкторы значений (как `ctypes`-обёртки) |
| `qualisp_is_int/float/string/tensor/nil` / `qualisp_as_int/float/string` | `isinstance` / приведение (акессоры `ctypes`) |
| `qualisp_tensor_wrap` / `qualisp_tensor_data` | borrowed-тензор над буфером хоста (`np.ctypeslib.as_array`) |
| `qualisp_set_global` / `qualisp_get_global` | запись/чтение глобалов |
| `qualisp_register_prim` | регистрация C-функции как примитива Lisp (Python C API: `PyCFunction`) |
| `qualisp_error` | текст последней ошибки (C++-исключения не пересекают границу) |

Маршалинг значений — зеркало FFI C: та же таможня, обратная сторона. Результаты `eval`/`call` клонируются в stable (scratch-арена формы умирает по её границе), значения хоста живут до `qualisp_close`.

## 18.11 Параллелизм и дифференцируемые функции

| Qualisp | Python-мир |
|---|---|
| `(future form)` → future | `executor.submit(fn)` — special form: форма исполняется в фоне |
| `(await f)` / `(await-all fs)` | `future.result()` / `[f.result() for f in fs]` |
| `(pmap fn lst)` | `list(executor.map(fn, lst))` — по задаче на элемент |
| `(chan)` / `(chan-put ch v)` / `(chan-take ch)` / `(chan-close ch)` | `queue.Queue()` / `put` / `get` / `close` |
| `(thread-count)` | размер пула `ThreadPoolExecutor` |
| тензорная операция (OpenMP + OpenBLAS внутри) | `torch`-операция над всеми ядрами |
| `(deftape! name (p...) :forward f :backward b)` | `torch.autograd.Function` (`forward`/`backward`) — определён в `core`, доступен и `dl`, и `ml` |

```lisp
(defvar f (future (+ 1 2)))   ;; форма, а не значение
(await f)                     ;; → 3
(await-all (list (future (+ 1 2))))   ;; → (3)
(pmap (lambda (x) (* x 2)) '(1 2))    ;; → (2 4)
```

> 🐍 Мир задачи в Qualisp — **снимок** (глобалы только для чтения, локальные копируются в задачу при её создании), а не общий heap: ближе к `multiprocessing`-модели, но в одном процессе и дешевле. У задачи свой интерпретатор (своя scratch-арена, реестры и лента), поэтому future можно ждать в любой форме.

> ⚠️ **Пул по умолчанию однопоточный.** Ядра заняты тензорными операциями, а параллелизм задач упирается в гонку в общем состоянии интерпретатора: при ≥ 3 воркерах задача с циклом по `let`-переменной может вернуть искажённое значение. `QUALISP_THREADS` поднимает число воркеров — на риск оператора (гл. 8).

## 18.12 Ошибки

| Qualisp | Python |
|---|---|
| `(catch-error f)` | `try/except` — успех: пара `(T . результат)`, ошибка: структура `ERROR` |
| `(error-message e)` | `str(e)` — текст ошибки |
| `(error-cause e)` | `e.__cause__` — ошибка, пойманная под ней (у корневой — `nil`) |
| `(error-traceback e)` | `traceback.extract_tb(e.__traceback__)` — список кадров-структур (`name`/`file`/`line`/`form`) |

```lisp
(catch-error (lambda () (+ 1 2)))       ;; → (T . 3)
(error-message (catch-error (lambda () (tref (randn (list 4)) 4))))
;; → "tref: index 4 out of range for dimension 0 of size 4"
```

Разбор пары: `(car r)` — флаг успеха, `(cdr r)` — значение или структура ошибки. `try`/`catch` в языке — форма с образцом: `(try expr (e handler))` связывает `e` со структурой ошибки и возвращает `handler`; ошибку можно пробросить дальше через `throw`. Проверка аргументов не универсальна: `(car 5)` → `NIL`, ошибку дают примитивы с явной проверкой (`tref`, `tset!`, `matmul`, `print`, `string-view`, …).
