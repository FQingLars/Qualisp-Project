# Глава 18. Шпаргалка: Qualisp ↔ Python

Быстрый перевод между Qualisp и Python-стеком (NumPy / PyTorch / scikit-learn). Имена примитивов в Qualisp после `SymbolTable::intern` хранятся в верхнем регистре (`T+`, `MATMUL!`, `MSE!`, `PARAM`, `GRAD!`); пользователь может писать их в любом регистре.

## 18.0 Гомоиконность

| Qualisp | Python |
|---|---|
| `(read-from-string "(+ 1 2)")` | `ast.literal_eval("(1 + 2)")` / `eval("(1+2)", {})` |
| `(write-to-string expr)` | `repr(expr)` / `ast.unparse(ast.parse(...))` |
| `(eval expr)` | `eval(...)` (но нативно над S-выражениями) |
| `(functionp 'f)` | `callable(f)` |
| `(function-params 'f)` | `inspect.signature(f).parameters` |
| `(function-body 'f)` | `inspect.getsource(f)` (но структурно) |
| `(function-env 'f)` | `f.__closure__` |
| `(symbol-name 'x)` | `str(x)` (но для имён) |
| `(boundp 'x)` | `'x' in globals()` |
| `(jit f)` / `(setq f (jit f))` | CPython 3.13 JIT (copy-and-patch, тоже без IR в рантайме) — гл. 20 |
| `(jit-sig f)` / `(jit-stale-p f)` | (нет аналога) — наблюдаемые типы параметров JIT |
| `(save-image "w.qimg")` / `(load-image ...)` | `pickle(globals())` + `torch.save` в одном файле — гл. 21 |
| `(discard-jit)` | (нет аналога) — сброс всех нативных страниц, поведение не меняется |
| `(set-reader-macro '!' f)` | (нет аналога) |
| `(randint 5 10)` | `random.randint(5, 10)` |
| `(import-from (ml a b))` | `from ml import a, b` |
| `(defvar_autograd w (randn ...))` | `w = nn.Parameter(torch.randn(...))` |

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
| `(if c a b)` | `a if c else b` |
| `(cond (c1 r1) (c2 r2) (t r))` | `if/elif/else` |
| `(while c body)` | `while c:` |
| `(progn a b c)` | тело функции / блок |
| `; комментарий` | `# комментарий` |
| `'(1 2 3)` | `[1, 2, 3]` (литерал) |
| `(= a b)` | `a == b` — числа сравниваются **по значению**: `(= 0 0.0)` → `t` |
| `(equal a b)` | `a == b` (cons/строки/символы — структурно; замыкания, HLO/JIT-программы, инстансы, тензоры — по идентичности, как `a is b`) |
| `(print (strcat "x " (to-string 42)))` | `print("x", 42)` — `print`/`display` принимают **ровно один аргумент** |
| `(type-of x)` | `type(x).__name__` (в Qualisp — uppercase) |

## 18.2 Списки

| Qualisp | Python |
|---|---|
| `(car xs)` / `(cdr xs)` | `xs[0]` / `xs[1:]` |
| `(cons x xs)` | `[x] + xs` |
| `(nth i xs)` | `xs[i]` |
| `(length xs)` | `len(xs)` |
| `(map f xs)` | `list(map(f, xs))` |
| `(filter p xs)` | `list(filter(p, xs))` |
| `(reduce + xs)` | `functools.reduce(op.add, xs)` |
| `(zip-with f xs ys)` | `[f(a,b) for a,b in zip(xs,ys)]` |
| `(member x xs)` | `x in xs` |
| `(range 5)` | `range(5)` / `list(range(5))` |
| `(take n xs)` / `(drop n xs)` | `xs[:n]` / `xs[n:]` |
| `(reverse xs)` | `xs[::-1]` |
| `(append xs ys)` | `xs + ys` |

## 18.3 Тензоры (NumPy)

| Qualisp | NumPy |
|---|---|
| `(tensor ((1.0 2.0) (3.0 4.0)))` | `np.array([[1.,2.],[3.,4.]])` |
| `(zeros (list 2 3))` | `np.zeros((2,3))` |
| `(ones (list 4 2))` | `np.ones((4,2))` |
| `(randn (list 100 1))` | `np.random.randn(100,1)` |
| `(t+ a b)` / `(t* a b)` | `a + b` / `a * b` |
| `(matmul a b)` | `a @ b` |
| `(reshape t (list 4 1))` | `t.reshape(4,1)` |
| `(transpose t)` | `t.T` |
| `(tsum t)` / `(tsum t (list 0))` | `t.sum()` / `t.sum(axis=0)` |
| `(tmean t)` | `t.mean()` |
| `(argmax t)` | `t.argmax()` |
| `(tensor-shape t)` | `t.shape` |
| `(tensor-dtype t)` | `t.dtype` (`F32`/`F16`/`F64`/`I32`/`I64`/`U8`) |
| `(tsin t)` / `(tcos t)` / `(t/tanh t)` | `np.sin(t)` / `np.cos(t)` / `np.tanh(t)` |
| `(concat a b)` / `(stack a b)` | `np.concatenate([a,b])` / `np.stack([a,b])` |
| `(tensor-f16 t)` / `(tensor-f32 t)` / `(tensor-u8 t)` | `t.astype(np.float16/float32/uint8)` |
| `(tref t i)` | `t.flatten()[i]` — один индекс = плоский номер элемента |
| `(tref t i j)` | `t[i, j]` — N индексов = координаты ровно ранга N; выход за границы — ошибка |
| `(tensor-item t i)` | `t.flatten()[i]` — элемент по индексу (индекс обязателен) |
| `(tensor->list t)` | `t.ravel().tolist()` |
| `(FINITE? t)` | `np.isfinite(t).all()` — битовая проверка конечности (NaN/Inf) |
| `(save-npy t "f.npy")` / `(load-npy "f.npy")` | `np.save` / `np.load` |

## 18.4 Автоград (PyTorch)

| Qualisp | PyTorch |
|---|---|
| `(param (randn (list 4 2)))` | `torch.randn(4,2, requires_grad=True)` |
| `(matmul! x w)` | `x @ w` (в графе) |
| `(relu! h)` | `F.relu(h)` |
| `(mse! pred y)` | `F.mse_loss(pred, y)` |
| `(cross-entropy! logits y)` | `F.cross_entropy(logits, y)` |
| `(grad! loss)` | `loss.backward()` |
| `(grad-of w)` | `w.grad` |
| `(sgd-step w lr)` | SGD `opt.step()` (на одном парам.) |
| `(adam-step w lr)` | Adam `opt.step()` |
| обычные `matmul`, `t+` (без `!`) | `with torch.no_grad():` |
| `(sample logits)` | `torch.distributions.Categorical(logits).sample()` (one-hot) |
| `(reinforce-loss! logits idx reward)` | REINFORCE: `reward * F.cross_entropy(logits, idx)` |

## 18.5 Слои (DL-модель ↔ torch.nn)

| Qualisp | PyTorch |
|---|---|
| `(linear 4 2)` (из DL) | `nn.Linear(4, 2)` (но слой — cons, а не объект) |
| `(list (linear 1 4) 'relu (linear 4 1))` | `nn.Sequential(nn.Linear(1,4), nn.ReLU(), nn.Linear(4,1))` |
| `(net-forward model x)` | `model(x)` |
| `(train-step model x y 'sgd lr)` | `loss.backward(); opt.step()` |
| `(train-epochs model x y 'adam lr 100)` | `for _ in range(100): train_step(...)` |
| `(layer-w layer)` / `(layer-b layer)` | `layer.weight` / `layer.bias` |
| `(batchnorm! h)` / `(layernorm! h)` | `nn.BatchNorm1d` / `nn.LayerNorm` |
| `(dropout! h 0.5)` | `nn.Dropout(0.5)` |
| `(conv2d! x k s p)` | `F.conv2d(x, k, stride=s, padding=p)` |
| `(maxpool2d! x k s)` | `F.max_pool2d(x, k, s)` |
| `(weighted-lookup probs v1 v2)` | взвешенная сумма (скаляр) |
| `(net-infer model x)` | `with torch.no_grad(): model(x)` (tape-free, для `while`) |

> 🐍 Ключевое отличие: Qualisp DL v2 — это **данные** (cons-список), а не объект. Модель сериализуется QSRD v2 как обычная структура. PyTorch-модель — объект с `state_dict`, требует отдельного протокола сериализации.

## 18.5.1 Строки (модуль `string` ↔ `str`)

| Qualisp | Python |
|---|---|
| `(string-length s)` | `len(s)` |
| `(string-split s ",")` | `s.split(",")` |
| `(string-join list ", ")` | `", ".join(list)` |
| `(string-upcase s)` | `s.upper()` |
| `(string-downcase s)` | `s.lower()` |
| `(string-trim s)` | `s.strip()` |
| `(string-replace s "a" "b")` | `s.replace("a", "b")` |
| `(string-index s "x")` | `s.find("x")` |
| `(string-reverse s)` | `s[::-1]` |
| `(char-code #\A)` | `ord("A")` |
| `(code-char 65)` | `chr(65)` |
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

## 18.6 Классический ML (sklearn)

| Qualisp | scikit-learn |
|---|---|
| `(range 5)` / `(range 5 10 2)` | `range(5)` / `range(5, 10, 2)` |
| `(train-test-split data target ratio)` | `train_test_split(train_size=ratio)` |
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
| `(linear-kernel a b)` / `(rbf-kernel a b g)` | kernels из `sklearn.metrics.pairwise` |
| `(softmax-list xs)` / `(log-softmax-list xs)` | `torch.softmax` / `log_softmax` (стабильный max-shift) |
| `(mse-loss pred target)` / `(mae-loss ...)` / `(cross-entropy-list ...)` | `nn.MSELoss` / `nn.L1Loss` / `nn.CrossEntropyLoss` |
| `(one-hot k n)` | `one_hot` / `np.eye(n)[k]` |
| `(is-fixnum? v)` / `(is-flonum? v)` / `(is-string? v)` / `(is-tensor? v)` | `isinstance(v, ...)` |
| `(sigmoid-kernel a b alpha c)` | `tanh(α·⟨a,b⟩+c)` |

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

## 18.8.1 QSRD v2 — бинарная сериализация

| Qualisp | Python |
|---|---|
| `(qsrd-save "f.qsrd" obj)` | `pickle.dump(obj, f)` |
| `(qsrd-load "f.qsrd")` | `obj = pickle.load(f)` |
| `(qsrd-get "f.qsrd" 'field)` | `obj.field` (если QSRD — структура) |

> 🐍 `pickle` теряет dtype/shape тензоров (нужен отдельный `torch.save` + `np.save`). QSRD v2 хранит всё вместе: S-выражения, тензоры с dtype и шейпом (на **любой глубине** структуры), скаляры — в одном файле с magic `"QSRD"` и u16-версией.

## 18.9 Компиляция

| Qualisp | Python-мир |
|---|---|
| `(setq f (jit f))` | CPython 3.13 JIT — copy-and-patch нативной страницы из S-выражения (без IR) |
| `(start-trace)` + `(graph-param x)` + `(hlo-compile out)` + `(hlo-run hlo x ...)` | `torch.compile` / `jax.jit` |
| `(defuse! f (x) ...)` | `@torch.compile(fullgraph=True)` |
| `(defuse f (x) ...)` | `@torch.compile` (с fallback) |
| `(graph-data prog)` / `(graph-from-data d)` | `torch.fx` граф как данные + pickle |
| `(graph-run-passes prog '("cse" "dce" "fusion"))` | прогон оптимизационных проходов |
| `(hlo-route-grad! prog labels)` | backward обучаемой ROUTE-ноды без ленты |
| `(hlo-stencil-p h)` / `(hlo-has-range h)` | интроспекция скомпилированного кёрнела |

## 18.10 FFI

| Qualisp | Python-мир |
|---|---|
| `(ffi-load "lib.so")` | `ctypes.CDLL("lib.so")` |
| `(ffi-call "sin" "float" 0.0)` | `lib.sin(0.0)` |
| `(ffi-call "atan2" "float" 1.0 1.0)` | `lib.atan2(1.0, 1.0)` |
| `(ffi-call "strlen" "int" "hello")` | `lib.strlen(b"hello")` |
| `(ffi-import "libfoo.so" 'fn)` | `cdll.libfoo.fn` |

## 18.11 Параллелизм и дифференцируемые функции

| Qualisp | Python-мир |
|---|---|
| `(async-run form)` → ручка | `executor.submit(fn)` |
| `(async-wait handle)` | `future.result()` |
| `(async-collect handles)` | `[f.result() for f in futures]` |
| `(async-map fn lst)` | `list(executor.map(fn, lst))` |
| `(async-chan)` / `(async-send ch v)` / `(async-recv ch)` / `(async-close ch)` | `queue.Queue()` / `put` / `get` / `task_done` |
| `(async-threads)` | размер пула `ThreadPoolExecutor` |
| тензорная операция (OpenMP + OpenBLAS внутри) | `torch`-операция над всеми ядрами |
| `(deftape! name (p...) :forward f :backward b)` | `torch.autograd.Function` (`forward`/`backward`) |

> 🐍 Мир задачи в Qualisp — **снимок** (глобалы только для чтения, локальные копируются в задачу), а не общий heap: ближе к `multiprocessing`-модели, но в одном процессе и дешевле.