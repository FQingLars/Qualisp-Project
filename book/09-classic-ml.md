# Глава 9. Классический ML (модуль ML)

Модуль `ML` — встроенный «мини-scikit-learn»: утилиты, ядра, метрики, препроцессинг, KNN, наивный Байес, деревья, ансамбли, K-Means, кросс-валидация и grid search. Всё реализовано на самом Qualisp (см. `stdlib/ml.qlsp`, ~337 строк) — хороший материал для изучения языка.

```lisp
(import ML)
```

## 9.1 Утилиты для списков

```lisp
(range n)                       ;; → (0 1 ... n-1)        range(n)
(range n start step)            ;; → (start start+step ...), start/step с &optional
(take 2 '(1 2 3))               ;; → (1 2)                lst[:2]
(drop 2 '(1 2 3))               ;; → (3)                  lst[2:]
(unique '(1 1 2))               ;; → (1 2)                set-подобно
(mean '(1 2 3))                 ;; → 2                    statistics.mean
(zip-with + '(1 2) '(10 20))    ;; поэлементный zip
(map fn xs)                     ;; → список результатов fn
(filter pred xs)                ;; → список x где (pred x)
(reduce + '(1 2 3 4))           ;; → 10
(majority-vote xs)              ;; → большинство (для ансамблей)
```

## 9.2 Ядра (kernels)

Имена из `stdlib/ml.qlsp`:

```lisp
(linear-kernel x1 x2)                       ;; ⟨x1, x2⟩
(poly-kernel x1 x2 degree coef0)            ;; (⟨x1,x2⟩+coef0)^degree
(rbf-kernel x1 x2 gamma)                    ;; exp(-γ‖x1−x2‖²)
(sigmoid-kernel x1 x2 alpha coef0)          ;; tanh(⟨x1,x2⟩·alpha+coef0)
```

> 🐍 Это ядра из `sklearn.metrics.pairwise` / `sklearn.svm`. NB: `sigmoid-kernel` использует имена `alpha coef0`, а не `a c`.

Дистанции:

```lisp
(euclidean-dist a b)         ;; sqrt(Σ(aᵢ-bᵢ)²)
(manhattan-dist a b)         ;; Σ|aᵢ-bᵢ|
```

## 9.3 Метрики

```lisp
(accuracy-score y-pred y-true)                  ;; → float [0,1]
(precision-score y-pred y-true pos-label)        ;; → float [0,1]
(recall-score y-pred y-true pos-label)           ;; → float [0,1]
(f1-score y-pred y-true pos-label)               ;; → float [0,1]
(confusion-matrix y-pred y-true labels)          ;; → матрица n×n
```

Аргумент `pos-label` — метка «позитивного» класса (символ/значение), обязательна для precision/recall/f1.

> 🐍 `sklearn.metrics.accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`.

## 9.4 Препроцессинг и разбиение

```lisp
(train-test-split data target ratio)    ;; → (train-d test-d train-t test-t)
(standard-scale-lst data)               ;; z-score     StandardScaler
(minmax-scale-lst data)                 ;; [0,1]       MinMaxScaler
```

NB: `train-test-split` в `stdlib/ml.qlsp` принимает данные и таргеты **отдельными аргументами** (см. исходник):

```lisp
(let ((split (train-test-split X y 0.8)))
  (let ((train-x (car split)) (test-x (cadr split))
        (train-y (caddr split)) (test-y (cadddr split)))
    ...))
```

> 🐍 `train_test_split(X, y, train_size=0.8)`, `StandardScaler().fit_transform`, `MinMaxScaler()`.

## 9.5 K-ближайших соседей

```lisp
;; x — точка, train-data — список векторов, train-labels — список меток
(defvar pred (knn-predict x train-data train-labels k))
```

Реализация (`stdlib/ml.qlsp:108`): считает евклидовы расстояния до всех точек, сортирует индексы (`argsort` через insert-sort), возвращает топ-k голосов.

> 🐍 `KNeighborsClassifier(n_neighbors=k).fit(...).predict([x])`.

## 9.6 Наивный Байес (гауссов)

```lisp
(defvar model (nb-fit X y))          ;; → (classes priors stats)
(defvar pred (nb-predict x model))   ;; → argmax класса
```

> 🐍 `GaussianNB().fit(X, y).predict([x])`.

## 9.7 Деревья решений

```lisp
(defvar tree (build-tree data labels max-depth))     ;; дерево
(defvar pred (tree-predict tree x))                   ;; предсказание
```

Жадный поиск сплита по критерию Джини: `best-split` перебирает фичи и пороги, выбирает сплит с минимальным Gini impurity.

## 9.8 Случайный лес и градиентный бустинг

```lisp
;; Случайный лес: бутстрэп + голосование деревьев
(defvar forest (random-forest-fit data labels n-trees max-depth))
(defvar pred (random-forest-predict forest x))

;; Градиентный бустинг: пни (stumps) на остатках
(defvar model (gb-fit data labels n-estimators lr max-depth))   ;; → (init trees)
(defvar pred (gb-predict model x))
```

> 🐍 `RandomForestClassifier(n_estimators=n_trees, max_depth=...)`, `GradientBoostingClassifier(n_estimators=n, learning_rate=lr, max_depth=...)`.

NB: порядок аргументов в `stdlib/ml.qlsp:234` — `(data labels n-trees max-depth)` и `(data labels n-estimators lr max-depth)`. В Python sklearn — те же имена.

## 9.9 K-Means

```lisp
(defvar centroids (kmeans-fit data k iters))   ;; список k центроидов
;; отнести точку к кластеру:
(defvar cluster (nearest-centroid x centroids 0 0 -1))
```

> 🐍 `KMeans(n_clusters=k, max_iter=iters).fit(data).cluster_centers_`.

## 9.10 Кросс-валидация и подбор гиперпараметров

```lisp
;; k-fold CV: model-fn(train-x train-y) → модель с вызовом (model x)
(defvar score (cross-val-score X y k fit-fn score-fn))

;; перебор параметров
(defvar best (grid-search X y param-list fit-fn score-fn))
```

`fit-fn` — функция вида `(lambda (tx ty) (lambda (x) (knn-predict x tx ty 3)))`.
`score-fn` — `(lambda (preds y-true) (accuracy-score preds y-true))`.

С v2.7.1 `grid-search` оценивает каждый вариант на всём `X` (исправлен баг: скорость считалась по несуществующему `test-x`).

> 🐍 `cross_val_score(model, X, y, cv=k)` и `GridSearchCV(...)`.

### Потери и активации над списками (v2.7.1)

В `ml` добавлен список-ориентированный слой потерь — удобно для символьных/нейросимвольных пайплайнов, где значения ходят списками, а не тензорами:

```lisp
(softmax-list '(1.0 2.0 3.0))       ;; → (0.0900305... 0.2447284... 0.6652409...) (max-shift, стабильно)
(log-softmax-list '(1.0 2.0 3.0))   ;; лог-вероятности без overflow
(mse-loss '(1.0 2.0) '(1.5 2.5))    ;; → 0.25
(mae-loss '(1.0 2.0) '(1.5 2.5))    ;; → 0.5
(cross-entropy-list pred target)    ;; цели — one-hot или мягкие метки
(one-hot 1 3)                       ;; → (0.0 1.0 0.0)  (k n): единица на позиции k
```

> 🐍 `torch.nn.functional.softmax(..., dim=-1)` (max-shift внутри), `nn.MSELoss`, `nn.CrossEntropyLoss` — только над списками, а не тензорами.

## 9.11 End-to-end пример

```lisp
(import ML)

;; датасет: 2 класса, 2 признака (данные — списки)
(defvar data (list (list 1.0 2.0) (list 1.5 1.8) (list 5.0 8.0)
                   (list 6.0 9.0) (list 1.2 0.6) (list 5.5 8.2)))
(defvar labels (list 'a 'a 'b 'b 'a 'b))

;; KNN с кросс-валидацией (k=3)
(defun fit-knn (tx ty)
  (lambda (x) (knn-predict x tx ty 3)))

(defvar cv (cross-val-score data labels 3 fit-knn accuracy-score))
(print cv)
```

> 🐍 Эквивалент:
> ```python
> from sklearn.neighbors import KNeighborsClassifier
> from sklearn.model_selection import cross_val_score
> model = KNeighborsClassifier(n_neighbors=3)
> cross_val_score(model, X, y, cv=3).mean()
> ```

## 9.12 Отличия от scikit-learn

- API функциональный: `(nb-fit ...)` вместо `.fit()`/`.predict()` классов.
- Данные — списки Qualisp (для табличных данных), тензоры — для нейросетей.
- Реализации учебно-минималистичные: глядя в `stdlib/ml.qlsp`, вы видите каждый алгоритм целиком — это одновременно и документация, и курс по ML-алгоритмам.
- `unique` дедуплицирует, `train-test-split` делит с заданным ratio; `cross-val-score` возвращает среднее по k фолдам.