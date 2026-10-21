# Глава 13. ООП

В Qualisp две ООП-системы: лёгкие записи (`defstruct`) и CLOS-подобный MOP (`defclass`, `defgeneric`/`defmethod`).

## 13.1 defstruct: структуры-записи

```lisp
(defstruct Point (x y))

(defvar p (make-instance 'Point 'x 10 'y 20))
(slot-value p 'x)          ;; → 10
```

Реализация: `do_defstruct` (`src/eval/interpreter.cpp`) разворачивает `(defstruct Name (slot1 slot2 ...))` в определение класса со слотами. Отдельных `make-point`/`point-x` в Qualisp нет — конструктор и доступ общие: `(make-instance 'Name 'slot v)` и `(slot-value obj 'slot)`.

> 🐍 **Python-аналогия.** Это `@dataclass`:
> ```python
> @dataclass
> class Point:
>     x: int
>     y: int
> p = Point(x=10, y=20); p.x
> ```

Слот заполняется при создании инстанса: `(make-instance 'Point 'x 10 'y 20)`. Отдельной формы для установки значения слота в Lisp нет — внутри интерпретатора слот ставится низкоуровневым `Instance::set_slot`.

Инстансы — обычные значения (живут по правилам HibLin, гл. 5; тег `SExpr::INSTANCE` в `sexpr.hpp`).

> ⚠️ **Инстансы сравниваются по идентичности.** `(equal p p)` — истинно (рефлексивность), `(equal p q)` для двух разных инстансов — `nil`, даже при одинаковых слотах: у мутабельного объекта со слотами равенство живёт по адресу, как `a is b` в Python. Для чисел, cons-ячеек, строк и символов `=`/`equal` сравнивают по значению (гл. 3).

## 13.2 defclass: классы со слотами

```lisp
(defclass Animal ()
  (name species))

(defvar cat (make-instance 'Animal 'name "Whiskers" 'species "cat"))
(slot-value cat 'name)     ;; → "Whiskers"
```

Реализация (`src/eval/interpreter.cpp`): `do_defclass` регистрирует `ClassObj` в `ClassRegistry::instance()` (см. `src/mop/CODE.md`). Класс может наследовать суперклассам через список в первом аргументе: `(defclass Dog (Animal) (breed))`. Слоты — `SlotDef { name, initform }`.

> 🐍 ≈ обычный `class Animal: __init__(self, name, species)`.

Слот наследуемого класса в слоте доступа не находится: `(make-instance 'Dog 'name "Rex")` даёт ошибку «set-slot: no slot NAME» — слоты суперкласса надо объявлять в самом классе. Это относится и к `defstruct`-записям: у них нет наследования вовсе.

## 13.3 Дженерики: defgeneric + defmethod

Полиморфизм по классу аргумента — через обобщённые функции (CLOS-стиль):

```lisp
(defgeneric area (shape))

(defclass Circle () (r))
(defclass Rect () (w h))

(defmethod area ((shape Circle))    (* 3.14159 (slot-value shape 'r) (slot-value shape 'r)))
(defmethod area ((shape Rect))      (* (slot-value shape 'w) (slot-value shape 'h)))

(area (make-instance 'Circle 'r 2))    ;; → 12.56636
(area (make-instance 'Rect 'w 2 'h 3)) ;; → 6
```

Диспетчеризация идёт **по типу первого аргумента**. Если подходящего метода нет — внятная ошибка «No applicable method for AREA», а не случайный вызов.

> 🐍 Python-аналог — `functools.singledispatch`:
> ```python
> @singledispatch
> def area(s): ...
> @area.register
> def _(s: Circle): return 3.14159 * s.r ** 2
> ```

## 13.4 Функциональный стиль: слои вместо классов

Важно понимать идиоматику ML-кода Qualisp: **модели обычно пишут не классами, а замыканиями** (гл. 8):

```lisp
(defvar model (linear 4 2))     ;; замыкание с параметрами внутри
(model x)
```

Классы уместны для **данных** (датасеты, конфиги, узлы деревьев), замыкания — для **поведения** (слои, модели). Это противоположно PyTorch, где `nn.Module` — класс.

## 13.5 Сводная таблица

| Qualisp | Python | Комментарий |
|---|---|---|
| `(defstruct Point (x y))` | `@dataclass class Point:` | записи |
| `(defclass A () (slot1 slot2))` | `class A:` | классы |
| `(defclass B (A) ...)` | `class B(A):` | наследование объявляется; слоты суперкласса в доступе не находятся |
| `(make-instance 'A 'slot v)` | `A(slot=v)` | конструктор |
| `(slot-value obj 'slot)` | `obj.slot` | доступ к полю |
| `(defgeneric f (x))` + `(defmethod f ((x A)))` | `singledispatch` | полиморфизм по типу первого аргумента |

## 13.6 Известные ограничения MOP

- **Method combinations** (`:before`/`:after`/`:around`) — не реализованы.
- **`change-class`** и **метаклассы** — не реализованы.
- **Наследование слотов** — не работает: суперкласс объявляется, но его слоты в `make-instance`/`slot-value` не находятся.
- Диспетчеризация `defmethod` — по типу **первого** аргумента; многоаргументная диспетчеризация не реализована.

Тем не менее базовый CLOS-цикл (`defclass`/`make-instance`/`slot-value`/`defgeneric`/`defmethod`) работает и покрыт тестами.
