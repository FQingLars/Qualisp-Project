# Глава 13. ООП

В Qualisp две ООП-системы: лёгкие записи (`defstruct`) и CLOS-подобный MOP (`defclass`, `defgeneric`/`defmethod`).

## 13.1 defstruct: структуры-записи

```lisp
(defstruct Point (x y))

(defvar p (make-instance 'Point 'x 10 'y 20))
(slot-value p 'x)          ;; → 10
```

Реализация: `do_defstruct` (`src/eval/interpreter.cpp:1268`) разворачивает `(defstruct Name (slot1 slot2 ...))` в серию определений: класс `Name`, конструктор `make-name`, и сеттеры/геттеры. По синтаксису Qualisp не имеет отдельного `make-name`/`name-x` — используется универсальный `(make-instance 'Name ...)` и `(slot-value obj 'slot)`.

> 🐍 **Python-аналогия.** Это `@dataclass`:
> ```python
> @dataclass
> class Point:
>     x: int
>     y: int
> p = Point(x=10, y=20); p.x
> ```

`defstruct` автоматически создаёт конструктор и слоты; экземпляры — обычные значения (живут по правилам HibLin, гл. 5; `SExpr::INSTANCE`-тег в `sexpr.hpp`).

> ⚠️ **Инстансы сравниваются по идентичности (v2.8.4).** `(equal p p)` — истинно (рефлексивность), `(equal p q)` для двух разных инстансов — `nil`, даже при одинаковых слотах: у мутабельного объекта со слотами равенство живёт по адресу, как `a is b` в Python. До v2.8.4 `equal` не перечислял тег `INSTANCE` и возвращал `nil` даже для самого себя (гл. 3.7).

## 13.2 defclass: классы со слотами

```lisp
(defclass Animal ()
  (name species))

(defvar cat (make-instance 'Animal 'name "Whiskers" 'species "cat"))
(slot-value cat 'name)     ;; → "Whiskers"
```

Реализация (`src/eval/interpreter.cpp:1230`): `do_defclass` регистрирует `ClassObj` в `ClassRegistry::instance()` (см. `src/mop/CODE.md`). Класс может наследовать суперклассам через список в первом аргументе: `(defclass Dog (Animal) (breed))`. Слоты — `SlotDef { name, initform }`.

> 🐍 ≈ обычный `class Animal: __init__(self, name, species)`.

Установка слота: `(setf (slot-value obj 'field) value)` или через низкоуровневый `Instance::set_slot` (используется внутри `defstruct`-макросов).

## 13.3 Дженерики: defgeneric + defmethod

Полиморфизм по классу аргумента — через обобщённые функции (CLOS-стиль):

```lisp
(defgeneric area (shape))

(defmethod area ((shape Circle))    (* 3.14159 (slot-value shape 'r) (slot-value shape 'r)))
(defmethod area ((shape Rect))      (* (slot-value shape 'w) (slot-value shape 'h)))

(area circle1)    ;; диспетчеризация по типу первого аргумента
(area rect1)
```

Реализация: `do_defgeneric` (`src/eval/interpreter.cpp:1301`) регистрирует generic function, `do_defmethod` (`src/eval/interpreter.cpp:1311`) привязывает методы к специализациям (по классу первого аргумента).

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
| `(defclass B (A) ...)` | `class B(A):` | наследование |
| `(make-instance 'A 'slot v)` | `A(slot=v)` | конструктор |
| `(slot-value obj 'slot)` | `obj.slot` | доступ к полю |
| `(defgeneric f (x))` + `(defmethod f ((x A)))` | `singledispatch` | полиморфизм по типу первого аргумента |

## 13.6 Известные ограничения MOP

Из `src/mop/CODE.md`:

- **Method combinations** (`:before`/`:after`/`:around`) — не реализованы.
- **`change-class`** и **метаклассы** — не реализованы.
- Диспетчеризация `defmethod` — по типу **первого** аргумента; многоаргументная диспетчеризация — в плане.

Тем не менее базовый CLOS-цикл (`defclass`/`make-instance`/`slot-value`/`defgeneric`/`defmethod`) работает и покрыт тестами.