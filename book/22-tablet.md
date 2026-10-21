# Глава 22. Tablet: DataFrame-аналог pandas

В stdlib есть **Tablet** — модуль табличных данных в духе pandas: типизированные колонки, NA-дисциплина, стабильные сортировки, groupby, merge, CSV/JSON I/O. ~1300 строк чистого Qualisp поверх примитивов — т.е. модуль сам написан на языке (гл. 15), а всё табличное хозяйство доступно из REPL сразу.

> 🐍 **Python-аналогия.** `import tablet` ≈ `import pandas as pd`: `Tablet` ≈ `DataFrame`, `tcol` ≈ `df["col"]`, `tgroup-agg` ≈ `groupby().agg()`, `tmerge` ≈ `merge`. Разница: колонки — списки или тензоры Qualisp, NA — обычный `nil`, и весь модуль детерминирован до байта.

## 22.1 Контракт детерминизма

Единственное жёсткое правило модуля (TODO_TABLET.md §0): **вывод не зависит ни от чего, кроме входных данных** — порядок колонок везде порядок вставки, сортировки только стабильные (merge sort), равные ключи сохраняют исходный порядок строк, NaN при сортировке всегда в конец, ключи groupby отсортированы, внутри группы — исходный порядок строк, `tvalue-counts` — по убыванию с ничьими в порядке первого появления. Никаких unordered-структур в путях вывода. Это делает Table-пайплайны воспроизводимыми побитово — двойной прогон даёт идентичный мир (проверено тестом `double-run` в `tests/tablet.qlsp`).

## 22.2 Быстрый старт

Конструктор — список пар «имя колонки + список значений»; порядок пар = порядок колонок:

```lisp
(import TABLET)

(defvar t (tablet (list
  (list "city" (list "MSK" "SPB" "MSK" "KZN"))
  (list "age"  (list 33 25 41 19)))))

(tshape t)        ;; → (4 2)   — строки, колонки
(tcols t)         ;; → ("city" "age")
(thead t 2)       ;; → (TABLET ("city" "age") (("MSK" "SPB") (33 25)))
(trow t 1)        ;; → ("SPB" 25)
```

Альтернатива — построение из строк (`tablet-from-rows`, имена по умолчанию `"0".."n-1"`, как `RangeIndex`-колонки pandas).

## 22.3 Выбор и фильтрация

```lisp
(tcol t "age")                    ;; → (33 25 41 19)      df["age"]
(tcols-pick t (list "age" "city")) ;; колонки в указанном порядке (reindex)
(tloc t (list 0 2))               ;; строки по меткам индекса
(tiloc t 0 1)                     ;; срез по позициям
(twhere t (list t nil nil t))     ;; булева маска          df[mask]
```

Отсутствующая колонка бросает ошибку (как `KeyError`). Фильтрация через `twhere`:

```lisp
(tcol (twhere t (map (\ (a) (> a 30)) (tcol t "age"))) "city")
;; → ("MSK" "MSK" "KZN")
```

## 22.4 Сортировка и уникальность

```lisp
(tsort t "age")                  ;; по возрастанию, стабильная, NA — в конец
(tsort t "age" 'desc)            ;; по убыванию
(tsort t (list "name" "w"))      ;; multi-key
(tunique t "city")               ;; уникальные (порядок первого появления)
(tvalue-counts t "city")         ;; → (("MSK" 2) ("SPB" 1) ("KZN" 1))
```

`tvalue-counts` — по убыванию частот; ничья сохраняет порядок первого появления, NA исключаются (`dropna`-семантика).

## 22.5 NA-дисциплина

NA — это `nil`. Числовая колонка с NA остаётся колонкой; столбец из одних NA получает dtype `empty`:

```lisp
(defvar tb (tablet (list (list "x" (list 1.0 nil 3.0)))))
(tdtypes tb)                     ;; → (FLOAT)
(tcol (tfillna tb 0.0) "x")      ;; → (1.0 0.0 3.0)       fillna(0.0)
(tfillna tb nil 'ffill)          ;; разнос вперёд         method='ffill'
(tdropna tb)                     ;; выбросить строки с NA
```

## 22.6 Группировка и агрегации

```lisp
(tgroupby t "city")              ;; ключи отсортированы, NA-ключ отброшен
(tgroup-agg (tgroupby t "city") (list (list "age" 'mean)))
;; → таблица: KZN→19, MSK→37, SPB→25 (ключи по возрастанию)
```

Агрегации: `sum`, `mean`, `count`, `min`, `max`, `first`, `last`; несколько пар «колонка + функция» за один вызов. Мультиключ: `(tgroupby t (list "k1" "k2"))`.

## 22.7 Соединения

```lisp
(tmerge ta tb "k")               ;; inner (по умолчанию)
(tmerge ta tb "k" 'left)         ;; порядок левых ключей сохранён
(tmerge ta tb "k" 'right)        ;; порядок правых
(tmerge ta tb "k" 'outer)        ;; левые ключи, затем новые правые
(tappend ta tb)                  ;; concat по строкам
```

Порядок результата в каждом режиме зафиксирован контрактом §22.1 (§0 TODO_TABLET) — как `how='left'` у pandas сохраняет порядок левых строк.

## 22.8 Типы, категории, строки

```lisp
(tdtypes t)                      ;; → ('string 'int) по колонкам
(tastype t "age" 'float)         ;; → (33.0 25.0 41.0 19.0)  astype
(tastype col 'string)            ;; печатное представление
(tcategorical t "city")          ;; коды: (0 1 0 2) + уровни по первому появлению
(tstr-upper t "city")            ;; строковые аксессоры, NA сохраняются
(tstr-contains t "city" "S")
(tiso-format t "ts")             ;; epoch-колонка → ISO-строка (UTC, гражданский календарь)
```

`tastype` парсит строки в числа (`"2.5"` → `2.5`), неразбираемое значение → NA, неизвестный тип — ошибка.

## 22.9 IO: CSV, JSON, тензоры

```lisp
(tcsv-write t "/tmp/demo-t.csv")         ;; RFC-4180 (кавычки, переводы строк в полях)
(tshape (tcsv-read "/tmp/demo-t.csv"))   ;; → (4 2)
```

`tcsv-read` **типизирует колонки сам**: целая колонка → int, с точкой → float, смешанная → строки, пустое поле → NA. Круговой рейс `tcsv-write` → `tcsv-read` идентичен исходной таблице — тесты гоняют его с кавычками, запятыми и переводами строк внутри полей. `tcsv-read` (он зовёт `read-csv` из модуля `io`) держит **продакшен-размерные** CSV на дефолтной глубине `eval`: парсер итеративен (гл. 15.3). На закоммиченных в репозиторий датасетах (`tests/data/housing.csv`, `moons.csv`, `wine13.csv`) грузятся обученные модели.

```lisp
(tjson-rows t)
;; → [{"city":"MSK","age":33},{"city":"SPB","age":25},...]   orient=records
(tfrom-json-rows "[{\"a\":1},{\"a\":2,\"b\":\"x\"}]")  ;; обратно в Tablet

(tcol->tensor t "age")           ;; числовая колонка без NA → Tensor F32 (ML-пайпы)
```

## 22.10 Статистика и окна

```lisp
(tcol-sum t "age") (tcol-mean t "age") (tcol-min ...) (tcol-max ...) (tcol-count ...)
(tcov t "x" "y") (tcorr t "x" "y")     ;; парная ковариация/корреляция, NA попарно
(tcumsum t "v") (tcummax ...) (tcummin ...)   ;; NA пропускается, счёт идёт
(trank t "v")                          ;; средние ранги ничьих ('average), NA не ранжируется
(tshift t "v" 1) (tdiff t "v" 1)       ;; сдвиг и разность (лаг ≥ 0)
(trolling t "v" 3 'mean)               ;; скользящее окно (nil 1.5 2.5) для окна 2
(texpanding t "v" 'sum)                ;; расширяющееся окно
```

## 22.11 Чего в Tablet нет (честно)

- **MultiIndex** — явно вне области видимости: индекс — плоский список меток, дубликаты в `loc`-пути запрещены.
- **pivot/melt** — в плане (T3-карта TODO_TABLET), не реализованы.
- Денежных типов, datetime64-арифметики — только `tiso-format` поверх epoch-колонок.

Тесты: `tests/tablet.qlsp` — 156 проверок (deftest), включая батарею детерминизма T9: двойной прогон, стабильность сортировок на дубликатах, порядок ключей groupby, порядок merge во всех режимах, тайбрейки `tvalue-counts`, неизменяемость исходной таблицы (immutability chain).
