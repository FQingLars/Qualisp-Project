# Глава 2. Начало работы

## 2.1 Установка

### Готовый бинарник (Linux)

Релиз v0.19.15 публикует ассеты на странице релиза; нужно скачать `qualisp`, при необходимости `qprof` и файл контрольных сумм:

```bash
curl -L https://github.com/FQingLars/Qualisp-Project/releases/download/v0.19.15/qualisp-0.19.15-linux-x86_64 -o qualisp
curl -L https://github.com/FQingLars/Qualisp-Project/releases/download/v0.19.15/SHA256SUMS.txt -o SHA256SUMS.txt
sha256sum -c SHA256SUMS.txt
chmod +x qualisp && mv qualisp ~/.local/bin/
```

Состав релиза:

| Файл | Платформа | Состав |
|---|---|---|
| `qualisp-0.19.15-linux-x86_64` | Linux x86_64 | REPL/раннер + JIT |
| `qualisp-0.19.15-windows-x86_64.exe` | Windows x86_64 | REPL/раннер + JIT |
| `qprof-0.19.15-linux-x86_64` | Linux x86_64 | агрегатор Chrome Trace JSON |
| `qprof-0.19.15-windows-x86_64.exe` | Windows x86_64 | агрегатор Chrome Trace JSON |
| `qprof-0.19.15-windows-x86_64.pdb` | Windows x86_64 | символы отладки `qprof` |
| `SHA256SUMS.txt` | — | контрольные суммы всех файлов |

> ✅ **Проверка установки.** `qualisp --version` печатает `qualisp 0.19.15`; `printf '%s\n' '(+ 1 2)' | qualisp` печатает `3`. Кроме системной `libc`, внешних библиотек не нужно: checker и JIT-эмиттер собраны в одном исполняемом файле.

### Готовый бинарник (Windows)

Скачайте `qualisp-0.19.15-windows-x86_64.exe` со страницы релиза; при желании возьмите `qprof-0.19.15-windows-x86_64.exe` и `qprof-0.19.15-windows-x86_64.pdb`. Это те же JIT-страницы SysV, что и на Linux: различия только в механизме W^X (`VirtualAlloc`/`VirtualProtect`), файловых хэндлах и резерве стека в PE.

### Сборка из исходников

Сборка стандартная для Zig-проекта:

```bash
zig build          # zig-out/bin/qualisp
zig build test     # 834 теста на Linux
zig build run
zig fmt src
```

Для кросс-проверки под Windows: `zig build test -Dtarget=x86_64-windows-gnu` (тесты запускаются через Wine). Версию инструментария Zig берут из конфигурации исходного дерева; CMake, OpenBLAS, OpenMP и LLVM не нужны — весь нативный код создаётся JIT-эмиттером внутри процесса.

## 2.2 Один интерпретатор и одна утилита

| Инструмент | Назначение | Python-аналог |
|---|---|---|
| `qualisp` | REPL + запуск `.qlsp` + JIT | `python` |
| `qprof` | агрегатор self/inclusive времени по трассе | `cProfile` + `pstats` |

Отдельного компилятора в духе `nuitka` нет: checker и JIT живут в одном бинарнике `qualisp` — это следствие инварианта одной формы (гл. 1.3, 20). Профилировщик — инструмент, а не форма языка: включается он только извне, `qualisp --trace out.json script.qlsp`; форм `profile-start`/`profile-stop`/`PROFILE-ON`/`SERVE` и TCP-отладчика в релизе нет.

В релизе **нет** `qualisp-lsp` и `qvalent`: language server и пакетный менеджер в состав не входят.

> 🐍 **Python-аналогия.** Один `qualisp` совмещает роли `python` и встроенного JIT, а `qprof` — роль `pstats`: программа пишет трассу, утилита её разбирает.

## 2.3 REPL

```text
$ qualisp
Qualisp
> (+ 1 2 3 4 5)
15
> (defun sq (x :fixnum) :fixnum (* x x))
SQ
> (sq 7)
49
> (car 5)
error at 1:1: car expects a list
> (/ 1 0)
error: RuntimeError
>
```

Баннер — `Qualisp`, промпт — `> `, продолжение многострочной формы — `... `. Префикса `=>` нет: REPL печатает значение формы как есть. Формы `defun`/`defvar` печатают имя символа (`SQ`, `Y`), а не результат.

Ошибки checker'а и reader'а имеют вид `error at LINE:COL: message`, например `error at 1:1: condition must be :bool`. Ошибки времени исполнения часто печатают только `error: RuntimeError`; при запуске файла позиция выводится как `путь:строка:колонка: сообщение` плюс `error: TypeError`.

> ⚠️ **`quit` и `(exit)` — не команды.** Первое даёт `unbound variable QUIT`, второе — `unknown function EXIT`. Сессию завершает EOF (Ctrl+D). Каждая введённая форма — отдельный scope: временные значения освобождаются после неё.

`core` — предустановленный prelude: он загружается и импортируется при старте. Остальные модули подключаются явно через `import`/`import-from` (гл. 2.5).

## 2.4 Запуск файла

```lisp
;; hello.qlsp
(print "Hello, Qualisp!")
(print (+ 40 2))
```

```text
$ qualisp hello.qlsp
Hello, Qualisp!
42
```

Файловый раннер не печатает результат последней формы — только явный `print`. Синтаксис вызова:

```text
qualisp [--trace <file.json>] [script.qlsp [args...]]
```

Флаги интерпретатора разбираются **только до** имени скрипта; всё после имени скрипта — аргументы программы, доступные через `(command-line-args)`. `--version` печатает `qualisp 0.19.15`; `--help`, `-h`, `-v` и `--trace` без файла дают usage и `error: BadArguments` (код выхода 1).

Профилирование:

```bash
qualisp --trace out.json script.qlsp
qprof out.json --top 10
```

`qprof` без аргументов печатает `usage: qprof <trace.json> [--top N]` (код выхода 2); таблица содержит колонки `name / calls / total_us / self_us / self%`. Трасса — Chrome Trace JSON; хуки стоят на границах вызовов, HLO-графов и `ns-if`.

## 2.5 Импорт модулей

В релизе 11 модулей: `core`, `string`, `collections`, `test`, `io`, `ml`, `dl`, `symbolic`, `neuro`, `argparse`, `tablet`. Все они — Qualisp-код, вшитый в бинарник; `core` грузится при старте, остальные — лениво при первом `import`.

```lisp
(import string)                 ; → STRING, печатает имя модуля
(print (string-contains "abc" "b"))   ; → t

(import-from CORE (identity caar))    ; только названные экспорты
(print (identity 5))                  ; → 5
(print (caar '((1 2) 3)))             ; → 1
```

`import` открывает все экспорты модуля, `import-from` — только перечисленные; повторный импорт идемпотентен, конфликт имён — `ImportConflict`, неизвестный экспорт — `UnknownExport`, неизвестный модуль — `UnknownModule`. Приватные хелперы модуля недостижимы из исходника.

> 🐍 **Python-аналогия.** `(import string)` ≈ `import string`, а `(import-from CORE (identity caar))` ≈ `from core import identity, caar`. Модулей `regex`, `audio`, `datetime`, `pkg`, `ns`, `ffi` в stdlib нет: `FFI` и `NS` при импорте дают `UnknownModule`.

## 2.6 JIT: когда нужен нативный код

Отдельной команды «скомпилировать функцию» в релизе нет: тело `defun` компилируется при определении (`emitDefun` строит `FunctionTemplate` с готовым `entry`), а top-level форма — сразу.

```lisp
(defun sq (x :fixnum) :fixnum (* x x))
(sq 4)          ; → 16: тело уже скомпилировано при определении
(sq 5)          ; → 25: исполняется уже скомпилированная страница
```

Мастер-форма функции хранится в `Func`; страница — выбрасываемый кэш, который можно пересобрать из S-выражения. Вызовы компилируются по calling convention `entry(frame, env, rt, out)`: кал сам пишет результат в `*out`, что позволяет делать хвостовые вызовы без роста кадра (гл. 20). Форм `jit`/`discard-jit` и автокомпиляции по порогу горячих вызовов в релизе нет.

Тензорный код компилируется по тому же принципу, но через HLO: `defgraph` трассирует тело и компилирует граф при первом вызове, `defuse` — явное fused-ядро. Всё это происходит внутри работающего процесса; отдельного офлайн-компилятора нет.

> 🐍 **Python-аналогия.** Роль, которую в CPython играет JIT 3.13, здесь выполняет JIT copy-and-patch: компилируются не байткод-хот-циклы, а тела функций языка; управляет процессом рантайм, а не программист.

## 2.7 Редактор и средства разработки

Language server в состав релиза не входит: `qualisp-lsp` не поставляется. Файлы `.qlsp` — обычный текст, поэтому подсветку синтаксиса можно настроить в редакторе как для Лиспа/Common Lisp (`;` — комментарий, префиксная нотация, скобки). Никаких дополнительных расширений или серверных протоколов для работы с кодом не требуется.

Что доступно вместо навигации по проекту:

- **Диагностика с позицией.** Ошибки checker'а печатают `error at LINE:COL: message`, ошибки файла — `путь:строка:колонка: message`: этого достаточно, чтобы редактор или терминал указали на место ошибки.
- **REPL как инспектор.** Можно определить форму, тут же её вызвать и увидеть значение или отказ с диагностикой; definition печатает имя символа.
- **Профиль.** `qualisp --trace out.json script.qlsp` + `qprof out.json --top N` дают таблицу вызовов и собственного времени по именам функций.
- **Исходные карты.** Внутри JIT-страниц есть `MapEntry{start,end,span}`, поэтому runtime-ошибка получает позицию в исходнике; пошагового отладчика с точками останова при этом нет.

> 🐍 **Python-аналогия.** Это как работать с `python` без language server: подсветку даёт редактор, а сигнал об ошибках и их местоположении — сам интерпретатор и `pstats`.

## 2.8 Первая ML-программа

```lisp
;; first-ml.qlsp: восстановить c в y = c*x^2
(defvar xs (:tensor :f32 (4)) (tensor (0.5 1.0 1.5 2.0)))
(defvar ys (:tensor :f32 (4)) (tensor (0.5 2.0 4.5 8.0)))

(defvar c (:tensor :f32 (1)) (param (tensor (0.5))))   ; обучаемый вес
(defvar lr :flonum 0.05)

(defgraph loss (c (:tensor :f32 (1)) x (:tensor :f32 (4)) y (:tensor :f32 (4))) :flonum
  (let ((d (- (* c (* x x)) y)))
    (tmean (* d d))))

(defvar epoch :fixnum 0)
(while (< epoch 200)
  (grad! (loss c xs ys))
  (tset! c 0 (- (tref c 0) (* lr (tref (grad-of c) 0))))
  (setq epoch (+ epoch 1)))

(print (tref c 0))     ; → 2.0
```

Здесь `defgraph` — мастер-форма графа: тело трассируется один раз, `param` помечает `c` как обучаемый, `grad!` считает градиент по последнему запуску графа, `grad-of` читает накопленный градиент, а `tset!` обновляет вес на месте. Все шаги цикла лежат внутри одной top-level формы `while`, поэтому последний запуск графа остаётся доступным для `grad!`. Не пугайтесь деталей — тензоры и autograd разобраны в главах 6–7, HLO — в главе 10.

> 🐍 **Python-аналогия.** Это `loss.backward()` + ручной `w -= lr * w.grad` внутри цикла, только граф собирается из `defgraph`, а не из невидимого autograd-движка.

## 2.9 Куда смотреть дальше

- Синтаксис языка целиком — глава 3; типы — глава 4.
- Тензоры и autograd — главы 6–7; HLO-графы — глава 10; JIT — глава 20.
- Образы и персистенция — глава 21.
- Нейросимвольный слой — глава 14.
- Аргументы командной строки: `(command-line-args)` и модуль `argparse` — главы 3, 15.
- FFI: `ffi-load`/`ffi-defun` — глава 16.
- Шпаргалка «Qualisp ↔ Python» — глава 18.
