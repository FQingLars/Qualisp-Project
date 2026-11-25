# Глава 2. Начало работы

## 2.1 Установка

### Готовый бинарник (Linux)

Релиз v0.19.20 публикует ассеты на странице релиза; нужно скачать `qualisp`, при необходимости `qprof` и файл контрольных сумм:

```bash
curl -L https://github.com/FQingLars/Qualisp-Project/releases/download/v0.19.20/qualisp-0.19.20-linux-x86_64 -o qualisp
curl -L https://github.com/FQingLars/Qualisp-Project/releases/download/v0.19.20/SHA256SUMS.txt -o SHA256SUMS.txt
sha256sum -c SHA256SUMS.txt
chmod +x qualisp && mv qualisp ~/.local/bin/
```

Состав релиза:

| Файл | Платформа | Состав |
|---|---|---|
| `qualisp-0.19.20-linux-x86_64` | Linux x86_64 | REPL/раннер + JIT |
| `qualisp-0.19.20-windows-x86_64.exe` | Windows x86_64 | REPL/раннер + JIT |
| `qprof-0.19.20-linux-x86_64` | Linux x86_64 | агрегатор Chrome Trace JSON |
| `qprof-0.19.20-windows-x86_64.exe` | Windows x86_64 | агрегатор Chrome Trace JSON |
| `qprof-0.19.20-windows-x86_64.pdb` | Windows x86_64 | символы отладки `qprof` |
| `qualisp-0.19.20-py3-none-linux_x86_64.whl` | Linux x86_64 | Python-пакет: `libqualisp.so` + `stdlib.qlsl` + ctypes-обёртка |
| `qualisp-0.19.20-py3-none-win_amd64.whl` | Windows x86_64 | то же для `qualisp.dll` |
| `SHA256SUMS.txt` | — | контрольные суммы всех файлов |

Python-обёртка ставится из wheel в чистый venv и не требует Zig:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install qualisp-0.19.20-py3-none-linux_x86_64.whl
python - <<'PY'
from qualisp import Session
with Session() as ql:
    print(ql.eval("(+ 1 2)"))      # 3
PY
```

Внутри wheel — `libqualisp` (исполняющий C ABI), `stdlib.qlsl` и модуль `qualisp`; системный `qualisp` для этого не нужен (гл. 16). Для внешних пакетов `Session` принимает `search_path="packages"` или `lock="qualisp.lock"` (одновременно их передавать нельзя), а также методы `add_search_path()` и `load_lock()`.

> ✅ **Проверка установки.** `qualisp --version` печатает `qualisp 0.19.20`; `printf '%s\n' '(+ 1 2)' | qualisp` печатает `3`. Кроме системной `libc`, внешних библиотек не нужно: checker и JIT-эмиттер собраны в одном исполняемом файле.
>
> ⚠️ **Ассеты 0.19.20 собраны в edge-профиле** (`-Dembed-stdlib=false`): в бинарнике только bootstrap `core`+`error`, поэтому `(import string)`/`(import ml)` без пакета дают `UnknownModule`. Возьмите `stdlib.qlsl` из Python-wheel (`qualisp/stdlib.qlsl`) и запускайте `qualisp --package stdlib.qlsl script.qlsp`; либо установите wheel и работайте через Python; либо соберите `qualisp` из исходников обычным `zig build`.

### Готовый бинарник (Windows)

Скачайте `qualisp-0.19.20-windows-x86_64.exe` со страницы релиза; при желании возьмите `qprof-0.19.20-windows-x86_64.exe` и `qprof-0.19.20-windows-x86_64.pdb`. Это те же JIT-страницы SysV, что и на Linux: различия только в механизме W^X (`VirtualAlloc`/`VirtualProtect`), файловых хэндлах и резерве стека в PE.

### Сборка из исходников

Сборка стандартная для Zig-проекта:

```bash
zig build          # zig-out/bin/qualisp
zig build test     # 1095 тестов на Linux
zig build run
zig fmt src
```

Для кросс-проверки под Windows: `zig build test -Dtarget=x86_64-windows-gnu` (тесты запускаются через Wine). Версию инструментария Zig берут из конфигурации исходного дерева; CMake, OpenBLAS, OpenMP и LLVM не нужны — весь нативный код создаётся JIT-эмиттером внутри процесса.

## 2.2 Основные инструменты

| Инструмент | Назначение | Python-аналог |
|---|---|---|
| `qualisp` | REPL + запуск `.qlsp` + JIT | `python` |
| `qprof` | агрегатор self/inclusive времени по трассе | `cProfile` + `pstats` |
| `qualisp-pack` | сборка детерминированного `.qlsl` из манифеста stdlib | `pip wheel`/`zipapp` |
| `libqualisp` + `qualisp` (Python) | встраивание рантайма через C ABI/ctypes | `Py_Initialize`/`ctypes` |

Отдельного компилятора в духе `nuitka` нет: checker и JIT живут в одном бинарнике `qualisp` — это следствие инварианта одной формы (гл. 1.3, 20). Профилировщик — инструмент, а не форма языка: включается он только извне, `qualisp --trace out.json script.qlsp`; форм `profile-start`/`profile-stop`/`PROFILE-ON`/`SERVE` и TCP-отладчика в релизе нет. `qualisp-pack` — build-step (`zig build pack` → `zig-out/share/qualisp/stdlib.qlsl`), а не отдельный менеджер пакетов.

В релизе **нет** `qvalent` и готового language server'а: LSP поставляется в исходниках (`libqualisplsp`, `qualisp-lsp`, расширение VSCodium), но бинарника `qualisp-lsp` в релизных ассетах нет — его собирают локально (гл. 2.7). Исполняющий `libqualisp` — не LSP-библиотека: он входит в wheel и собирается `zig build` (гл. 16).

> 🐍 **Python-аналогия.** Один `qualisp` совмещает роли `python` и встроенного JIT, а `qprof` — роль `pstats`: программа пишет трассу, утилита её разбирает.

## 2.3 REPL

```text
$ qualisp
Qualisp
> (+ 1 2 3 4 5)
15
> (defun sq ((x . :fixnum)) ((* x x) . :fixnum))
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

Ошибки checker'а и reader'а имеют вид `error at LINE:COL: message`, например `error at 1:1: condition must be :bool`. Диагностика пишется в stderr, баннер, промпт и результаты — в stdout. Ошибки времени исполнения часто печатают только `error: RuntimeError`; при запуске файла позиция выводится как `путь:строка:колонка: сообщение` плюс `error: RuntimeError`.

> ⚠️ **Аннотации только парные.** `(defun sq (x :fixnum) :fixnum (* x x))` даёт `error at 1:1: defun expects parameters, a result and a body`, а `(defvar y 20)` — `error at 1:1: defvar expects a name, a typed value and nothing else`. Рабочая запись — `(defun sq ((x . :fixnum)) ((* x x) . :fixnum))` и `(defvar y (20 . :fixnum))` (гл. 3).

> ⚠️ **`quit` и `(exit)` — не команды.** Первое даёт `unbound variable QUIT`, второе — `unknown function EXIT`. Сессию завершает EOF (Ctrl+D). Каждая введённая форма — отдельный scope: временные значения освобождаются после неё.

`core` и `error` — предустановленный prelude: они загружаются и импортируются при старте. Остальные модули подключаются явно через `import`/`import-from` (гл. 2.5).

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
qualisp [--trace <file.json>] [--package <pkg.qlsl>] [--path <dir>]
        [--lock <file>] [--no-lock] [script.qlsp [args...]]
```

Флаги интерпретатора разбираются **только до** имени скрипта; всё после имени скрипта — аргументы программы, доступные через `(command-line-args)`. `--package` и `--path` можно повторять: пакеты подключаются по порядку и участвуют в поиске `import` после встроенных модулей. `--path` задаёт каталог с `*.qlsl` (сканируется лексикографически) или конкретный архив; `QUALISP_PATH` делает то же через переменную окружения. `--lock` загружает `qualisp.lock`, `--no-lock` отключает автопоиск lock-файла рядом со скриптом. `--version` печатает `qualisp 0.19.20`; `--help`, `-h`, `-v`, `--trace`, `--package`, `--path` или `--lock` без значения дают usage и `error: BadArguments` (код выхода 1).

Профилирование:

```bash
qualisp --trace out.json script.qlsp
qprof out.json --top 10
```

`qprof` без аргументов печатает `usage: qprof <trace.json> [--top N]` (код выхода 2); таблица содержит колонки `name / calls / total_us / self_us / self%`. Трасса — Chrome Trace JSON; хуки стоят на границах вызовов, HLO-графов и `ns-if`.

## 2.5 Импорт модулей

В релизе 13 модулей в порядке загрузки: `core`, `error`, `string`, `collections`, `test`, `ml`, `dl`, `symbolic`, `argparse`, `tablet`, `io`, `datetime`, `neuro`. Обычная сборка `zig build` вшивает их в бинарник; `core` и `error` грузятся при старте как prelude, остальные — лениво при первом `import`. Бинарные ассеты 0.19.20 собраны в edge-профиле, поэтому для них нужен `stdlib.qlsl` из wheel или `--package` (см. §2.1). Модуль ищется сначала среди встроенных, затем в подключённых `.qlsl`-пакетах (`--package`, `Session.add_package`, `stdlib.qlsl`). Edge-сборка `-Dembed-stdlib=false` оставляет в бинаре только bootstrap `core`+`error`, а 11 модулей читает из `stdlib.qlsl`; бинарь ищет пакет рядом с собой (`bin/../share/qualisp/stdlib.qlsl`).

```lisp
(import string)                 ; → STRING, печатает имя модуля
(print (string-contains "abc" "b"))   ; → t

(import-from CORE (identity caar))    ; только названные экспорты
(print (identity 5))                  ; → 5
(print (caar '((1 2) 3)))             ; → 1

(import datetime)                     ; → DATETIME
(print (iso-utc 0))                   ; → "1970-01-01T00:00:00Z"
```

`import` открывает все экспорты модуля, `import-from` — только перечисленные; повторный импорт идемпотентен, конфликт имён — `ImportConflict`, неизвестный экспорт — `UnknownExport`, неизвестный модуль — `UnknownModule`. Приватные хелперы модуля недостижимы из исходника.

Пакет `.qlsl` — это ZIP с `manifest.qlsp` и исходниками модулей; порядок файлов и CRC32 детерминированы. Подключение:

```bash
qualisp --package ./my-lib.qlsl app.qlsp
qualisp --path ./packages app.qlsp          # все *.qlsl из каталога
QUALISP_PATH=./packages:./vendor qualisp app.qlsp
```

Модуль из пакета загружается лениво при первом `import`, а исходник после загрузки освобождается. Собрать пакет из манифеста можно `qualisp-pack` (в дереве исходников — `zig build pack`). Повторное подключение того же архива идемпотентно; другой пакет с тем же именем даёт `DuplicatePackage`, отсутствующий путь — `MissingPackage`.

Для воспроизводимого набора пакетов есть lock-файл:

```bash
qualisp-pack lock qualisp.lock packages/*.qlsl
qualisp --lock qualisp.lock app.qlsp
qualisp app.qlsp                 # qualisp.lock рядом со скриптом — автоматически
qualisp --no-lock app.qlsp       # отключить автоподхват
```

`qualisp.lock` хранит относительные пути, имена, версии и SHA-256 всех пакетов. Перед применением проверяются **все** записи; несовпадение хэша — `LockMismatch`, отсутствие файла — `MissingPackage`, битый формат — `BadLock`. Когда lock активен, поисковые пути и соседний edge-`stdlib.qlsl` не используются: набор пакетов фиксирован (явные `--package` при этом всё ещё разрешены). `QUALISP_LOCK` задаёт lock-файл через окружение, `--no-lock` отключает и его автоподхват.

> 🐍 **Python-аналогия.** `(import string)` ≈ `import string`, а `(import-from CORE (identity caar))` ≈ `from core import identity, caar`. Модулей `regex`, `audio`, `pkg`, `ns`, `ffi` в stdlib нет: `FFI` и `NS` при импорте дают `UnknownModule`. Модуль `error` уже входит в prelude, а `datetime` подключается явно через `(import datetime)`.

## 2.6 JIT: когда нужен нативный код

Отдельной команды «скомпилировать функцию» в релизе нет: тело `defun` компилируется при определении (`emitDefun` строит `FunctionTemplate` с готовым `entry`), а top-level форма — сразу.

```lisp
(defun sq ((x . :fixnum)) ((* x x) . :fixnum))
(sq 4)          ; → 16: тело уже скомпилировано при определении
(sq 5)          ; → 25: исполняется уже скомпилированная страница
```

Мастер-форма функции хранится в `Func`; страница — выбрасываемый кэш, который можно пересобрать из S-выражения. Вызовы компилируются по calling convention `entry(frame, env, rt, out)`: кал сам пишет результат в `*out`, что позволяет делать хвостовые вызовы без роста кадра (гл. 20). Форм `jit`/`discard-jit` и автокомпиляции по порогу горячих вызовов в релизе нет.

Параметры и `let`-биндинги с доказанным `:fixnum` живут в raw-слотах кадра без тега: `emitRaw` считает арифметику и сравнения над ними, а боксинг происходит на границе — при вызове, возврате и захвате замыканием. С 0.19.17 обязательные параметры `:fixnum` **и** `:flonum` занимают raw-слот (`:flonum` — SSE2 `movsd`/`addsd`/…), `&optional`/`&rest` остаются боксованными, а `:fixnum → :flonum` расширяется через `cvtsi2sd`. `tail_self` копирует аргументы в `param_slots`, поэтому raw-параметр обновляется и цикл сходится.

Вызов глобальной функции эмитится инлайном: разрешение `Func`, проверка формы, bump кадра, копии аргументов и `call [func.entry]` вместо хелпера `ql_call_global → callFunc`. Методы по тегу, `&optional`/`&rest`, хвостовой вызов, активный профайлер и переполнение кадра уходят в generic-путь. Traceback стал ленивым: `FrameInfo` хранит только имя и адрес сайта, а `file:line:column` резолвится при ошибке; адрес патчится до W^X. В цикле без вызовов fixnum-глобал заменяется raw-значением с одной записью назад. Замеры ReleaseFast: fib (N=27) 42.2 → **25.8 мс**, вызов в локальном цикле 26.6 → **13.9 нс**, глобальный арифметический цикл 17.6 → **10.4 нс**.

Тензорный код компилируется по тому же принципу, но через HLO: `defgraph` трассирует тело и компилирует граф при первом вызове, `defuse` — явное fused-ядро. Всё это происходит внутри работающего процесса; отдельного офлайн-компилятора нет. Вычислительные ядра (`elementwise`/`map`, batched `matmul`, `reshape`/`transpose`, осевые `tsum`/`tmean`/`tmax`, `softmax`, `broadcastRows`/`reduceRows`, conv/pool, NS-роутинг) исполняются на пуле параллельных ядер `kernel_pool` (до 8 воркеров), давая побайтово тот же результат, что и последовательный проход; `QUALISP_SERIAL=1` отключает пул. `conv2d`/pooling параллелятся по batch (backward — двумя проходами: `dx` по batch, `dw` по выходным каналам).

> 🐍 **Python-аналогия.** Роль, которую в CPython играет JIT 3.13, здесь выполняет JIT copy-and-patch: компилируются не байткод-хот-циклы, а тела функций языка; управляет процессом рантайм, а не программист. Параллельные ядра близки по смыслу к `multiprocessing` с разделением диапазонов буфера, но без отдельного интерпретатора воркера.

## 2.7 Редактор и средства разработки

LSP поставляется в исходниках и собирается локально. Синтаксис, типы и диагностика живут в Zig-компиляторе (`src/lsp_abi.zig`, библиотека `libqualisplsp` с C ABI); Rust-сервер `tools/qualisp-lsp/` (`qualisp-lsp`) — только протокольная оболочка, без второго ридера или чекера. Анализатор не исполняет формы верхнего уровня документа: `read → macroexpand → typecheck`.

```bash
bash tools/lsp/build.sh                    # zig-out/lsp/libqualisplsp.* + qualisp-lsp
node editors/vscode/test/smoke.js          # проверка расширения против сервера
python3 editors/vscode/package-vsix.py     # самодостаточный zig-out/qualisp.vsix
```

Сервер даёт `publishDiagnostics` (в том числе диагностику старого синтаксиса аннотаций с позицией), `definition` по `.qlsp`-файлам воркспейса и встроенным модулям, `hover` со статическим типом и исходной формой определения вместе с комментарием над ней, `documentSymbol`, `completion`, `documentHighlight` и `semanticTokens`. Расширение для VSCodium/VS Code на чистом JS без npm-зависимостей (`editors/vscode/`) ищет сервер в `zig-out/lsp`.

> ⚠️ **Честно про релиз.** Готового бинарника `qualisp-lsp`, библиотеки `libqualisplsp` и `.vsix` в релизных ассетах нет — LSP не «устанавливается из релиза», а собирается из исходников. Это не относится к исполняющему `libqualisp`: он лежит внутри Python-wheel (и собирается `zig build`), но LSP его не использует.

Без сборки LSP доступны следующие средства:

- **Диагностика с позицией.** Ошибки checker'а печатают `error at LINE:COL: message` в stderr, ошибки файла — `путь:строка:колонка: message`: этого достаточно, чтобы редактор или терминал указали на место ошибки.
- **REPL как инспектор.** Можно определить форму, тут же её вызвать и увидеть значение или отказ с диагностикой; definition печатает имя символа.
- **Профиль.** `qualisp --trace out.json script.qlsp` + `qprof out.json --top N` дают таблицу вызовов и собственного времени по именам функций.
- **Исходные карты.** Внутри JIT-страниц есть `MapEntry{start,end,span}`, поэтому runtime-ошибка получает позицию в исходнике; пошагового отладчика с точками останова при этом нет.

> 🐍 **Python-аналогия.** Собранный language server даёт привычные Pyright/Pylance-подсказки (definition, hover, completion); без него подсветку обеспечивает редактор, а сигнал об ошибках и их местоположении — сам интерпретатор и `pstats`.

## 2.8 Первая ML-программа

```lisp
;; first-ml.qlsp: восстановить c в y = c*x^2
(defvar xs ((tensor (0.5 1.0 1.5 2.0)) . (:tensor :f32 (4))))
(defvar ys ((tensor (0.5 2.0 4.5 8.0)) . (:tensor :f32 (4))))

(defvar c ((param (tensor (0.5))) . (:tensor :f32 (1))))   ; обучаемый вес
(defvar lr (0.05 . :flonum))

(defgraph loss ((c . (:tensor :f32 (1))) (x . (:tensor :f32 (4))) (y . (:tensor :f32 (4))))
  ((let ((d (- (* c (* x x)) y))) (tmean (* d d))) . :flonum))

(defvar epoch (0 . :fixnum))
(while (< epoch 200)
  (grad! (loss c xs ys))
  (tset! c 0 (- (tref c 0) (* lr (tref (grad-of c) 0))))
  (setq epoch (+ epoch 1)))

(print (tref c 0))     ; → 2.0
```

Здесь `defgraph` — мастер-форма графа: параметры пишутся как `(имя . :тип)`, а результат — последняя форма тела в записи `(выражение . :тип)`. Тело трассируется один раз, `param` помечает `c` как обучаемый, `grad!` считает градиент по последнему запуску графа, `grad-of` читает накопленный градиент, а `tset!` обновляет вес на месте. Все шаги цикла лежат внутри одной top-level формы `while`, поэтому последний запуск графа остаётся доступным для `grad!`. Не пугайтесь деталей — тензоры и autograd разобраны в главах 6–7, HLO — в главе 10.

> 🐍 **Python-аналогия.** Это `loss.backward()` + ручной `w -= lr * w.grad` внутри цикла, только граф собирается из `defgraph`, а не из невидимого autograd-движка.

## 2.9 Куда смотреть дальше

- Синтаксис языка целиком — глава 3; типы — глава 4.
- Тензоры и autograd — главы 6–7; HLO-графы — глава 10; JIT — глава 20.
- Образы и персистенция — глава 21.
- Нейросимвольный слой — глава 14.
- Аргументы командной строки: `(command-line-args)` и модуль `argparse` — главы 3, 15.
- FFI: `ffi-load`/`ffi-defun` — глава 16.
- Шпаргалка «Qualisp ↔ Python» — глава 18.
