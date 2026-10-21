# Глава 2. Начало работы

## 2.1 Установка

### Готовый бинарник (Linux)

```bash
curl -L https://github.com/FQingLars/Qualisp-Project/releases/download/v2.9.0/qualisp-linux-x86_64 -o qualisp
chmod +x qualisp && mv qualisp ~/.local/bin/
```

Бинарники (`qualisp`, `qualisp-lsp`, `qvalent`) публикуются в релизах этого репозитория и собираются CI из исходников компилятора (Linux x86_64; Windows — MSYS2 MinGW64). Контрольные суммы — в файле `SHA256SUMS` рядом с ассетами релиза.

> ✅ **Рантайм (Linux).** Бинарник **не линкуется с LLVM** — динамические зависимости скромные: `libopenblas`, `libgomp`, `libstdc++`, `libc` (плюс `libgfortran` косвенно, через OpenBLAS). Достаточно `apt install libopenblas0` (или эквивалента). Проверка: `echo '(+ 1 2)' | qualisp`.

### Готовый бинарник (Windows)

Скачайте `qualisp-windows-x86_64.zip` со страницы релизов и распакуйте; `qualisp.exe`, `qualisp-lsp.exe`, `qvalent.exe` готовы к работе (MinGW-сборка, x86_64; CI зелёный на Linux и Windows).

### Сборка из исходников

Исходники компилятора распространяются с деревом разработки и в открытый репозиторий не публикуются — если у вас есть исходное дерево (например, вместе с доступом к разработке), сборка стандартная:

```bash
cd <корень исходников>
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Требования: GCC 13+ / Clang 16+ (C++20), CMake 3.16+, OpenBLAS, OpenMP. **LLVM не нужен** — весь нативный код (JIT замыканий и HLO-кёрнелы) строится собственными стенсил-эмиттерами (гл. 10, 20). GPU-бэкенда в проекте нет: вычисления идут на CPU (гл. 11).

Кросс-компиляция вспомогательных бинарников (`qualisp-lsp`/`qvalent`) под Windows с Linux-машины через MinGW описана в `README.md` исходного дерева (`cmake/x86_64-w64-mingw32.cmake`); для `qualisp` корректный путь — нативная MSYS2-сборка (см. гл. 19).

## 2.2 Три инструмента

| Инструмент | Назначение | Python-аналог |
|---|---|---|
| `qualisp` | REPL + интерпретатор `.qlsp` + JIT (нативный код на лету) | `python` |
| `qualisp-lsp` | Language Server (VS Code, neovim, Emacs) | pylance |
| `qvalent` | Пакетный менеджер (`qvalent.toml`) | `pip` |

Отдельного компилятора нет: интерпретатор и JIT живут в одном бинарнике `qualisp` — это следствие «инварианта одной формы» (гл. 1.3, 20). Все три инструмента собираются из одних исходников; `qualisp-lsp` — отдельная цель только из `src/lsp/server.cpp`.

## 2.3 REPL

```text
$ qualisp
Qualisp
Enter expressions, press Ctrl+D to exit.

> (+ 1 2 3 4 5)
=> 15
> (defun square (x: Float) -> Float (* x x))
=> square
> (square 4.0)
=> 16.0
```

REPL при старте **автоматически загружает все 13 модулей стандартной библиотеки** (см. `embedded_modules[]` в `src/main.cpp`): `core`, `string`, `dl`, `ns`, `ml`, `io`, `regex`, `audio`, `datetime`, `pkg`, `errors`, `visual`, `tablet` (гл. 22). Они вшиты в бинарник через `cmake/stdlib_embed.cmake` (на этапе сборки `.qlsp` превращаются в `.h` через `cmake/gen_stdlib_header.cmake` и компилируются в бинарник). Поддержан частичный импорт: `(import-from (ml knn-predict nb-fit))`.

> 🐍 **Python-аналогия.** Это как если бы `python` стартовал с предимпортированными `numpy`, `torch`, `sklearn`, `re`, `datetime`, `time`, `urllib`, `tensorboard`, `str`.

Многострочный ввод работает по скобкам и кавычкам: незакрытый `(` или незакрытая `"` продолжают ввод. Состояние «мы внутри строки» — отдельный флаг, поэтому REPL не зависает, если строковый литерал стоит в первой строке формы, и перенос строки внутри строки его не сбрасывает.

После каждого выражения REPL вызывает `Interpreter::reset_scratch()` — disposable-область сбрасывается, буферы тензоров и cons-ячейки возвращаются в пулы (гл. 5).

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
:) | Successfully ran hello.qlsp
```

## 2.5 Импорт модулей

Модули стандартной библиотеки уже загружены в REPL, но в коде принято объявлять импорты явно (повторный импорт — дешёвый no-op; `Interpreter::mark_module_loaded()`):

```lisp
(import ML)        ;; классический ML: KNN, деревья, метрики, кросс-валидация
(import IO)        ;; файлы: slurp, spit, read-lines, read-csv
(import REGEX)     ;; регулярные выражения: re-find, re-replace, re-split
(import AUDIO)     ;; FFT, STFT, окна, mel-шкала, hz-to-mel
(import DATETIME)  ;; время: now-ts, now-iso-str, sleep, elapsed
(import PKG)       ;; пакетный менеджер: pkg-install из GitHub
(import ERRORS)    ;; ошибки: make-error, throw, assert, try
(import VISUAL)    ;; дашборд метрик (TensorBoard-подобный HTTP)
```

В исходниках есть также модуль `DL` (~100 строк в `stdlib/dl.qlsp`: `linear`, `net-forward`, `train-step`, `train-epochs`) — он вшит в бинарник наравне с остальными и доступен через `(import DL)`. Модель — cons-список, слои — данные, а не замыкания.

`(import M)` возвращает `"m-loaded"` и гарантирует, что модуль загружен однократно. Для частичного импорта: `(import-from (dl linear train-step))` поднимает только указанные имена.

## 2.6 JIT: когда нужен нативный код

Отдельного компилятора в духе `nuitka` в Qualisp нет — вместо него нативный код рождается **внутри работающего процесса**. Горячее замыкание компилируется в нативную RX-страницу методом copy-and-patch одним вызовом:

```lisp
(defun add (a b) (+ a b))
(setq add (JIT add))    ;; скомпилировать тело в нативную страницу
(add 2 3)               ;; → 5 — исполняется нативный код
```

`(JIT f)` возвращает обёртку-программу, внутри которой живёт нативная страница, а оригинальное S-выражение — **мастер-копия**: её можно в любой момент выбросить (`DISCARD-JIT`) и поведение не изменится, потому что страница — лишь кэш (гл. 20). Тензорный код компилируется через HLO-пайплайн тем же принципом (гл. 10).

> 🐍 **Python-аналогия.** Это роль, которую в Python играет CPython 3.13 JIT — тоже copy-and-patch, тоже без IR в рантайме, — только в Qualisp компилируются не байткод-хот-циклы, а замыкания языка целиком, и результатом управляет сам программист.

Если нужен «самодостаточный бинарник» — сохраните образ мира (`SAVE-IMAGE`, гл. 21) и загрузите его в `qualisp` на целевой машине: S-выражения и тензоры переедут, нативные страницы пересоберутся лениво при первом вызове.

## 2.7 Редактор и LSP

`qualisp-lsp` — Language Server без внешних зависимостей (JSON-парсер встроен, компиляторный движок не линкуется). Это полноценный сервер:

- **Диагностики**: незакрытая форма (с позицией открывающей скобки), лишняя `)`, незакрытая строка — точные диапазоны, инкрементальная синхронизация документа (`change: 2`).
- **Навигация**: goto-definition (в т.ч. cross-file), references, documentHighlight, rename по всему воркспейсу, `workspace/symbol`, `documentSymbol` (14 def-видов: `defun`/`defmacro`/`defvar`/`defhloop`/… + `deftest`).
- **Комфорт**: folding ranges (пары скобок на разных строках), форматирование с Lisp-отступами (тело `defun`/`let`/`if` = col+2, аргументы выравниваются после головы).
- Автодополнения (~140 встроенных символов с hover-документацией по примитивам).

- **VS Code**: любой LSP-клиент, команда запуска `qualisp-lsp`.
- **neovim**: `nvim-lspconfig` с custom config.
- **Emacs**: `lsp-mode` / `eglot`.

Тот же движок работает в пакетном режиме: `python3 tools/qlsp-check.py FILE.qlsp` — диагностика читателя (незакрытые/лишние скобки) и структурный план `defun`-дерева (`--symbols`). Для C++-исходников компилятора навигация идёт через `clangd` по `build/compile_commands.json`.

## 2.8 Первая ML-программа

```lisp
;; Первый градиентный спуск
(defvar x-data (tensor ((1.0) (2.0) (3.0) (4.0))))
(defvar y-data (tensor ((2.0) (4.0) (6.0) (8.0))))   ;; y = 2x

(defvar w (param (randn (list 1 1))))                ;; обучаемый вес
(defvar lr 0.01)

(defun train-step ()
  (let ((pred (matmul! x-data w)))     ;; forward
    (let ((loss (mse! pred y-data)))   ;; loss
      (grad! loss)                     ;; backward
      (sgd-step w lr)                  ;; обновить веса
      loss)))

(defvar epoch 0)
(while (< epoch 100)
  (let ((loss (train-step)))
    (if (equal (mod epoch 20) 0) (print loss)))
  (setq epoch (+ epoch 1)))
```

Не пугайтесь деталей — `param`, `matmul!`, `grad!` разобраны в главах 6–7. Полные рабочие примеры — в главе 17.

## 2.9 Куда смотреть дальше

- Синтаксис языка целиком — глава 3.
- Тензоры и автоград — главы 6–7.
- HLO-стенсил-компиляция — глава 10, JIT замыканий — глава 20, образы — глава 21.
- Нейросимвольный слой — глава 14.
- Шпаргалка «Qualisp ↔ Python» — глава 18.