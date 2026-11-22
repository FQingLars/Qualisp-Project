# 📖 Qualisp Book

**Книга по языку программирования Qualisp** — статически типизированному Лиспу для машинного обучения и AI.

Qualisp — диалект Лиспа с полной статической типизацией, JIT-компиляцией прямо по S-выражениям (copy-and-patch в нативные страницы, без промежуточного IR и без AOT), встроенными тензорами, HLO-графами с автоматическим дифференцированием и нейросимвольной логикой `ns-if`. Реализация — на Zig, один бинарник, без внешних зависимостей для запуска.

Свойства, которые отличают Qualisp от обычного ML-стека:

- **Одно представление программы.** Единственная мастер-копия — S-выражение. JIT-страница и скомпилированный HLO-граф — выбрасываемые кэши, выведенные из формы. AOT-компилятора нет.
- **Полная статика с парными аннотациями.** `:any` запрещён: `(x . :fixnum)`, `((expr) . :type)` для результата, `(name (value . :type))` для `defvar`. Старый позиционный вид удалён. Ошибка типа — отказ компиляции с позицией `file:line:col`.
- **Calling convention вызова.** `entry(frame, env, rt, out)`: кал сам пишет результат в `*out`. Это.base полного TCO — хвостовой самовызов становится прыжком, хвостовой вызов другой функции использует повтор кадра; взаимная рекурсия работает на 1M вызовов.
- **Память без GC и без RC.** Bump-арена кадров, скоупы, отдельные миры `scratch`/`world` и пул тензорных буферов. Writable views нет: `reshape`/`transpose`/`t-rows` копируют.
- **Тензоры и графы.** DTypes F16/F32/F64/U8, elementwise `+ - * /`, `matmul`, редукции, `defgraph`/`defuse`, forward/backward с O(n)-обратным проходом, `(grad term var)`, autograd `sin/cos/sqrt/atan`, fused-ядра, AVX2-пути и параллельные ядра.
- **Нейросимвольная логика.** `ns-if` с ROBDD-покрытием и дифференцируемыми ветвями, примитивы `ns-grt`/`ns-geq`/`ns-lss`/`ns-leq`/`ns-and`/`ns-or`/`ns-not`/`ns-eq`/`ns-neq`; мост `neuro` превращает символьный терм в тензорный код.
- **Стандартная библиотека из 13 модулей:** `core`, `error`, `datetime`, `string`, `collections`, `test`, `io`, `ml`, `dl`, `symbolic`, `argparse`, `tablet`, `neuro`.
- **Кроссплатформенность.** Linux x86-64 (SysV, AVX2) и Windows x86-64 (`gnu`/`msvc`), W^X через `mmap`/`mprotect` или `VirtualAlloc`/`VirtualProtect`, резерв стека 256 МБ.
- **Инструменты.** `qualisp --trace out.json script.qlsp` пишет Chrome Trace JSON; `qprof` агрегирует self/inclusive время. QSRD v1/Q2 с crc32 и секцией `model` (`save-weights`/`load-weights`). LSP (`qualisp-lsp`, `libqualisplsp`) и VSCodium-расширение собираются из исходников.
- **Проверено.** `zig build test`: 910 тестов на Linux и Windows.

Книга написана для **инженеров с Python-бэкграундом**: каждая глава содержит аналогии с Python, NumPy, PyTorch и scikit-learn.

## Оглавление

| Глава | Тема | Python-аналог |
|-------|------|---------------|
| [01. Введение](book/01-introduction.md) | Что такое Qualisp, философия, пайплайн | «Почему не Python?» |
| [02. Начало работы](book/02-getting-started.md) | Установка, REPL, первый файл, `qprof` | `python`, `pip` |
| [03. Основы языка](book/03-language-basics.md) | S-выражения, типы, `defun`, `let`, макросы, `eval` | синтаксис, `def`, `lambda`, `ast` |
| [04. Система типов](book/04-type-system.md) | Статическая типизация, `:fixnum`, `:tensor`, overload | type hints, mypy |
| [05. Модель памяти](book/05-memory-model.md) | Скоупы, bump-арена кадров, scratch/world | ручное управление памятью |
| [06. Тензоры](book/06-tensors.md) | DType, форма, `tref`/`tset!`, `matmul`, broadcast | NumPy / torch |
| [07. Автоградиент](book/07-autograd.md) | `param`, `grad!`, `grad-of`, `defgraph`, O(n)-backward | `torch.autograd` |
| [08. Глубокое обучение](book/08-deep-learning.md) | Модуль `dl`: слои, обучение, инференс | `torch.nn` |
| [09. Классический ML](book/09-classic-ml.md) | KNN, деревья, ансамбли, метрики, CV | scikit-learn |
| [10. HLO-компиляция](book/10-hlo-compilation.md) | `defgraph`/`defuse`, граф, fused-ядра, пул | `torch.compile`, XLA |
| [11. Устройства](book/11-gpu.md) | Вычисления и платформы: CPU x86-64, AVX2, W^X | `.to('cuda')` (нет) |
| [12. Макросы](book/12-macros.md) | `:syntax`, `&rest`/`&optional`, `macroexpand-all`, `eval` | декораторы, метаклассы |
| [13. Структуры и методы](book/13-oop.md) | `defstruct`, трейты, `impl`, диспетч | `@dataclass`, `Protocol` |
| [14. Символьное программирование](book/14-symbolic-programming.md) | `ns-if`, ROBDD, два спуска, `symbolic`, `neuro` | `match-case`, Prolog |
| [15. Стандартная библиотека](book/15-standard-library.md) | 13 модулей, `error`, `datetime`, изоляция | `re`, `argparse`, `collections` |
| [16. Пакеты и FFI](book/16-packages-and-ffi.md) | FFI POSIX/Win64, QSRD v1/Q2, LSP из исходников | `ctypes`, `pickle` |
| [17. Разбор примеров](book/17-full-examples.md) | XOR, `term`, аргументы CLI | end-to-end туториалы |
| [18. Шпаргалка](book/18-python-cheatsheet.md) | Qualisp ↔ Python/NumPy/PyTorch/sklearn | справочник |
| [19. Устройство компилятора](book/19-internals-overview.md) | Reader → checker → JIT → run, 910 тестов, kernel_pool | CPython internals |
| [20. JIT-компиляция](book/20-jit.md) | Unboxed fixnum, calling convention, TCO, W^X | CPython 3.13 JIT |
| [21. Образы](book/21-images.md) | QSRD v1/Q2, `save-image`, веса модели | `pickle` + `torch.save` |
| [22. Tablet](book/22-tablet.md) | Числовая таблица: `make-tablet`, `where`, `sort-by` | `pandas` |

## Быстрый старт

```bash
# Скачать бинарник (Linux x86_64, релиз v0.19.16)
curl -L https://github.com/FQingLars/Qualisp-Project/releases/download/v0.19.16/qualisp-0.19.16-linux-x86_64 -o qualisp
chmod +x qualisp && mv qualisp ~/.local/bin/

# Windows: qualisp-0.19.16-windows-x86_64.exe со страницы релизов
# Рядом: qprof-0.19.16-{linux-x86_64,windows-x86_64.exe,windows-x86_64.pdb}, SHA256SUMS.txt

# REPL
qualisp
```

```text
Qualisp
> (+ 1 2)
3
> (defun sq ((x . :fixnum)) ((* x x) . :fixnum))
SQ
> (sq 7)
49
```

Запуск файла и профилирование:

```bash
qualisp script.qlsp
qualisp --trace out.json script.qlsp
qprof out.json
```

## Для кого эта книга

- **ML-инженеры на Python**, которым нужен компилируемый язык со статической типизацией, без GIL и с нативной производительностью.
- **Лисперы**, которым нужен современный типизированный инструментарий: тензоры, autograd, HLO, нейросимвольная логика.
- **Исследователи нейросимвольного AI**: `ns-if` с ROBDD-покрытием, дифференцируемые символьные ветви и мост `neuro` из терма в тензорный код.

## Что описывает книга

Книга синхронизирована с **Qualisp 0.19.16** и описывает язык в его текущем состоянии — без истории изменений: устройство и контракты, а не список релизов.

- **Язык и типы** — S-выражения, reader, статическая типизация, `defun`/`lambda`, `&optional`/`&rest`, структуры/трейты, `:syntax` и макросы, `eval`, calling convention — гл. 3, 4, 12, 13, 19.
- **Память** — скоупы, bump-арена кадров, миры `scratch`/`world`, пул тензорных буферов, отсутствие GC/RC и writable views — гл. 5.
- **ML-подсистемы** — тензоры, `matmul`, `defgraph`/`defuse`, `grad!`/`grad-of`, `dl`, `ml`, `neuro`, `tablet` — гл. 6–10, 14, 22.
- **Нейросимвольная логика** — NS-примитивы, `ns-if`, ROBDD-покрытие, два спуска, диагностика — гл. 14.
- **Инструменты и данные** — 13 модулей stdlib, FFI POSIX/Win64, QSRD v1/Q2, `save-weights`/`load-weights`, `--trace` и `qprof` — гл. 2, 15, 16, 19, 21.
- **Производительность** — JIT по S-выражениям, полный TCO, SIMD AVX2, fused-ядра, Windows-поддержка — гл. 11, 20.

Известные ограничения (подробности — в соответствующих главах):

- **Статическая типизация строгая**: `:any` запрещён, аннотации обязательны и записываются парами `(x . :fixnum)`; старый позиционный вид удалён. Ламбда-литерал нельзя вызвать напрямую — только через `funcall`.
- **GPU-бэкенда нет**: `Device` в реализации отсутствует; вычисления идут на CPU x86-64 через JIT и AVX2-ядра.
- **Ограниченный набор платформ**: x86-64 SysV/Win64; других архитектур JIT не поддерживает.
- **Бинарные ассеты только для `qualisp` и `qprof`**: `qualisp-lsp`, `libqualisplsp` и VSIX собираются из исходников (`docs/lsp.md`, `editors/vscode`); пакетного менеджера нет; FFI под Windows требует Win64-трамплина.
- **Нет writable views**: `reshape`/`transpose`/`t-rows` копируют данные; `t-slice` отсутствует.
- **Ограничения тензорного API**: `tensor-shape`, `t+`, `t*`, `tensor-dtype`, `sigmoid` отсутствуют — elementwise делает обычный `+`/`*`; `softmax` и `argmax` работают только с 1-D.
- **Нет пользовательских перегрузок**: одно имя — одно определение.
- **Ограничения stdlib**: 13 модулей, `async` отдельным модулем не существует (есть примитивы `future`/`await`), `FFI` и `NS` не являются импортируемыми модулями.
- **Образы и сериализация**: QSRD v1/Q2 и `save-image`/`save-weights` есть, но `qsrd-save`/`qsrd-load` как API отсутствуют.
- **Тестирование**: 910 тестов в `zig build test`, без системы сюит `deftest`/`run-tests`.

## Лицензия

Условия использования компилятора определяются его репозиторием; эта книга — документация проекта.
