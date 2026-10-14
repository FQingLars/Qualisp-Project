# 📖 Qualisp Book

**Ориентировочная книга по языку программирования Qualisp** (ранее QLISP) — Лиспу для машинного обучения и AI.

Qualisp — это диалект Лиспа, интерпретатор и JIT-компилятор которого работают прямо на S-выражениях (copy-and-patch в нативные страницы без промежуточного IR), со встроенной тензорной системой (F16-поддержка с v2.2.0), автоградиентом, стенсил-компиляцией HLO-кёрнелов и нейросимвольным движком (rule engine, унификация, `NS-IF`/`NS-GRAD!` с v2.2.0+) — всё в стандартной библиотеке, без единого `pip install`. Серия v2.3.x принесла: гомоиконность (`EVAL`/`READ-FROM-STRING`/`FUNCTION-BODY`/`SYMBOL-NAME`), QSRD v2 для бинарной сериализации, модуль `string`, частичный импорт `IMPORT-FROM`, переписанный `dl` со слоями-как-данными, переживающий границы форм gradient tape, производительность уровня PyTorch на ключевых операциях (SIMD-редукции, тайловый transpose, векторизованный softmax, memcpy-im2col conv), нейросимвольное RL-ядро (`SAMPLE`, макро-слоты, `EVAL-SANDBOXED`, REINFORCE), маршрутизацию внутри HLO-графа (`ROUTE`-нода, `HLO-ROUTE-GRAD!` без ленты), граф-как-данные (`GRAPH-DATA`/`GRAPH-FROM-DATA`/`GRAPH-RUN-PASSES`), полноценный LSP-сервер (диагностики, навигация по воркспейсу, folding, форматирование) и нейросимвольный RL-движок **blackit** (агент-кодер, слоты, песочница).

Серия **v2.4–v2.7** переписала компиляторную часть: легаси AOT-путь и мир `qvalue` захоронены в пользу **инварианта одной формы** — S-выражение остаётся единственным представлением программы, а нативный код (`JIT`/`UNJIT`/`DISCARD-JIT`, copy-and-patch) — выбрасываемым кэшем, который выводится из S-выражений напрямую, без промежуточного IR; добавлены образы мира `SAVE-IMAGE`/`LOAD-IMAGE`, а HLO-кёрнелы компилируются стенсил-эмиттером — рантайм больше не линкует LLVM. Детали — в [release notes v2.7.0](https://github.com/FQingLars/Qualisp-Project/releases/tag/v2.7.0).

Книга написана для **инженеров с Python-бэкграундом**: каждая глава содержит аналогии с Python, NumPy, PyTorch и scikit-learn.

## Оглавление

| Глава | Тема | Python-аналог |
|-------|------|---------------|
| [01. Введение](book/01-introduction.md) | Зачем нужен Qualisp, философия, пайплайн | «Почему не Python?» |
| [02. Начало работы](book/02-getting-started.md) | Установка, REPL, первая программа, JIT | `python`, `pip` |
| [03. Основы языка](book/03-language-basics.md) | S-выражения, функции, гомоиконность (`EVAL`, `FUNCTION-BODY`) | синтаксис, `def`, `lambda`, `ast` |
| [04. Система типов](book/04-type-system.md) | Постепенная типизация, линейные типы тензоров | type hints, mypy |
| [05. Модель памяти](book/05-memory-model.md) | HibLin: scratch/stable, пулы, Scope, `defvar`, `reset_scratch` | GC, `numpy` views |
| [06. Тензоры](book/06-tensors.md) | N-мерные массивы, F16/F32, broadcasting, matmul | NumPy / torch |
| [07. Автоградиент](book/07-autograd.md) | Лента градиентов, `!`-операции, SGD/Adam, контракт tape между формами | `torch.autograd` |
| [08. Глубокое обучение](book/08-deep-learning.md) | Модуль DL v2: слои-как-данные, `net-forward`/`train-step`/`train-epochs` | `torch.nn` |
| [09. Классический ML](book/09-classic-ml.md) | KNN, деревья, ансамбли, метрики, CV, grid search | scikit-learn |
| [10. HLO-компиляция](book/10-hlo-compilation.md) | Стенсил-эмиттер: трассировка → CSE/DCE/fusion → copy-and-patch на W^X-страницах (без LLVM) | `torch.compile`, XLA |
| [11. GPU](book/11-gpu.md) | Устройства, статус CUDA-бэкенда | `.to('cuda')` |
| [12. Макросы](book/12-macros.md) | Код-как-данные, backquote, compiler macros, `EVAL`/`READ-FROM-STRING` | декораторы, метаклассы |
| [13. ООП](book/13-oop.md) | defstruct, defclass, дженерики (CLOS) | `class`, `@dataclass` |
| [14. Символьное программирование](book/14-symbolic-programming.md) | match, unify, rule engine, amb, `NS-IF`/`NS-GRAD!`, движок blackit (NS-RL) | `match-case`, Prolog |
| [15. Стандартная библиотека](book/15-standard-library.md) | IO, Regex, Audio, Datetime, Errors, Visual, **String** | `re`, `open`, `datetime` |
| [16. Пакеты и FFI](book/16-packages-and-ffi.md) | qvalent, pkg-install, FFI: dlopen+FFI-CALL/IMPORT | `pip`, `ctypes` |
| [17. Разбор примеров](book/17-full-examples.md) | Линейная регрессия и XOR-MLP с Python-эквивалентами | end-to-end туториалы |
| [18. Шпаргалка](book/18-python-cheatsheet.md) | Qualisp ↔ Python/NumPy/PyTorch/sklearn | справочник |
| [19. Устройство компилятора](book/19-internals-overview.md) | Reader, интерпретатор, HibLin, JIT, стенсилы, LSP | CPython internals |
| [20. JIT-компиляция](book/20-jit.md) | Copy-and-patch: RX-страницы из S-выражений, решётка типов, специализация, инвалидация | CPython 3.13 JIT |
| [21. Образы](book/21-images.md) | `SAVE-IMAGE`/`LOAD-IMAGE`: мир как данные, формат QLSI | `pickle` + `torch.save` |
| [22. Tablet](book/22-tablet.md) | Детерминированный DataFrame-аналог pandas: groupby/merge/окна, CSV/JSON, NA-дисциплина | `pandas` |

## Быстрый старт

```bash
# Скачать бинарник (Linux x86_64, релиз v2.8.6)
curl -L https://github.com/FQingLars/Qualisp-Project/releases/latest/download/qualisp-linux-x86_64 -o qualisp
chmod +x qualisp && mv qualisp ~/.local/bin/

# Windows: qualisp-windows-x86_64.zip со страницы релизов (qualisp.exe, qualisp-lsp.exe, qvalent.exe)

# REPL
qualisp
```

```lisp
> (+ 1 2 3 4 5)
=> 15
> (defun square (x: Float) -> Float (* x x))
=> square
> (square 4.0)
=> 16.0
> (import ML)
```

## Для кого эта книга

- **ML-инженеры на Python**, которым нужен компилируемый язык без GIL, без GC-пауз и с нативной производительностью.
- **Лисперы**, которым нужен современный инструментарий ML: тензоры, автоград, HLO-AOT, нейросимвольный движок.
- **Исследователи нейросимвольного AI**: Qualisp уникально сочетает гомоиконный символьный движок (правила, унификация, паттерн-матчинг, `NS-IF`/`NS-GRAD!` с guilt detector) с нейросетевым бэктендом (автоград, HLO, GPU).

## Статус книги

Книга синхронизирована с компилятором **v2.8.6**. Отражены все ключевые серии: похороны легаси AOT и мира `qvalue` (v2.4.0, «инвариант одной формы»), JIT copy-and-patch с типовыми слотами и специализацией по наблюдаемым типам — гл. 20 (v2.5.0–v2.5.5), образы `SAVE-IMAGE`/`LOAD-IMAGE` — гл. 21 (v2.6.0), стенсил-эмиттер HLO и отказ от LLVM-пути — гл. 10 (v2.6.1–v2.7.0), нативные стенсилы для `SOFTMAX`/`ARGMAX`/`LOOKUP`/`WEIGHTED_LOOKUP` и CODEGEN-видов, CSV по RFC-4180, тип-предикаты и потери в `ml` (v2.7.1), модуль **Tablet** — детерминированный DataFrame-аналог pandas — гл. 22 (v2.7.2), HibLin OOM-фиксы (loop store, автоочистка ленты) и эвикция пулов с guard глубины рекурсии — гл. 3, 19 (v2.7.3–v2.7.4), серия контрактов времени жизни: O(1) spine-append для `setq`-накопителей, loop store = граница формы, part-time HLO, якорение property lists и итеративный спайн сериализатора образов — гл. 5.5, 21 (v2.7.5–v2.8.1), tail-call optimization — гл. 3, кэш раскрытия макросов — гл. 12 и журнал событий владения DEBUG-TRACE — гл. 19 (v2.8.2), переход на подсчёт ссылок для стабильных ячеек с отложенной смертью замен, live-резолвингом глобалов и закрытием регресса `&optional` — гл. 5.5, 19 (v2.8.3), закрытие контракта памяти: LSan-гейт в CI (`detect_leaks=1` по всему корпусу) и добивка последних утечек — гл. 5.5, 19 (v2.8.4), серия «громких отказов» вместо тихого мусора (`strcat` с не-строками, одноаргументные `print`/`display`, лишние аргументы вызова, границы `tref`, перечисление всех решений в `amb-collect`, вложенные тензоры в QSRD, предупреждение о градиенте, брошенном на границе формы) и **rebrand QLISP → Qualisp** (бинарники `qualisp`/`qualisp-lsp`, переменные `QUALISP_*`, кодовое имя NS-RL-движка **blackit**) — гл. 2, 3, 6, 7, 14, 16 (v2.8.5), а также **единая модель владения** (RC для структурных данных, точный пул без RC для тензоров): отмена отложенной смерти и loop-store в пользу stable+RC с реестром освобождения, шаринг хвостовых параметров TCO, header стабильного тензора умирает вместе с буфером, leak gate 92/92 — гл. 5.5, 19 (v2.8.6); плюс три продакшен-находки — `opt-apply` без клонирования весов (Adam сходится, state не осиротеет), `ADAM-STEP` помечает градиенты прочитанными, `LOAD-IMAGE` мира со скомпилированной HLO-программой, константы HLO-трейса владеют своими байтами, итеративный `read-csv`, `setq` с несколькими парами, числовое равенство через представления — гл. 3, 5.5, 8, 10, 15, 19, 21 (v2.8.6). Историческая серия **v2.3.x** задокументирована: фикс времени жизни градиентов и HLO-хендлов (2.3.1), корректность rule engine и завершение S7 (`SECOND-BEST`, 2.3.2), производительность тензоров/автограда до паритета с PyTorch и дисциплина памяти (2.3.3), нейросимвольное RL-ядро blackit T1–T4 (2.3.4), ROUTE-нода в HLO (2.3.5) и `HLO-ROUTE-GRAD!` (2.3.6), граф-как-данные + умная печать тензоров (2.3.7) и полноценный `qualisp-lsp` (2.3.8).

Известные ограничения реализации (см. соответствующие разделы в книге):

- CUDA-бэкенд (`src/gpu/kernels.cu`) присутствует, но **не тестирован на GPU**.
- Windows: используйте релизные бинарники (собираются в CI, MSYS2 MinGW64) или локальную MSYS2-сборку; после отказа от LLVM в codegen (v2.7.0) путь кросс-компиляции больше не ограничен ABI host-LLVM.
- HLO-fallback (без `HLO-COMPILE`) держит константы формы трассировки — кросс-форменное использование такого HLOPROG небезопасно.
- Графы с `ROUTE` и `COMPOSITE`/`TUPLE` исполняются fallback-программой (LLVM-путь codegen удалён в v2.7.0; `SOFTMAX`/`ARGMAX`/`LOOKUP`/`WEIGHTED_LOOKUP`/CODEGEN-виды компилируются стенсилами с v2.7.1; диспетчеризация ROUTE стенсилами — в плане).
- Известные фейлы ранних 2.3.x (`ns-nested-breed-guilty`, `neurosym-rule-check`) устранены в 2.3.1–2.3.2 (#28–#33); полный сюит к 2.3.8 зелёный в Release, ASan чист на ключевых путях.

## Лицензия

Компилятор Qualisp распространяется под лицензией **GPLv3** (`LICENSE.md` в репозитории компилятора); эта книга — под MIT.
