# Глава 11. GPU: сегодня CPU, завтра CUDA

Qualisp имеет абстракцию устройств: ядро оперирует понятием «device» тензора (`Device` в `src/gpu/device.hpp`), и код пишется одинаково для CPU и (в перспективе) GPU.

## 11.1 Что работает сегодня

Реализован и зарегистрирован по умолчанию **один бэкенд — CPU** (см. `src/gpu/device.cpp`):

```cpp
enum class DeviceType : uint8_t { CPU = 0, CUDA = 1, ROCM = 2 };

std::vector<Device *> Device::available_devices() {
    std::vector<Device *> devs;
    devs.push_back(&g_cpu_device);   // только CPU
    return devs;
}
```

Реализация CPU — `CPUDevice` (`src/gpu/device.cpp:29`) — обёртка над SIMD ядрами (`src/core/simd.hpp`):

- AVX2/FMA на x86_64: 8×float за инструкцию (`simd::add_ps`, `simd::mul_ps`, `simd::relu_ps`).
- NEON на ARM (см. CMakeLists.txt: `HAVE_NEON` → `QUALISP_HAS_NEON`).
- F16 ядра: `launch_*_half` через `static_cast<Half>(...)`; на x86_64 с `-march=native` компилятор генерирует F16C.
- `launch_matmul` — `cblas_sgemm` (OpenBLAS).
- `alloc_pooled` / `free_pooled` — TensorBufferPool.

`device-info` (`prim_device_info`):

```lisp
(device-info)    ;; → ((TYPE "CPU") (NAME "CPU") (GPU "false"))
```

Тензоры всегда на CPU; `(device-info)` это подтверждает.

CUDA-бэкенд (`src/gpu/kernels.cu`, 36 строк) присутствует в исходниках как каркас, но **не зарегистрирован** в `Device::available_devices()` и **не собирается** в релизном бинарнике v2.3.0 (требует CUDA Toolkit и тестирования на GPU). ROCm-бэкенд — в дорожной карте. Книга честно отражает это: на сегодня Qualisp — CPU-early.

## 11.2 Почему это нормально для деплоя

Для инференса обученной модели CPU-путь быстр:

- поэлементные операции — SIMD (AVX2/FMA);
- `MATMUL` — OpenBLAS (`cblas_sgemm`), производительность уровня NumPy/PyTorch CPU;
- матричные умножения в батчах идут через BLAS, а не интерпретатор;
- F16-тензоры (гл. 6) вдвое экономят память и пропускную способность к данным уже на CPU (half хранение, float аккумулятор в matmul).

> 🐍 **Python-аналогия.** Это как `torch.set_default_device('cpu')`: тот же код, что и на GPU, — просто без GPU. Когда CUDA-бэкенд созреет, он подключится через device-слой без переписывания пользовательского кода.

## 11.3 Интроспекция

```lisp
(device-info)    ;; какие бэкенды скомпилированы и активное устройство
```

Используйте её в новых окружениях, чтобы понимать, на чём выполняется код.

## 11.4 Связка с HLO

HLO-графы (гл. 10) компилируются под CPU сегодня. Кодоген генерирует C-ABI-функцию, которая вызывает те же CPU-кирнелы (SIMD/BLAS через `Device` API). Когда появится CUDA-кодоген, он встанет в тот же конвейер: `HloGraph → кодоген → .so/.ptx → вызов`.

## 11.5 Практические советы

1. Пишите код, не задумываясь об устройстве — он просто работает на CPU.
2. Проверяйте `(device-info)`, если кто-то обещал вам GPU-сборку.
3. Большие батчи — в тензоры сразу (`(randn (list 1000 256))`), а не в циклы по спискам: тензорный путь идёт через пулы (гл. 5) и SIMD/BLAS.
4. Хотите скорость на CPU — используйте `F16` для больших тензоров и `T-FMA`/`T-FMA-RELU` в горячих местах.

## 11.6 Архитектура устройств

```
              ┌─────────────────────┐
              │      Device         │
              │   (abstract)        │
              └──────────┬──────────┘
                         │
       ┌─────────────────┼─────────────────┐
       │                 │                 │
       ▼                 ▼                 ▼
  CPUDevice         CudaDevice        RocmDevice
  (active)         (kernels.cu       (в планах)
                    заглушка)
```

`Tensor::device` указывает на активное устройство; `Tensor::ensure_cpu()` мигрирует данные на CPU при необходимости.