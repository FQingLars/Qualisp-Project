# Глава 11. Устройства: одна реализация, открытый интерфейс

Qualisp имеет абстракцию устройств: ядро оперирует понятием «device» тензора (`Device` в `src/gpu/device.hpp`), и код пишется одинаково — независимо от того, на чём он будет исполняться.

## 11.1 Устройство одно: CPU

Реализация **одна**: CPU. В дереве проекта нет ни CUDA-, ни ROCm-кода — GPU-бэкенда нет, и книга этого не обещает.

```cpp
enum class DeviceType : uint8_t { CPU = 0 };
```

`DeviceType` — только `CPU`, а `Device::available_devices()` возвращает единственное устройство:

```cpp
std::vector<Device *> Device::available_devices() {
    std::vector<Device *> devs;
    devs.push_back(&g_cpu_device);   // только CPU
    return devs;
}
```

Реализация CPU — `CPUDevice` — обёртка над SIMD-ядрами:

- **AVX2/FMA** на x86_64: 8×float за инструкцию (`simd::add_ps`, `simd::mul_ps`, `simd::relu_ps`).
- **NEON** на aarch64: 4×float за инструкцию; в сборке без SIMD — скалярный путь.
- **F16-ядра:** `launch_*_half` — half-буферы обрабатываются через `static_cast<Half>(...)`; на x86_64 с `-march=native` компилятор задействует F16C.
- `launch_matmul` — `cblas_sgemm` (OpenBLAS).
- `alloc_pooled` / `free_pooled` — TensorBufferPool (гл. 5): exact-size выделение и возврат в пул.

`launch_*` покрывают float и half (F16). Double и int через них не проходят — для таких значений используется путь `get_float`/`set_float`.

## 11.2 Интерфейс Device

`Device` — абстрактный класс; всё, что умеет устройство, описано интерфейсом:

```cpp
class Device {
    virtual DeviceType type() const = 0;
    virtual void* alloc(size_t bytes) = 0;
    virtual void free(void* ptr) = 0;
    virtual void* alloc_pooled(size_t bytes);   // по умолчанию: raw alloc
    virtual void free_pooled(void* ptr, size_t);

    virtual void memcpy_to_device(void* dst, const void* src, size_t bytes) = 0;
    virtual void memcpy_from_device(void* dst, const void* src, size_t bytes) = 0;
    virtual void sync() = 0;

    virtual void launch_elementwise_add(const float* a, const float* b, float* c, size_t n) = 0;
    virtual void launch_elementwise_mul(...);
    virtual void launch_relu(const float* a, float* b, size_t n) = 0;
    virtual void launch_matmul(const float* a, const float* b, float* c, size_t M, size_t K, size_t N) = 0;

    static Device& default_device();
    static Device& cpu_device();
    static std::vector<Device*> available_devices();
    static void initialize();
};
```

Интерфейс оставлен не как задел на несуществующий код, а как **точка расширения**: если появится второй бэкенд, он реализует те же десять виртуальных методов, а примитивы, стенсильный эмиттер и интерпретатор продолжат работать без изменений. Тензоры создаются на `default_device()`, то есть на CPU.

`Tensor::device` указывает на активное устройство; `Tensor::ensure_cpu()` мигрирует данные на CPU при необходимости — сейчас это единственное устройство, поэтому миграция вырождена.

## 11.3 Интроспекция

```lisp
(device-info)    ;; → ((TYPE "CPU") (NAME "CPU") (GPU "false"))
```

Пользуйтесь ей в новых окружениях, чтобы понимать, на чём выполняется код: значение `GPU "false"` — честный ответ на вопрос «работает ли это на видеокарте».

## 11.4 Почему CPU хватает для деплоя

Для инференса обученной модели CPU-путь быстр:

- поэлементные операции — SIMD (AVX2/FMA на x86_64, NEON на aarch64);
- `MATMUL` — OpenBLAS (`cblas_sgemm`), производительность уровня NumPy/PyTorch CPU;
- матричные умножения в батчах идут через BLAS, а не интерпретатор;
- F16-тензоры (гл. 6) вдвое экономят память и пропускную способность к данным уже на CPU (half хранение, float аккумулятор в matmul).

> 🐍 **Python-аналогия.** Это как `torch.set_default_device('cpu')`: тот же код, что и на GPU, — просто без GPU. Код, написанный без привязки к устройству, переносится на другое устройство без переписывания — но переносить пока некуда.

## 11.5 Связка с HLO

HLO-графы (гл. 10) компилируются под CPU. Стенсильный эмиттер генерирует C-ABI-функцию, которая вызывает те же CPU-ядра (SIMD/BLAS), а узел `DOT` превращается в вызов `cblas_sgemm`. Никакого «кодогена под другое устройство» в проекте нет: эмиттер один, и он — CPU.

## 11.6 Практические советы

1. Пишите код, не задумываясь об устройстве — он просто работает на CPU.
2. Проверяйте `(device-info)`, если кто-то обещал вам GPU-сборку.
3. Большие батчи — в тензоры сразу (`(randn (list 1000 256))`), а не в циклы по спискам: тензорный путь идёт через пулы (гл. 5) и SIMD/BLAS.
4. Хотите скорость на CPU — используйте F16 для больших тензоров и `T-FMA`/`T-FMA-RELU` в горячих местах.

## 11.7 Устройства в коде

```
              ┌─────────────────────┐
              │      Device         │
              │   (abstract)        │
              └──────────┬──────────┘
                         │
                         ▼
                   CPUDevice
                   (единственная
                    реализация)
```

## 11.8 Известные ограничения

- **GPU-бэкенда нет.** Устройство одно — CPU; в дереве нет ни CUDA-, ни ROCm-кода. Обещать GPU в этой книге не на чем.
- Нет Vulkan/Metal-бэкенда.
- `launch_*` покрывают float и half (F16); double и int идут через `get_float`/`set_float` — это медленнее прямой обработки.
- Нет асинхронных операций: у интерфейса нет потоков и очередей, `sync()` — заглушка синхронизации.
