# Оптимизация Inference Pipeline

В данном репозитории представлены эксперименты по оптимизации инференса ML-моделей на CPU.  
Используемая модель: rubert-mini-frida.  
Полный отчет и графики находятся в файле REPORT.md  

## Структура проекта
part1_baseline.py - базовый сервер (HuggingFace + PyTorch)  
part2_onnx.py - оптимизированный сервер (ONNX Runtime)  
part3_batching.py - сервер с батчингом (asyncio.Queue + ONNX)  
export_onnx.py - конвертация модели в ONNX  
benchmark.py - скрипт нагрузки  
plots.py - скрипт отрисовки графиков  
requirements.txt - зависимости  

## Инструкция по запуску

### 1. Подготовка окружения
Создайте и активируйте виртуальное окружение, затем установите зависимости:

    python -m venv venv
    source venv/Scripts/activate
    pip install -r requirements.txt

### 2. Бенчмарк бейзлайна
В файле benchmark.py укажите: RUN_NAME = "Part 1: Baseline"  
В первом терминале запустите:  

    python part1_baseline.py

Во втором терминале запустите:   

    python benchmark.py

### 3. Бенчмарк ONNX  
Сначала сконвертируйте модель командой:  
    
    python export_onnx.py

В файле benchmark.py укажите: RUN_NAME = "Part 2: ONNX"  
В первом терминале запустите:   

    python part2_onnx.py

Во втором терминале запустите:  

    python benchmark.py

### 4. Бенчмарк батчинга
В файле benchmark.py укажите: RUN_NAME = "Part 3: Batching"  
В первом терминале запустите: 

    python part3_batching.py

Во втором терминале запустите:   

    python benchmark.py

### 5. Отрисовка графиков  
После проведения всех трех тестов выполните команду:
    
    python plots.py

Скрипт сохранит графики в папку plots и они будут подставлены в REPORT.md