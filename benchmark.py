import asyncio
import aiohttp
import time
import psutil
import statistics
import os
import json


URL = "http://127.0.0.1:8567/predict"
# RUN_NAME = "Part 1: Baseline"
# RUN_NAME = "Part 2: ONNX"
RUN_NAME = "Part 3: Batching"

CONCURRENCY_LEVELS = [1, 5, 10, 25, 50, 100, 200]
REQUESTS_PER_STEP = 1000

SAMPLE_TEXTS = [
    "Мышь бежит.",
    "Как поймать мышь в доме безопасно?",
    (
        "Домовая мышь — вид грызунов семейства мышиных. "
        "Это один из самых многочисленных видов млекопитающих на планете."
    ),
    (
        "Лабораторные белые мыши широко используются в научных "
        "исследованиях для тестирования новых медицинских препаратов."
    ),
    (
        "В дикой природе мыши питаются преимущественно растительной пищей, "
        "такой как семена, злаки и корни. Однако они всеядны и при "
        "возможности охотно поедают насекомых или продукты человека."
    ),
    (
        "Интересный факт: вопреки распространенному мифу, мыши не так "
        "уж сильно любят сыр. Ученые доказали, что они предпочитают "
        "пищу с высоким содержанием углеводов, например, фрукты или зерно. "
        "Сыр они могут съесть только при полном отсутствии альтернативы."
    ),
]


async def send_request(session, text):
    start_time = time.time()
    try:
        async with session.post(URL, json={"text": text}) as response:
            status = response.status
            if status == 200:
                await response.json()
            return time.time() - start_time, status == 200
    except Exception:
        return time.time() - start_time, False


async def run_single_load_test(concurrency):
    print(f"concurrency={concurrency} ...", end="", flush=True)
    psutil.cpu_percent(interval=None)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(REQUESTS_PER_STEP):
            text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
            tasks.append(send_request(session, text))

        sem = asyncio.Semaphore(concurrency)

        async def sem_task(task):
            async with sem:
                return await task

        start_time = time.time()
        results = await asyncio.gather(*(sem_task(t) for t in tasks))
        end_time = time.time()

    cpu_usage = psutil.cpu_percent(interval=None)

    latencies = [res[0] for res in results if res[1]]
    success_count = sum(1 for res in results if res[1])
    error_rate = (
        (REQUESTS_PER_STEP - success_count) / REQUESTS_PER_STEP
    ) * 100

    total_time = end_time - start_time
    throughput = REQUESTS_PER_STEP / total_time if total_time > 0 else 0

    if latencies:
        avg_latency = statistics.mean(latencies) * 1000
        sorted_lat = sorted(latencies)
        p95_latency = sorted_lat[int(len(sorted_lat) * 0.95)] * 1000
        p99_latency = sorted_lat[int(len(sorted_lat) * 0.99)] * 1000
    else:
        avg_latency = p95_latency = p99_latency = 0.0

    print(
        f"RPS: {throughput:.1f}, "
        f"p95: {p95_latency:.1f}ms, CPU: {cpu_usage:.1f}%"
    )

    return (
        throughput,
        avg_latency,
        p95_latency,
        p99_latency,
        cpu_usage,
        error_rate,
    )


async def main():
    print(f"Бенчмарк [{RUN_NAME}]")

    step_results = {}
    for c in CONCURRENCY_LEVELS:
        rps, avg_lat, p95, p99, cpu, err = await run_single_load_test(c)
        step_results[str(c)] = {
            "throughput": rps,
            "avg_latency": avg_lat,
            "p95_latency": p95,
            "p99_latency": p99,
            "cpu_usage": cpu,
            "error_rate": err,
        }

    results_file = "benchmark_results.json"
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results[RUN_NAME] = step_results

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"Результаты сохранены в {results_file}")


if __name__ == "__main__":
    asyncio.run(main())
