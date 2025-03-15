"""-------------------------------------------------------
CLIP GENIUS: MultiProcess
-------------------------------------------------------
Author:  JD
ID:      91786
Uses:    threading, queue
Version:  1.0.8
__updated__ = Fri Mar 14 2025
-------------------------------------------------------
"""

import threading
import queue


def worker_function(task_id):
    result = [f"{task_id}= {i}" for i in range(1, 4)]
    return result


def thread_wrapper(task_id, result_queue):
    result = worker_function(task_id)
    result_queue.put(result)


def run_tasks_with_threads(num_tasks):
    result_queue = queue.Queue()
    threads = []

    for task_id in range(num_tasks):
        thread = threading.Thread(target=thread_wrapper, args=(task_id, result_queue))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    results = []
    while not result_queue.empty():
        results.extend(result_queue.get())

    return results


if __name__ == "__main__":
    num_tasks = 10
    final_results = run_tasks_with_threads(num_tasks)
    print("Final Results:", final_results)
