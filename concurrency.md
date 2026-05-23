# Designing Asynchronous and Parallel Systems

### ⚡ The Golden Rule
*   **Network I/O (APIs/Web):** Use **AsyncIO**.
*   **Disk I/O / Shared Memory:** Use **Threading**.
*   **CPU Math (AI/Data):** Use **Multiprocessing**.

---

### 1. The Three Models

| Feature        | **AsyncIO**                  | **Threading**                | **Multiprocessing**              |
| :------------- | :--------------------------- | :--------------------------- | :------------------------------- |
| **Mechanism**  | Single Thread, Event Loop    | Multiple Threads, Shared Mem | Multiple Processes, Isolated Mem |
| **Type**       | Cooperative (Yields control) | Preemptive (OS switches)     | True Parallelism                 |
| **Best For**   | High-Concurrency Network I/O | Disk I/O, Background Tasks   | Heavy CPU Computation            |
| **Python GIL** | Irrelevant (Single thread)   | **Blocked** (No CPU speedup) | **Bypassed** (Full speedup)      |
| **Overhead**   | Ultra-Low (User-space)       | Medium (Context switches)    | High (Process creation/IPC)      |

---

### 2. Key Concepts & Metaphors

#### **AsyncIO (The Chef)**
*   **How:** One chef starts water boiling, then chops veggies while waiting. Never stands idle.
*   **Why:** Handles 10,000+ connections with minimal RAM. No context switch overhead.
*   **Risk:** **Blocking the Loop.** If one task does heavy CPU work without `await`, it freezes *everything*.

#### **Threading (The Multi-Armed Chef)**
*   **How:** One brain, multiple hands. Switches tasks so fast it looks parallel.
*   **Why:** Great for disk reads/writes where data needs to be shared easily in memory.
*   **Risk:** **Race Conditions.** Two threads writing to same memory = corruption. Needs Locks.
*   **Python Note:** The **GIL** prevents true CPU parallelism. Only useful for I/O.

#### **Multiprocessing (The Kitchen Franchise)**
*   **How:** Separate chefs in separate kitchens. No shared tools or memory.
*   **Why:** Bypasses Python’s GIL. Uses all CPU cores for math/AI training.
*   **Risk:** **IPC Overhead.** Sharing data between processes requires serialization (Pickling), which is slow.

---

### 3. Architectural Patterns

*   **Hybrid Approach (Production Standard):**
    *   Use **Multiprocessing** to spawn workers (one per CPU core).
    *   Inside each worker, use **AsyncIO** to handle thousands of network connections.
    *   *Example:* Uvicorn/Gunicorn serving an LLM API.

*   **Offloading CPU in Async:**
    *   Never do heavy math in an `async` function.
    *   Use `loop.run_in_executor()` to send CPU tasks to a thread/process pool so the event loop stays free.

---

### 🎯 Interview Cheat Sheet

1.  **"Why not just use Threads for everything?"**
    *   Threads have high memory overhead (stack size) and context-switching costs. They crash at ~10k connections. AsyncIO scales to millions.
2.  **"Why doesn't Threading speed up Python CPU tasks?"**
    *   The **Global Interpreter Lock (GIL)** allows only one thread to execute Python bytecode at a time. Use Multiprocessing instead.
3.  **"What is the biggest risk in AsyncIO?"**
    *   Accidentally blocking the event loop with synchronous code (e.g., `time.sleep()` or heavy `numpy` ops). Always use `async` libraries or executors.


### Concurrency Primatives

### 🔒 1. The Global Interpreter Lock (GIL)
*   **What is it?** A mutex (lock) in CPython that allows only **one thread** to execute Python bytecode at a time.
*   **Why does it exist?** To simplify memory management (reference counting) and make the interpreter thread-safe without complex locking mechanisms.
*   **Impact:**
    *   **CPU-Bound Tasks:** No parallelism. Adding threads does **not** speed up calculation; it often slows it down due to context switching overhead.
    *   **I/O-Bound Tasks:** Minimal impact. When a thread waits for I/O (network/disk), it releases the GIL, allowing other threads to run.
*   **Workarounds:**
    *   Use `multiprocessing` (bypasses GIL by using separate processes).
    *   Use C-extensions (NumPy/Cython) which release the GIL during heavy computation.
    *   Use alternative interpreters (Jython, IronPython) or Python 3.14+ free-threaded builds.

---

### 🛡️ 2. Locks (Mutexes)
*   **What is it?** A synchronization primitive that ensures **mutual exclusion**. Only one thread can hold the lock at a time.
*   **How it works:**
    1.  Thread A acquires lock.
    2.  Thread B tries to acquire lock → **Blocks/Waits**.
    3.  Thread A finishes and releases lock.
    4.  Thread B acquires lock and proceeds.
*   **Use Case:** Protecting shared resources (e.g., writing to a file, updating a global counter) from **Race Conditions**.
*   **Risk:** **Deadlock.** If Thread A holds Lock 1 and wants Lock 2, while Thread B holds Lock 2 and wants Lock 1, both wait forever.

```python
import threading

counter = 0
lock = threading.Lock()

def increment():
    global counter
    with lock:  # Acquires lock, releases automatically on exit
        counter += 1
```

---

### 🚦 3. Semaphores
*   **What is it?** A generalized lock that allows a **fixed number of threads** to access a resource simultaneously.
*   **How it works:** Maintains an internal counter.
    *   `acquire()`: Decrements counter. If counter == 0, thread blocks.
    *   `release()`: Increments counter. Wakes up a waiting thread.
*   **Types:**
    *   **Binary Semaphore:** Counter is 0 or 1. Behaves exactly like a **Lock**.
    *   **Counting Semaphore:** Counter is N. Allows N concurrent threads.
*   **Use Case:** Limiting concurrency.
    *   *Example:* You have 5 database connections but 100 threads. Use a Semaphore(5) to ensure no more than 5 threads hit the DB at once.

```python
import threading

# Allow max 3 concurrent downloads
semaphore = threading.Semaphore(3)

def download(url):
    with semaphore:  # Blocks if 3 threads are already downloading
        perform_download(url)
```

---

### 🆚 Quick Comparison

| Primitive     | Purpose             | Concurrency Level                     | Analogy                                                            |
| :------------ | :------------------ | :------------------------------------ | :----------------------------------------------------------------- |
| **GIL**       | Interpreter Safety  | **1 Thread** (at a time for bytecode) | One microphone in a debate club. Only one person speaks at a time. |
| **Lock**      | Resource Protection | **1 Thread** (holds exclusive access) | A single-key bathroom. Only one person inside.                     |
| **Semaphore** | Resource Limiting   | **N Threads** (limited pool)          | A parking lot with 10 spots. 10 cars can enter; others wait.       |

---

### ⚠️ Common Pitfalls & Best Practices

1.  **Deadlocks:**
    *   *Cause:* Two threads waiting for each other’s locks.
    *   *Fix:* Always acquire locks in the **same order** across all threads. Use timeouts (`lock.acquire(timeout=5)`).

2.  **Starvation:**
    *   *Cause:* High-priority threads constantly acquire the lock, preventing low-priority threads from ever running.
    *   *Fix:* Use fair locks (FIFO queues) or priority scheduling.

3.  **Over-Locking:**
    *   *Cause:* Holding a lock for too long (e.g., doing network I/O while holding a lock).
    *   *Fix:* Keep critical sections as small as possible. Release the lock before performing slow operations.

4.  **GIL Misconception:**
    *   *Myth:* "The GIL prevents multi-core usage."
    *   *Truth:* The GIL prevents multi-threaded *Python bytecode* execution. Multi-processing and C-extensions still utilize all cores.

---

### 💡 Interview Cheat Sheet

*   **"When do you need a Lock?"**
    *   Whenever multiple threads read/write **shared mutable state** (variables, files, DB connections).
*   **"When do you need a Semaphore?"**
    *   When you need to **limit concurrency** for a scarce resource (DB connections, API rate limits, GPU slots).
*   **"Does the GIL affect AsyncIO?"**
    *   No. AsyncIO runs on a single thread, so the GIL is irrelevant to its concurrency model.
*   **"How do you debug a Deadlock?"**
    *   Use thread dumps to see which threads are waiting on which locks. Look for circular dependencies.
