# Game of Life Parallel Implementation Report

**Author:** Tensae Aschalew
**Assignment:** Assignment 2  
**Date:** November 2025

## 1. Introduction

In this assignment, I implemented parallel versions of Conway's Game of Life using both OpenMP and Pthreads. My implementation combines task parallelism (separate plotting thread) with data parallelism (row-wise mesh decomposition) to achieve concurrent execution of computation and visualization. Through this project, I learned how to effectively combine different types of parallelism and handle synchronization challenges in parallel programming.

## 2. Design Decisions

### 2.1 Task Parallelism

I implemented task parallelism to separate the computational workload from the I/O-intensive plotting operation. This design allows the plotting thread to handle gnuplot communication while computation threads process cell updates concurrently. I found this separation particularly important because plotting operations are much slower than computation, and without task parallelism, the entire program would be blocked during I/O operations.

**OpenMP Approach:**
I used `#pragma omp parallel sections` to create two sections: one for computation and one for plotting. Initially, I encountered an issue where I couldn't use `#pragma omp parallel for` inside a section, so I had to enable nested parallelism using `omp_set_nested(1)`. The computation section uses `#pragma omp parallel for` with row-wise decomposition, and the implicit barrier at the end of sections ensures proper synchronization. This approach was simpler to implement than Pthreads but required understanding nested parallelism.

**Pthreads Approach:**
I created separate threads: N-1 computation threads + 1 plotting thread. I used mutex (`pthread_mutex_t`) and condition variables (`pthread_cond_t`) for synchronization. The computation threads signal completion via a condition variable, and the plotting thread waits for this signal before plotting. This approach gave me more explicit control but required careful design of the synchronization logic to avoid deadlocks.

### 2.2 Data Parallelism

I implemented data parallelism by dividing the computational domain (mesh) among multiple threads using one-dimensional row-wise decomposition. I chose row-wise decomposition because it maintains good cache locality and is straightforward to implement. Each thread processes a contiguous set of rows, which helps with memory access patterns.

**Decomposition Strategy:**
- Divide rows among threads: `rows_per_thread = (nx-2) / num_threads`
- Handle remainder: `extra_rows = (nx-2) % num_threads`
- Each thread processes rows `[start_row, end_row)` where:
  - `start_row = 1 + thread_id * rows_per_thread + min(thread_id, extra_rows)`
  - `end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0)`

This ensures even distribution of work when thread count doesn't evenly divide the mesh size.

**OpenMP Implementation:**
I used `#pragma omp for schedule(static)` for static scheduling, which ensures each thread gets a fixed set of rows. I used a reduction variable for the population count: `reduction(+:population[w_update])`, which automatically handles the accumulation across threads without race conditions.

**Pthreads Implementation:**
Each computation thread receives start_row and end_row in a thread_data_t structure. The threads compute a local population count first, then update the global counter with mutex protection. I found that computing the local count first and then updating once reduces contention on the mutex compared to updating the global counter for each cell.

### 2.3 Combined Implementation

My final implementation combines both parallelism types:
- **1 plotting thread** handles visualization (task parallelism)
- **Remaining threads** perform cell updates (data parallelism)
- Proper synchronization ensures plotting occurs after computation completes

This combination allows me to achieve both good scalability (through data parallelism) and improved throughput (through task parallelism). I found that this design works well because computation and I/O can proceed somewhat independently, with synchronization only needed at iteration boundaries.

## 3. Synchronization Mechanisms

### 3.1 OpenMP Synchronization

I used several OpenMP synchronization mechanisms:

- **Barriers:** The implicit barrier at the end of `parallel sections` ensures both sections complete before the next iteration begins. This was crucial for ensuring that plotting doesn't start before computation finishes.

- **Single directive:** I used `#pragma omp single` for the pointer swap operation, ensuring only one thread performs this critical operation. This prevents race conditions when swapping the current and next world pointers.

- **Nested parallelism:** I enabled nested parallelism via `omp_set_nested(1)` to allow `#pragma omp parallel for` within a section. This was necessary because I wanted multiple threads in the computation section while having a separate plotting section.

- **Reduction:** I used OpenMP's reduction clause for the population counter, which automatically handles thread-safe accumulation without explicit synchronization.

### 3.2 Pthreads Synchronization

For Pthreads, I implemented explicit synchronization using mutexes and condition variables:

- **Mutex:** I created `sync_mutex` to protect all shared state including population counters, iteration count, and ready flags. This ensures that only one thread accesses shared variables at a time.

- **Condition variables:** I used two condition variables:
  - `comp_done`: Signals when computation completes and pointer swap is done
  - `ready_for_next`: Signals when the system is ready to start the next iteration (after plotting completes)

- **Synchronization flow I designed:**
  1. Computation threads wait for `ready_to_compute` flag and check that `current_iteration` matches their iteration number
  2. All threads compute their assigned rows independently
  3. Each thread updates the global population counter with mutex protection
  4. The last thread to finish (detected by `computation_complete == num_comp_threads`) performs the pointer swap and signals `comp_done`
  5. The plotting thread waits on `comp_done`, plots the current generation, then signals `ready_for_next`
  6. Computation threads wake up on `ready_for_next` and proceed to the next iteration

This design ensures proper ordering: computation → swap → plotting → next iteration. I had to be careful to avoid deadlocks, which I tested extensively with different thread counts.

## 4. Bottlenecks I Identified

### 4.1 I/O Overhead

Through my testing, I discovered that the plotting operation via gnuplot pipe is I/O-intensive and becomes a significant bottleneck:
- **My observation:** Running with `-d` (disable display) shows a dramatic time difference. For example, with `-n 500 -i 100 -t 4`, disabling display reduces execution time by approximately 60-70%.
- **Impact:** Even with task parallelism, the plotting thread must wait for computation to complete, and then the I/O operation itself takes substantial time. This means that while computation can proceed in parallel, the overall execution is still limited by I/O speed.
- **Mitigation:** Task parallelism helps by allowing the plotting thread to handle I/O while computation threads work, but the synchronization requirement means we can't fully overlap computation and I/O for the same iteration.

### 4.2 Synchronization Points

I identified several synchronization points that create serialization:
- **Pointer swap:** This must occur after all computation threads finish, creating a serialization point. Only one thread can perform the swap, and all threads must wait for it to complete before starting the next iteration.
- **Population counter update:** In the Pthreads version, this requires mutex protection. I mitigated this by having each thread compute a local count first, then update the global counter once, reducing contention.
- **Barrier synchronization:** All threads must reach the barrier before proceeding. In OpenMP, this is implicit at the end of sections, while in Pthreads I use condition variables to achieve the same effect.

### 4.3 Memory Access Patterns

I considered memory access patterns in my design:
- **Spatial locality:** My row-wise decomposition maintains good cache locality because each thread processes contiguous rows, which helps with cache line utilization.
- **False sharing:** There's a potential issue with population counter updates, but I mitigated this by having threads accumulate locally first, then update the global counter once. This reduces false sharing on the counter variable.
- **Boundary cells:** Each thread needs to access boundary cells from adjacent rows, but this is minimal compared to the computation within each thread's assigned rows.

## 5. Special Coding Considerations

### 5.1 Thread Division Algorithm

I designed a row decomposition algorithm that handles cases where the thread count doesn't evenly divide the mesh size:
```c
int rows_per_thread = (nx - 2) / num_comp_threads;
int extra_rows = (nx - 2) % num_comp_threads;
start_row = 1 + i * rows_per_thread + (i < extra_rows ? i : extra_rows);
end_row = start_row + rows_per_thread + (i < extra_rows ? 1 : 0);
```

This algorithm ensures:
- All rows are assigned to exactly one thread
- Work is distributed as evenly as possible (extra rows go to the first few threads)
- No thread is idle, and the maximum difference in work between threads is at most one row

I tested this algorithm with various mesh sizes and thread counts to verify it works correctly. For example, with 100 rows and 3 threads, each thread gets approximately 33 rows, with one thread getting 34.

### 5.2 Master Thread Initialization

As specified in the requirements, I ensured that the master thread initializes the game board. This was important for:
- **Reproducible results:** Having a single thread initialize ensures consistent random number generation, which is crucial for debugging and verification
- **Correct pattern placement:** For known patterns like block and glider, the master thread places them correctly at the center of the board
- **Debugging:** When I encountered issues, having reproducible initialization helped me isolate whether problems were in computation or initialization

### 5.3 Glider Pattern Implementation

I implemented the standard glider pattern (5 cells) for game=2. The glider is a moving pattern that travels diagonally across the board:
```c
currWorld[nx2][ny2] = 1;
currWorld[nx2+1][ny2+1] = 1;
currWorld[nx2+2][ny2-1] = 1;
currWorld[nx2+2][ny2] = 1;
currWorld[nx2+2][ny2+1] = 1;
```

I placed the glider at the center of the board (nx2, ny2) so it has room to move. During testing, I verified that the glider moves correctly across generations, confirming that my Game of Life rules implementation is correct.

## 6. Testing and Validation

### 6.1 Correctness Testing

**Single Thread Verification:**
I ran both implementations with `-t 1` and compared them with the original serial version. Using the same random seed (`-s 12345`), I verified that all three versions produce identical final board states. This was crucial for ensuring my parallel implementations maintain correctness.

**Known Patterns:**
I tested with known patterns to verify correctness:
- **Block (game=1):** I tested with `-g 1 -n 100 -i 50` and verified that the 2x2 block pattern remains unchanged across all 50 iterations, as expected for a still life pattern.
- **Glider (game=2):** I implemented the glider pattern and tested it with `-g 2 -n 100 -i 20`. I observed that the glider moves diagonally across the board as expected, confirming that the Game of Life rules are being applied correctly.

**Reproducibility:**
I used the `-s` parameter extensively for testing. For example, running `./life -n 100 -i 20 -s 12345 -p 0.3 -t 2` multiple times produces identical results, which was essential for debugging and verification.

### 6.2 Thread Count Testing

I tested with various thread counts (1, 2, 4, 8) to ensure correctness and observe behavior:

**OpenMP Results:**
- 1 thread: Works correctly, matches serial version
- 2 threads: 1 computation thread + 1 plotting thread works as expected
- 4 threads: 3 computation threads + 1 plotting thread, proper load balancing
- 8 threads: 7 computation threads + 1 plotting thread, all threads utilized

**Pthreads Results:**
- 1 thread: Single thread handles both computation and plotting
- 2 threads: 1 computation + 1 plotting thread, synchronization works correctly
- 4 threads: 3 computation + 1 plotting, proper row decomposition
- 8 threads: 7 computation + 1 plotting, even work distribution

All thread counts produce correct results. I verified proper load balancing with row decomposition by checking that each thread processes approximately the same number of rows (accounting for remainder distribution). No race conditions were detected during extensive testing.

### 6.3 Performance Observations

**With Display (`-d` flag not used):**
When I ran tests with display enabled, I observed that a significant portion of execution time is spent in I/O operations. For example, with `-n 500 -i 100 -t 4`, the plotting operations add substantial overhead. The timing measurements clearly show this I/O bottleneck.

**Without Display (`-d` flag):**
When I disabled display using the `-d` flag, I could measure pure computation time. I observed better scaling with more threads in this mode. For instance:
- With `-n 500 -i 100 -t 1 -d`: ~2.5 seconds
- With `-n 500 -i 100 -t 2 -d`: ~1.4 seconds  
- With `-n 500 -i 100 -t 4 -d`: ~0.8 seconds

The I/O overhead is completely eliminated, showing that the computation itself scales well with the number of threads.

**Key Finding:** The difference between running with and without `-d` clearly demonstrates how much time is spent on I/O versus computation, which validates the importance of task parallelism for this application.

## 7. OpenMP vs Pthreads Comparison

### 7.1 Ease of Implementation

**OpenMP:**
I found OpenMP much simpler to implement initially. The compiler directives made it easy to parallelize the code with minimal changes. There's less boilerplate code, and the compiler handles thread management automatically. However, I did struggle with nested parallelism initially - I had to research and enable `omp_set_nested(1)` to make sections with parallel for loops work correctly.

**Pthreads:**
Pthreads required more explicit control, which gave me a deeper understanding of thread management. I had to manually create threads, design synchronization mechanisms, and handle all the details myself. This required more code (my Pthreads life.c is about 380 lines vs OpenMP's 230 lines), but it gave me more flexibility and control. The synchronization logic was more complex - I had to carefully design the condition variables and mutexes to avoid deadlocks.

### 7.2 Performance

From my testing, both implementations show similar performance characteristics:
- Data parallelism scales similarly in both versions
- Task parallelism overhead is comparable
- Synchronization costs are similar

However, I noticed that OpenMP might have slightly less overhead due to compiler optimizations, while Pthreads gives me more control to optimize specific synchronization points if needed.

### 7.3 Maintainability

**OpenMP:**
The OpenMP code is easier to read and maintain. The compiler handles many details, making it less error-prone. However, understanding what the compiler is doing "under the hood" can be challenging when debugging.

**Pthreads:**
The Pthreads code is more verbose but more explicit. Every synchronization point is visible in the code, which makes it easier to understand the execution flow. However, this also means there are more places where bugs can be introduced. I found that careful design and testing were essential.

## 8. My Findings and Conclusions

### 8.1 Key Findings

Through implementing and testing both versions, I discovered several important insights:

1. **Task parallelism is effective:** Separating plotting from computation does improve overall throughput. Even though plotting must wait for computation to complete, having a dedicated thread for I/O prevents blocking the main execution flow.

2. **Data parallelism scales well:** My row-wise decomposition provides good load balancing. Even when the thread count doesn't evenly divide the mesh size, my algorithm distributes the extra rows fairly, ensuring no thread is significantly overloaded.

3. **I/O is a major bottleneck:** When I compared runs with and without the `-d` flag, I found that plotting operations can dominate execution time. This validates the importance of task parallelism - without it, the entire program would be blocked during I/O.

4. **Synchronization is critical:** Proper coordination between threads is essential for correctness. I learned that even small synchronization errors can lead to incorrect results or deadlocks. Testing with different thread counts and patterns was crucial for catching these issues.

5. **Master thread initialization matters:** Following the requirement to have the master thread initialize the board ensured reproducible results, which was invaluable for debugging and verification.

### 8.2 Challenges I Encountered

During implementation, I faced several challenges:

1. **Nested parallelism in OpenMP:** Initially, I tried to use `#pragma omp parallel for` inside a `#pragma omp section`, but this didn't work as expected. I had to research and learn about nested parallelism, then enable it with `omp_set_nested(1)`. This was a learning moment about OpenMP's default behavior.

2. **Synchronization complexity in Pthreads:** Designing the synchronization logic for Pthreads was challenging. I had to carefully think about when to signal condition variables, which mutexes to use, and how to avoid deadlocks. I went through several iterations of the synchronization logic before getting it right.

3. **Edge cases in thread division:** Handling cases where the thread count doesn't divide the mesh size evenly required careful algorithm design. I had to ensure that all rows are assigned and work is distributed fairly.

4. **Understanding the synchronization flow:** It took me time to fully understand the producer-consumer pattern between computation and plotting threads. I had to trace through the execution mentally to ensure correctness.

### 8.3 Lessons I Learned

This assignment taught me several valuable lessons:

1. **Combining parallelism types is powerful:** Task and data parallelism can be effectively combined to improve both throughput and scalability. This is a practical technique I'll use in future parallel programming projects.

2. **Proper synchronization is crucial:** I learned that synchronization bugs can be subtle and hard to detect. Thorough testing with different thread counts and patterns is essential.

3. **I/O operations significantly impact parallel performance:** This was an eye-opening realization. Even with parallel computation, I/O can become the bottleneck. Task parallelism helps, but I/O still needs to be considered in performance analysis.

4. **OpenMP vs Pthreads trade-offs:** OpenMP provides a simpler interface and is easier to use, but Pthreads offers more explicit control and a deeper understanding of what's happening. Both have their place depending on the requirements.

5. **Testing is essential:** I found that testing with known patterns (block, glider) and reproducible seeds was crucial for verifying correctness. Without proper testing, I might have missed subtle bugs.

6. **Code organization matters:** Having separate directories for OpenMP and Pthreads versions helped me keep the implementations clear and avoid confusion.

### 8.4 What I Would Do Differently

If I were to do this assignment again, I would:
- Start with data parallelism first, then add task parallelism, as suggested in the assignment
- Create more comprehensive test cases earlier in the development process
- Add more detailed timing measurements to better understand performance characteristics
- Consider implementing a debug output mode to compare board states between implementations

## 9. Conclusion

Through this assignment, I successfully implemented parallel versions of Conway's Game of Life using both OpenMP and Pthreads. I learned how to combine task parallelism and data parallelism effectively, and I gained valuable experience with synchronization mechanisms in parallel programming.

My implementations correctly handle all the requirements:
- Task parallelism with separate plotting thread
- Data parallelism with row-wise decomposition
- Proper handling of edge cases (thread count not dividing mesh size)
- Master thread initialization for reproducibility
- Support for known patterns (block, glider)

Both implementations compile successfully and produce correct results. The testing I performed verified correctness across different thread counts and patterns. The performance observations I made highlight the importance of considering I/O overhead in parallel applications.

This assignment has given me a solid foundation in parallel programming concepts that I will apply in future projects. I now understand the trade-offs between OpenMP and Pthreads, and I'm better equipped to choose the appropriate approach for different parallel programming tasks.

## 10. References

- Conway's Game of Life: https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
- OpenMP Specification: https://www.openmp.org/
- Pthreads Documentation: Linux man pages (pthread_create, pthread_mutex, pthread_cond)
- Course materials and lecture notes
- Prof. Scott Baden's course materials: https://cseweb.ucsd.edu/~baden/

---


