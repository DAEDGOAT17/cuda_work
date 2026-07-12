# My CUDA Learning 

Welcome! This repository is my personal sandbox where I am tracking my progress as I learn high-performance computing and GPU acceleration from scratch. 

> [!NOTE]
> **Disclaimer for the Records:** This is an educational, work-in-progress repository built for learning and personal tracking. It is **not** a professional production reference or enterprise-grade code library. I am exploring the foundational concepts, breaking things, and documenting what I learn along the way!

---

##  What I've Built So Far

### 1. 1D Vector Addition (`course_work/mp1.cu`) ✅ **Complete**
*   **The Goal:** Learn how to move data from the CPU to the GPU and add millions of numbers in parallel.
*   **Core Concepts Mastered:** 
    *   Allocating VRAM using `cudaMalloc` and cleaning it up cleanly using `cudaFree`.
    *   Shipping data back and forth across the PCIe bus using `cudaMemcpy`.
    *   Writing custom GPU kernels (`__global__`) and calculating global thread indexes using `blockIdx.x * blockDim.x + threadIdx.x`.
    *   Proper memory management for float arrays to preserve decimal precision.
*   **Key Implementation Details:**
    *   Successfully tested with **10,000 elements** running on the GPU.
    *   Uses **256 threads per block** (aligned with GPU warp sizes for efficiency).
    *   Grid configuration uses ceiling division: `(N + BlockSize - 1) / BlockSize` to ensure all elements are covered.
    *   Boundary checks (`if (i < n)`) prevent out-of-bounds memory access.
*   **Why 256 threads?** The GPU processes things in chunks of 32 threads (called a "Warp"). Setting the block size to 256 keeps the hardware happy and ensures efficient utilization!

---

### 2. RGB-to-Grayscale Image Conversion (`course_work/RGB2GRAYScale.cu`) ✅ **In Progress**
*   **The Goal:** Learn 2D thread indexing and how to process images in parallel across a 2D grid.
*   **Core Concepts Tackled:**
    *   2D thread block and grid configuration using `dim3` structs.
    *   Row-major indexing for 2D data: `offset = row * Width + col`.
    *   Multi-channel image processing (RGB → Grayscale using weighted average).
    *   Proper boundary checking in 2D: `if (row < Height && col < Width)`.
*   **Key Implementation Details:**
    *   Uses **16×16 thread blocks** (256 threads per block, same as MP1).
    *   Implements the standard grayscale formula: `0.299R + 0.587G + 0.114B`.
    *   Handles BGR memory layout from GDI+ (Windows image library).
    *   Grid dimensions computed safely with ceiling division for both X and Y axes.
*   **Windows Integration:** Integrated with GDI+ (`gdiplus.h`) for potential image I/O.
*   **Status:** Kernel structure complete; next step is to wire up actual image file loading/saving.

---

##  Key Mental Models & Math Notes

To make sure I don't forget how the scaling logic works, here is the math I'm tracking for grid configurations:

### 1D Grid Configuration (Vector Addition)
If I have an array size $N$ that isn't a perfect multiple of my block size, I use integer ceiling division to calculate the grid size safely:
$$\text{GridSize} = \lfloor (N + \text{BlockSize} - 1) / \text{BlockSize} \rfloor$$

This ensures that if I have leftover elements, an extra block is automatically created to handle them, and the `if (i < n)` boundary check inside my kernel keeps threads from touching bad memory.

### 2D Grid Configuration (Image Processing)
For 2D data like images, I extend this to both dimensions:
$$\text{GridSize}_X = \lfloor (\text{Width} + \text{BlockSize}_X - 1) / \text{BlockSize}_X \rfloor$$
$$\text{GridSize}_Y = \lfloor (\text{Height} + \text{BlockSize}_Y - 1) / \text{BlockSize}_Y \rfloor$$

Each thread computes its position using:
- `col = threadIdx.x + blockIdx.x * blockDim.x` (X-axis / Width)
- `row = threadIdx.y + blockIdx.y * blockDim.y` (Y-axis / Height)

---

##  My Local Setup Notes
*   **IDE:** Microsoft Visual Studio 2022
*   **CUDA Version:** CUDA 13.3
*   **Target Architecture:** x64 Debug/Release
*   **Dependencies:** 
    *   `cudart_static.lib` (CUDA Runtime)
    *   GDI+ (`gdiplus.lib`) for future image processing
*   **Important Fix:** If Visual Studio complains about duplicate symbols (`LNK2005`), right-click the default `kernel.cu` placeholder file and select **Exclude From Build**!
*   **Project Structure:**
    - `course_work/mp1.cu` — 1D vector addition (currently set as active project)
    - `course_work/RGB2GRAYScale.cu` — 2D image processing (in development)
    - `course_work/kernel.cu` — Original template file (excluded from build to avoid conflicts)

---

##  Lessons Learned Along the Way

1. **Memory Layout Matters:** When working with multi-channel data (like RGB images), always be careful about the byte ordering. GDI+ uses BGR, not RGB!
2. **Boundary Checking is Non-Negotiable:** If your grid doesn't divide evenly, you'll have "leftover" threads. Always check `if (thread_index < valid_range)` to avoid undefined behavior.
3. **Block Size Alignment:** Sticking to multiples of the warp size (32) makes kernel launches more efficient. 256 = 8 warps is a sweet spot.
4. **CPU ↔ GPU Communication is Expensive:** The PCIe bus is slower than GPU memory. Always think about whether data needs to shuttle back and forth.

---

##  Future Milestones I Want to Tackle

- [ ] Wire up image file I/O for RGB2GRAYScale (load `.bmp` or `.jpg`, save output).
- [ ] Implement a 2D Matrix Multiplication kernel.
- [ ] Explore Shared Memory allocation to make memory access even faster.
- [ ] Implement proper **CUDA error checking** (wrap all CUDA calls in a macro that validates return codes).
- [ ] Measure actual execution times: GPU kernel vs. CPU `for` loop vs. GPU with different block sizes.
- [ ] Experiment with **constant memory** for read-only data (like weights in the grayscale formula).
- [ ] Learn about **atomic operations** for reduction-style problems (e.g., summing all pixels).

---

##  References & Resources

*   **CUDA Programming Guide:** Understanding grid/block indexing and memory hierarchy.
*   **NVIDIA Best Practices:** Why 256 threads per block is a good default; warp size implications.
*   **Windows GDI+:** For image encoding/decoding (future integration).
