# My CUDA Learning Journey 🧠⚡

Welcome! This repository is my personal sandbox where I am tracking my progress as I learn high-performance computing and GPU acceleration from scratch. 

> [!NOTE]
> **Disclaimer for the Records:** This is an educational, work-in-progress repository built for learning and personal tracking. It is **not** a professional production reference or enterprise-grade code library. I am exploring the foundational concepts, breaking things, and documenting what I learn along the way!

---

## 🚀 What I've Built So Far

### 1. 1D Vector Addition (`course_work/mp1.cu`)
*   **The Goal:** Learn how to move data from the CPU to the GPU and add millions of numbers in parallel.
*   **Core Concepts Mastered:** 
    *   Allocating VRAM using `cudaMalloc` and cleaning it up cleanly using `cudaFree`.
    *   Shipping data back and forth across the PCIe bus using `cudaMemcpy`.
    *   Writing my first custom GPU kernel (`__global__`) and calculating global indexes using `blockIdx.x * blockDim.x + threadIdx.x`.
*   **Why 256 threads?** I learned that the GPU processes things in chunks of 32 threads (called a "Warp"). Setting my block size to 256 keeps the hardware happy and aligned!

---

## 📐 Key Mental Models & Math Notes

To make sure I don't forget how the scaling logic works, here is the math I'm tracking for grid configurations:

If I have an array size $N$ that isn't a perfect multiple of my block size, I use integer ceiling division to calculate the grid size safely:
$$\text{GridSize} = \lfloor (N + \text{BlockSize} - 1) / \text{BlockSize} \rfloor$$

This ensures that if I have leftover elements, an extra block is automatically created to handle them, and the `if (i < n)` boundary check inside my kernel keeps threads from touching bad memory.

---

## 🛠️ My Local Setup Notes
*   **IDE:** Microsoft Visual Studio (using the CUDA Runtime templates)
*   **Target Architecture:** x64 Debug/Release
*   **Important Fix:** If Visual Studio complains about duplicate symbols (`LNK2005`), remember to right-click the default `kernel.cu` placeholder file and select **Exclude From Build**!

---

## 📈 Future Milestones I Want to Tackle
- [ ] Implement a 2D Matrix Multiplication kernel.
- [ ] Explore Shared Memory allocation to make memory access even faster.
- [ ] Measure actual execution times between a standard CPU `for` loop and my GPU kernels.
