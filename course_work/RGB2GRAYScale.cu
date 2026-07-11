#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <Windows.h>
#include <gdiplus.h>
#include "gdiplusimaging.h"

// image interpretation
using namespace Gdiplus;
#pragma comment (lib, "Gdiplus.lib")

//kernel function
/// <summary>
/// the single program to be executed on multiple threads inside the GPU
/// </summary>
/// <param name="rgb_Image"></param>
/// <param name="Gray_Image"></param>
/// <param name="Height"></param>
/// <param name="Width"></param>
/// <param name="Channels"></param>
/// <returns></returns>
__global__ void Rgb2Gray(unsigned char* rgb_Image, unsigned char* Gray_Image, int Height, int Width, int Channels) {
    int col = threadIdx.x + blockIdx.x * blockDim.x; // X-axis (Width)
    int row = threadIdx.y + blockIdx.y * blockDim.y; // Y-axis (Height)

    // FIXED: Correct boundary check
    if (row < Height && col < Width) {
        int gray_offset = row * Width + col;
        int rgb_offset = gray_offset * Channels;

        // GDI+ uses BGR memory layout layout:
        unsigned char b = rgb_Image[rgb_offset];
        unsigned char g = rgb_Image[rgb_offset + 1];
        unsigned char r = rgb_Image[rgb_offset + 2];

        Gray_Image[gray_offset] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

//host instructions
/// <summary>
/// host instructions are derived and written down here
/// </summary>
/// <param name="hrgb"></param>
/// <param name="hGray"></param>
/// <param name="Height"></param>
/// <param name="Width"></param>
/// <param name="Channels"></param>
/// <returns></returns>
__host__ void Rgb2Gray_Host(unsigned char* hrgb, unsigned char* hGray, int Height, int Width, int Channels) {
    // assigning pointers for GPU memory 
    unsigned char* d_rgb, * d_gray;
    int rgb_size = Width * Height * Channels * sizeof(unsigned char);
    int gray_size = Width * Height * sizeof(unsigned char);

    //allocating the GPU memory
    if (cudaMalloc((void**)&d_rgb, rgb_size) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_rgb failed\n");
        return;
    }
    if (cudaMalloc((void**)&d_gray, gray_size) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_gray failed\n");
        cudaFree(d_rgb);
        return;
    }

    //copy the data from host to device 
    if (cudaMemcpy(d_rgb, hrgb, rgb_size, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to device failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        cudaFree(d_rgb);
        cudaFree(d_gray);
        return;
    }

    //defining the thread block and dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((Width + blockSize.x - 1) / blockSize.x,
        (Height + blockSize.y - 1) / blockSize.y
    );

    //kernel commands 
    Rgb2Gray << < gridSize, blockSize >> > (d_rgb, d_gray, Height, Width, Channels);

    // synchronize and check kernel launch
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed or device error: %s\n", cudaGetErrorString(err));
        cudaFree(d_rgb);
        cudaFree(d_gray);
        return;
    }

    // copy processed data from device to host
    if (cudaMemcpy(hGray, d_gray, gray_size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to host failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        cudaFree(d_rgb);
        cudaFree(d_gray);
        return;
    }

    //free the allocated memory
    cudaFree(d_rgb);
    cudaFree(d_gray);
}

/// <summary>
/// a finction to handle all the image to array processing tasks using the GDI+ standard utility
/// </summary>
/// <param name="Input_image"></param>
/// <param name="Output_image"></param>
/// <returns></returns>
bool ImageProcessing(const wchar_t* Input_image, const wchar_t* Output_image) {
    //initialize GDI+
    GdiplusStartupInput gdiplusstartupinput;
    ULONG_PTR gdiplustoken;

    //failure prevention 
    if (GdiplusStartup(&gdiplustoken, &gdiplusstartupinput, NULL) != Ok) {
        printf("failed to start  the GDIplus instance");
        return false;
    }

    bool success = false;
    {
        Bitmap bitmap(Input_image);
        // failure prevention 
        if (bitmap.GetLastStatus() != Ok) {
            printf("failed to load an image\n");
            GdiplusShutdown(gdiplustoken);
            return false;
        }

        //image info 
        int height = bitmap.GetHeight();
        int width = bitmap.GetWidth();
        int channels = 3; // adding the three channels 

        // layingup bitmap data for GDI+ to work effectively 
        BitmapData bitmapdata;
        Rect rect(0, 0, width, height);
        bitmap.LockBits(&rect, ImageLockModeRead, PixelFormat24bppRGB, &bitmapdata);
        unsigned char* hrgb = (unsigned char*)bitmapdata.Scan0;

        // Copy bitmap data into a contiguous RGB buffer because GDI+ Scan0 may have row padding (stride)
        int stride = bitmapdata.Stride;
        int rgb_size = width * height * channels;
        unsigned char* hrgb_contig = (unsigned char*)malloc(rgb_size);
        if (hrgb_contig == NULL) {
            printf("failed to allocate contiguous rgb buffer\n");
            bitmap.UnlockBits(&bitmapdata);
            GdiplusShutdown(gdiplustoken);
            return false;
        }
        for (int y = 0; y < height; y++) {
            memcpy(hrgb_contig + (y * width * channels), hrgb + (y * stride), width * channels);
        }

        //memory allocation for output gray scale image 
        unsigned char* h_gray = (unsigned char*)malloc(width * height);

        //performing the conversion process
        if (h_gray != NULL) {

            // callling the host function or the coversion process
            Rgb2Gray_Host(hrgb_contig, h_gray, height, width, channels);

            // new bitmap for the allocation on the gdi+ to work on gray 
            Bitmap gray_bitmap(width, height, PixelFormat8bppIndexed);

            //setting the pallete size as gdi+ requires 256 shades of gray 
            int pallete_size = sizeof(ColorPalette) + (256 * sizeof(ARGB));
            ColorPalette* pallete = (ColorPalette*)malloc(pallete_size);
            pallete->Flags = PaletteFlagsGrayScale;
            pallete->Count = 256;
            // genrating gray scale palette for the bitmap to be painted 
            for (int i = 0; i < 256; i++) {
                pallete->Entries[i] = Color::MakeARGB(255, i, i, i);
            }

            //
            gray_bitmap.SetPalette(pallete);
            free(pallete);
            //locking the o/p bits and getting the data out of the cpu
            BitmapData out_data;
            gray_bitmap.LockBits(&rect, ImageLockModeWrite, PixelFormat8bppIndexed, &out_data);

            //handleing the row padding 
            for (int i = 0; i < height; i++) {
                memcpy((char*)out_data.Scan0 + (i * out_data.Stride), h_gray + (i * width), width);
            }

            //unlock pairing 
            gray_bitmap.UnlockBits(&out_data);

            //conversion and saving into the jpeg format
            CLSID jpegclsid;
            CLSIDFromString(L"{557CF406-1A04-11D3-9A73-0000F81EF32E}", &jpegclsid);

            //saving and failure check
            if (gray_bitmap.Save(Output_image, &jpegclsid, NULL) == Ok) {
                printf("SUCESSFULLY PROCESSED AND SAVED IMAGE \n ");
                success = true;
            }
            else {
                printf("FAILED TO PROCESS AND SAVE THE IMAGE \n ");
            }

            free(h_gray);
        }

        free(hrgb_contig);
        //unlock pairing 
        bitmap.UnlockBits(&bitmapdata);
    }
    //shutting down the GDI+ tools
    GdiplusShutdown(gdiplustoken);
    return success;
}

//main entry point 
int main() {
    // FIXED: Escaped the backslashes in the absolute paths
    const wchar_t* inputPath = L"C:\\Users\\Lenovo\\Downloads\\sample.bmp";
    const wchar_t* outputPath = L"C:\\Users\\Lenovo\\Downloads\\output.bmp";

    printf("Starting CUDA Grayscale conversion...\n");
    printf("Target Image: C:\\Users\\Lenovo\\Downloads\\sample.bmp\n");
    printf("--------------------------------------------------\n");

    bool success = ImageProcessing(inputPath, outputPath);

    if (success) {
        printf("--------------------------------------------------\n");
        printf("Task completed successfully!\n");
        printf("Grayscale image saved in your Downloads folder.\n");
    }
    else {
        printf("--------------------------------------------------\n");
        printf("Image processing failed.\n");
        return -1;
    }

    return 0;
}