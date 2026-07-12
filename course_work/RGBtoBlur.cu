#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <Windows.h>
#include <gdiplus.h>
#include "gdiplusimaging.h"

// image interpretation  using the image blur utility function to do the stuff
using namespace Gdiplus;
#pragma comment (lib, "Gdiplus.lib")

__global__ void ImageBlurring(unsigned char* rgb_Image, unsigned char* blurred_image, int Height, int Width, int blurSize) {
    // Thread assignment to process individual pixels
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < Width && row < Height) {
        // Track the pixel accumulation for each channel separately
        int r_val = 0, g_val = 0, b_val = 0;
        int pixels = 0;

        // Neighborhood inspection 
        for (int blurRow = -blurSize; blurRow <= blurSize; blurRow++) {
            for (int blurCol = -blurSize; blurCol <= blurSize; blurCol++) {

                int curRow = row + blurRow;
                int curCol = col + blurCol;

                // Edge validation check
                if (curRow >= 0 && curRow < Height && curCol >= 0 && curCol < Width) {
                    // Find the starting memory address of the target neighbor pixel
                    int neighborOffset = (curRow * Width + curCol) * 3;

                    r_val += rgb_Image[neighborOffset + 0]; // Red
                    g_val += rgb_Image[neighborOffset + 1]; // Green
                    b_val += rgb_Image[neighborOffset + 2]; // Blue

                    pixels++;
                }
            }
        }

        // Find the starting memory address of the output pixel for this thread
        int outOffset = (row * Width + col) * 3;

        // Write out the averaged values to each respective channel
        blurred_image[outOffset + 0] = (unsigned char)(r_val / pixels);
        blurred_image[outOffset + 1] = (unsigned char)(g_val / pixels);
        blurred_image[outOffset + 2] = (unsigned char)(b_val / pixels);
    }
}




__host__ void ImageBlurringHost(unsigned char* hrgb, unsigned char* hBlur, int Width, int Height, int Channels, int blurSize) {
    // Pointers to the GPU memory
    unsigned char* d_rgb = nullptr;
    unsigned char* d_Blur = nullptr;

    // Using the Channels variable to ensure correct dynamic sizing
    size_t rgb_size = (size_t)Width * Height * Channels * sizeof(unsigned char);
    size_t Blurred_size = (size_t)Width * Height * Channels * sizeof(unsigned char);

    // Allocating the GPU Memory
    if (cudaMalloc((void**)&d_rgb, rgb_size) != cudaSuccess) {
        fprintf(stderr, "unable to allocate the input image memory\n");
        return;
    }

    // allocating memory on the device 
    if (cudaMalloc((void**)&d_Blur, Blurred_size) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc Blur failed\n");
        cudaFree(d_rgb);
        return;
    }

    // Copying memory to device
    if (cudaMemcpy(d_rgb, hrgb, rgb_size, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "failed to copy the input from host to device (blur)\n");
        cudaFree(d_rgb);
        cudaFree(d_Blur);
        return;
    }

    // Defining the thread block sizes 
    dim3 blockSize(16, 16);

    // FIXED: Added parentheses to fix operator precedence for ceil division
    dim3 GridSize(
        (Width + blockSize.x - 1) / blockSize.x,
        (Height + blockSize.y - 1) / blockSize.y
    );

    // Kernel call 
    ImageBlurring << <GridSize, blockSize >> > (d_rgb, d_Blur, Height, Width, blurSize);

    // Aftermath clearances
    // Synchronize and check kernel launch
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed or device error: %s\n", cudaGetErrorString(err));
        cudaFree(d_rgb);
        cudaFree(d_Blur);
        return;
    }

    // Memory copy from device to host
    if (cudaMemcpy(hBlur, d_Blur, Blurred_size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to host failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        cudaFree(d_rgb);
        cudaFree(d_Blur);
        return;
    }

    // Allocated memory freeing 
    cudaFree(d_Blur);
    cudaFree(d_rgb);
}


//format encoder finder utility function


/// <summary>
/// this a utility to get the clsid of any type of image format
/// </summary>
/// <param name="format"></param>
/// <param name="p_clsid"></param>
/// <returns>if found</returns>
bool getEncoderClsid(const WCHAR* format, CLSID* p_clsid) {
    UINT num = 0, size = 0;
    GetImageEncodersSize(&num, &size);

    if (size == 0) return false;

    //allocate memory for encoders
    ImageCodecInfo* pImageCodecInfo = (ImageCodecInfo*)(malloc(size));
    if (pImageCodecInfo == NULL) return false;

    //retrieve image encoders
    if (GetImageEncoders(num, size, pImageCodecInfo) != Status::Ok) {
        fprintf(stderr, "couldnt retrive the image encoders");
        free(pImageCodecInfo);
        return false;
    }

    //search loop
    bool found = false;

    for (UINT j = 0; j < num; j++) {
        if (wcscmp(pImageCodecInfo[j].MimeType, format) == 0) {
            *p_clsid = pImageCodecInfo[j].Clsid;
            found = true;
            break;
        }
    }

    //free pairing so no memory leaks
    free(pImageCodecInfo);
    return found;
}

// here we will directly work with the GDI+ image pipeline
int main() {
    //intializing the GDI+ Engine
    GdiplusStartupInput gdiplusstartupinput;
    ULONG_PTR gdiplustoken;
    GdiplusStartup(&gdiplustoken, &gdiplusstartupinput, NULL);

    //Image Processing 
    Bitmap* input_bitmap = new Bitmap(L"C:\\Users\\Lenovo\\Downloads\\sample.bmp");
    int width = input_bitmap->GetWidth();
    int Height = input_bitmap->GetHeight();
    int Channels = 3;
    int BlurSize = 3;

    //rect definig 
    Rect rectangle(0, 0, width, Height);
    BitmapData bitmapdata;

    input_bitmap->LockBits(&rectangle, ImageLockModeRead, PixelFormat24bppRGB, &bitmapdata);


    // getting the input buffer interms of the inout array 
    unsigned char* inputImageBuffer = (unsigned char*)bitmapdata.Scan0;

    //getting the output buffer
    unsigned char* outputImageBuffer = new unsigned char[width * Height * Channels];
    
    // cuda function call 
    ImageBlurringHost(inputImageBuffer, outputImageBuffer, width, Height, Channels, BlurSize);

    input_bitmap->UnlockBits(&bitmapdata);

    //obtaining the output image 
    Bitmap* OutPutBitmap = new Bitmap(width, Height, PixelFormat24bppRGB);

    //output bitmap data
    BitmapData out_bitmapdata;
    OutPutBitmap->LockBits(&rectangle, ImageLockModeWrite, PixelFormat24bppRGB ,&out_bitmapdata);

    //captureing the output into the new buffer from scan0
    unsigned char* dest_buffer = (unsigned char* )out_bitmapdata.Scan0;
    int mem_width = Channels * width;


    //row stride padding safety (row by row copying)
    for (int r = 0; r < Height; r++) {
        memcpy(dest_buffer + (r * out_bitmapdata.Stride), outputImageBuffer + (r * mem_width), mem_width);
    }

    //unlock pairing 
    OutPutBitmap->UnlockBits(&out_bitmapdata);


    //png encoding 
    CLSID pngClsid;

    if (!getEncoderClsid(L"image/png", &pngClsid)) {
        fprintf(stderr, "unable to find the image clsid\n");
    }

    //image saving process
    Status status = OutPutBitmap->Save(L"C:\\Users\\Lenovo\\Downloads\\sample_blurred_order_3.png", &pngClsid, NULL);
    if (status == Ok) {
        printf("image blurred sucessfully");
    }
    else {
        fprintf(stderr, "could'nt save image " , status);
    }


    //memory cleaning (i always put these comments so i dont forget the cleaning part )
    delete[] outputImageBuffer;
    delete OutPutBitmap;
    delete input_bitmap;


    //shuting down the GDI+ Engine
    GdiplusShutdown(gdiplustoken);
    return 0;
}
