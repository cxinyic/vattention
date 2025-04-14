#include <iostream>
#include <vector>
#include <chrono>
#include <cuda.h>
#include <iomanip>
#include <thread>
#include <mutex>
#include <atomic>

// Error checking macro for CUDA driver API
#define CHECK_CUDA(x) \
    do { \
        CUresult res = x; \
        if (res != CUDA_SUCCESS) { \
            const char *errStr = NULL; \
            (void)cuGetErrorString(res, &errStr); \
            std::cerr << __FILE__ << ':' << __LINE__ << ' ' << #x \
                      << " failed (" << (unsigned)res << "): " << errStr << std::endl; \
            exit(1); \
        } \
    } while (0)

// Benchmark parameters
constexpr size_t PAGE_SIZE = 256*1024; // 2MB
constexpr size_t NUM_PAGES_PER_GPU = 32000; // 15,000 pages per GPU
constexpr int BATCH_SIZE = 1000; // Report progress every BATCH_SIZE pages
constexpr int NUM_GPUS = 4; // Number of GPUs to use

// Mutex for thread-safe console output
std::mutex console_mutex;

// Atomic counter for progress tracking
std::atomic<int> progress_counter(0);

// Function to initialize CUDA context for a specific device
CUcontext init_cuda_context(int deviceId) {
    CUdevice device;
    CHECK_CUDA(cuDeviceGet(&device, deviceId));
    
    // Create context
    CUcontext context;
    CHECK_CUDA(cuCtxCreate(&context, 0, device));
    
    // Get device name
    char deviceName[256];
    CHECK_CUDA(cuDeviceGetName(deviceName, sizeof(deviceName), device));
    
    {
        std::lock_guard<std::mutex> lock(console_mutex);
        std::cout << "CUDA context initialized on device " << deviceId << ": " << deviceName << std::endl;
    }
    
    return context;
}

// Worker function to benchmark memory operations on a single GPU
void benchmark_gpu_worker(int deviceId) {
    // Create context for this device
    CUcontext context = init_cuda_context(deviceId);
    
    // Setup allocation properties
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = deviceId;
    
    // Setup access descriptor
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    accessDesc.location.id = deviceId;
    
    // Vector to store allocation handles for this GPU
    std::vector<CUmemGenericAllocationHandle> cuda_pages;
    cuda_pages.reserve(NUM_PAGES_PER_GPU);
    
    // Get granularity for memory allocations
    size_t granularity;
    CHECK_CUDA(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    
    {
        std::lock_guard<std::mutex> lock(console_mutex);
        std::cout << "GPU " << deviceId << " - Memory allocation granularity: " << granularity << " bytes" << std::endl;
    }
    
    // ======= Combined Operations (Allocate + Push) =======
    auto alloc_start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < NUM_PAGES_PER_GPU; i++) {
        CUmemGenericAllocationHandle handle;
        CHECK_CUDA(cuMemCreate(&handle, PAGE_SIZE, &prop, 0));
        cuda_pages.push_back(handle);
        
        // Update progress counter
        int current_progress = ++progress_counter;
        
        // Print progress every BATCH_SIZE pages (across all GPUs)
        if (current_progress % BATCH_SIZE == 0) {
            std::lock_guard<std::mutex> lock(console_mutex);
            std::cout << "Created " << current_progress << " pages across all GPUs..." << std::endl;
        }
    }
    
    auto alloc_end = std::chrono::high_resolution_clock::now();
    auto alloc_duration = std::chrono::duration_cast<std::chrono::milliseconds>(alloc_end - alloc_start).count();
    
    {
        std::lock_guard<std::mutex> lock(console_mutex);
        std::cout << "GPU " << deviceId << " - Allocation completed in " << alloc_duration << " ms" << std::endl;
        std::cout << "GPU " << deviceId << " - Average alloc time per page: " 
                  << (alloc_duration / (double)NUM_PAGES_PER_GPU) << " ms" << std::endl;
    }
    
    // Reset progress counter for release phase
    if (deviceId == 0) {
        progress_counter.store(0);
    }
    
    // Barrier to ensure all GPUs have finished allocation before starting release
    // This is a simple barrier using thread synchronization
    static std::atomic<int> barrier(0);
    barrier++;
    while (barrier < NUM_GPUS) {
        std::this_thread::yield();
    }
    
    // ======= Combined Operations (Release + Pop) =======
    auto release_start = std::chrono::high_resolution_clock::now();
    
    while (!cuda_pages.empty()) {
        CUmemGenericAllocationHandle handle = cuda_pages.back();
        CHECK_CUDA(cuMemRelease(handle));
        cuda_pages.pop_back();
        
        // Update progress counter
        int current_progress = ++progress_counter;
        
        // Print progress every BATCH_SIZE pages (across all GPUs)
        if (current_progress % BATCH_SIZE == 0) {
            std::lock_guard<std::mutex> lock(console_mutex);
            std::cout << "Released " << current_progress << " pages across all GPUs..." << std::endl;
        }
    }
    
    auto release_end = std::chrono::high_resolution_clock::now();
    auto release_duration = std::chrono::duration_cast<std::chrono::milliseconds>(release_end - release_start).count();
    
    {
        std::lock_guard<std::mutex> lock(console_mutex);
        std::cout << "GPU " << deviceId << " - Release completed in " << release_duration << " ms" << std::endl;
        std::cout << "GPU " << deviceId << " - Average release time per page: " 
                  << (release_duration / (double)NUM_PAGES_PER_GPU) << " ms" << std::endl;
        std::cout << "GPU " << deviceId << " - Ratio (Release/Alloc): " 
                  << (double)release_duration / alloc_duration << std::endl;
    }
    
    // Destroy context when done
    CHECK_CUDA(cuCtxDestroy(context));
}

void multi_gpu_benchmark() {
    // Initialize CUDA driver
    CHECK_CUDA(cuInit(0));
    
    // Get device count
    int deviceCount = 0;
    CHECK_CUDA(cuDeviceGetCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        exit(1);
    }
    
    if (deviceCount < NUM_GPUS) {
        std::cerr << "Warning: Only " << deviceCount << " GPUs available, but benchmark is configured for " 
                  << NUM_GPUS << " GPUs." << std::endl;
        std::cerr << "Running with " << deviceCount << " GPUs instead." << std::endl;
    }
    
    const int actual_gpus = std::min(deviceCount, NUM_GPUS);
    std::cout << "=== Starting Multi-GPU Benchmark with " << actual_gpus << " GPUs ===" << std::endl;
    std::cout << "Page size: " << (PAGE_SIZE / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Pages per GPU: " << NUM_PAGES_PER_GPU << std::endl;
    std::cout << "Total pages: " << (NUM_PAGES_PER_GPU * actual_gpus) << std::endl;
    std::cout << "Total memory: " << (PAGE_SIZE * NUM_PAGES_PER_GPU * actual_gpus / (1024.0 * 1024 * 1024)) << " GB" << std::endl;
    
    // Start timing overall benchmark
    auto benchmark_start = std::chrono::high_resolution_clock::now();
    
    // Create threads for each GPU
    std::vector<std::thread> threads;
    for (int i = 0; i < actual_gpus; i++) {
        threads.emplace_back(benchmark_gpu_worker, i);
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto benchmark_end = std::chrono::high_resolution_clock::now();
    auto benchmark_duration = std::chrono::duration_cast<std::chrono::milliseconds>(benchmark_end - benchmark_start).count();
    
    // ======= Summary =======
    std::cout << "\n=== MULTI-GPU BENCHMARK SUMMARY ===" << std::endl;
    std::cout << "Number of GPUs: " << actual_gpus << std::endl;
    std::cout << "Page size: " << (PAGE_SIZE / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Pages per GPU: " << NUM_PAGES_PER_GPU << std::endl;
    std::cout << "Total pages: " << (NUM_PAGES_PER_GPU * actual_gpus) << std::endl;
    std::cout << "Total memory: " << (PAGE_SIZE * NUM_PAGES_PER_GPU * actual_gpus / (1024.0 * 1024 * 1024)) << " GB" << std::endl;
    std::cout << "Total benchmark time: " << benchmark_duration << " ms" << std::endl;
    std::cout << "Throughput: " << (NUM_PAGES_PER_GPU * actual_gpus * 1000.0 / benchmark_duration) << " pages/second" << std::endl;
    std::cout << "Memory bandwidth: " << (PAGE_SIZE * NUM_PAGES_PER_GPU * actual_gpus / (1024.0 * 1024 * 1024) / (benchmark_duration / 1000.0)) << " GB/second" << std::endl;
}

int main() {
    try {
        multi_gpu_benchmark();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
    
    return 0;
}