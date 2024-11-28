#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>

/*
Compile w ./a.out 1000 1000 asdf

*/

// Struct to get time
double gettime(void)
{
    struct timeval tval;
    gettimeofday(&tval, NULL);
    return ((double)tval.tv_sec + (double)tval.tv_usec / 10000000.0);
}

// Function for performing warp shuffle
__device__ double warp_shuffle(double var)
{
    // Gets every thread in warp
    unsigned mask = 0xffffffff;
    for (int diff = 32 / 2; diff > 0; diff = diff / 2)
    {
        var += __shfl_down_sync(mask, var, diff);
    }
    // Only lane 0 gets true value
    return var;
}

// Main conway function
__global__ void Conway(int *map1, int *map2, float *nChanges, int nRows, int thread_count, int sm_count, int remainder_for_last_block)
{

    // Value for just this row's changes, will be warp shuffled together later
    int local_changes = 0;

    // Get identification
    int blockNum = blockIdx.x;
    int threadNum = threadIdx.x;
    // int threadNumGlobal = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("Hello from thread %d in block %d!\n", threadNum, blockNum);

    // Analogous to global thread id
    int my_row = blockNum * thread_count + threadNum;
    // printf("my row: %d\n", my_row);

    // This makes sure not to get the ghost row at the top and bottom
    if (my_row < nRows - 1 && my_row > 0)
    {

        // Each thread iterates over the duration of a row
        for (int j = 1; j < nRows - 1; j++)
        {
            // center = map1[my_row * nRows + j];
            // All neighbors
            int neighborCount = map1[my_row * nRows + j - 1] +
                                map1[my_row * nRows + j + 1] +
                                map1[(my_row - 1) * nRows + j - 1] +
                                map1[(my_row - 1) * nRows + j] +
                                map1[(my_row - 1) * nRows + j + 1] +
                                map1[(my_row + 1) * nRows + j - 1] +
                                map1[(my_row + 1) * nRows + j] +
                                map1[(my_row + 1) * nRows + j + 1];

            // GOL rules
            if (neighborCount == 3)
            {
                map2[my_row * nRows + j] = 1;
            }
            else if (neighborCount == 2)
            {
                map2[my_row * nRows + j] = map1[my_row * nRows + j];
            }
            else
            {
                map2[my_row * nRows + j] = 0;
            }


            // Keeps track of local changes
            if (map2[my_row * nRows + j] != map1[my_row * nRows + j])
            {
                local_changes++;
            }
        }
    }

    // Warp shuffles the local changes down to lane 0 and atomic adds
    local_changes = warp_shuffle(local_changes);
    int my_lane = threadIdx.x % 32;
    if (my_lane == 0)
    {
        atomicAdd(nChanges, local_changes);
    }

} /*Runs conway*/

int main(int argc, char *argv[])
{

    // Checks cmd args
    if (argc != 4)
    {
        printf("Usage: ./<executable(conway)> <board-size> <num-generations> <output_directory>\n");
        return 0;
    }

    int nRows = atoi(argv[1]) + 2;
    int nCols = nRows;
    int numGenerations = atoi(argv[2]);
    char *output_directory = argv[3];

    // Declares arrays
    int *map1;
    int *map2;
    cudaMallocManaged(&map1, nRows * nCols * sizeof(int));
    cudaMallocManaged(&map2, nRows * nCols * sizeof(int));

    // Fills initial state
    srand48(12345);
    for (int i = 0; i < nRows; i++)
    {

        for (int j = 0; j < nCols; j++)
        {

            if (i == 0 || i == nCols - 1 || j == 0 || j == nCols - 1)
            {
                map1[i * nRows + j] = 0;
                map2[i * nRows + j] = 0;
            }
            else
            {
                map1[i * nRows + j] = drand48() > 0.5 ? 1 : 0;
            }
        }
    }

    float *nChanges;
    cudaMallocManaged(&nChanges, sizeof(float));
    *nChanges = 0;

    // Assume each thread block has 32 threads
    // This allows us to use warp-wide operations
    // Its gonna make the last case weird but so be it
    int thread_count = 32;
    int sm_count = (nRows) / 32;
    int remainder_for_last_block = (nRows) % 32;
    if (remainder_for_last_block > 0)
    {
        sm_count++;
    }
    // printf("%d\n", sm_count);
    // printf("%d\n", remainder_for_last_block);

    int gen = 0;

    double start = gettime();
    for (gen = 0; gen < numGenerations; gen++)
    {
        if (gen % 2 == 0)
        {
            Conway<<<sm_count, thread_count>>>(map1, map2, nChanges, nRows, thread_count, sm_count, remainder_for_last_block);
        }
        else
        {
            Conway<<<sm_count, thread_count>>>(map2, map1, nChanges, nRows, thread_count, sm_count, remainder_for_last_block);
        }
        cudaDeviceSynchronize();

        if (*nChanges == 0)
        {
            printf("Exiting early due to no changes at generation %d\n", gen);
            break;
        }
    }
    double finish = gettime();
    printf("Finished Conway\n");

    printf("Test details:\n %d by %d board\n", nRows, nCols);
    printf("%d generations\n", numGenerations);
    printf("Time taken = %lf seconds\n", (finish - start));
    printf("Terminated at generation %d\n", gen);

    // printf("%d\n", *nChanges);

    // Writes results to the file
    FILE *file = fopen(output_directory, "w");
    if (file == NULL)
    {
        perror("Error opening file\n");
        return 1;
    }
    for (int i = 1; i < nRows - 1; i++)
    {
        for (int j = 1; j < nCols - 1; j++)
        {
            // printf("%d\n", i);
            if (gen % 2 == 0)
            {
                fprintf(file, "%d ", map1[i * nRows + j]);
            }
            else
            {
                fprintf(file, "%d ", map2[i * nRows + j]);
            }
        }

        fprintf(file, "\n"); // New line after each row
    }
    printf("Done\n");
    fclose(file);

    cudaFree(map1);
    cudaFree(map2);

    printf("Finished\n");

    return 0;
}
