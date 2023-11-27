#include <stdio.h>
#include <CL/cl.h>
#include "oclboiler.h"

#define UPPERBOUND_PRIMARY_POSITIONS 32768 // 2^15
#define UPPERBOUND_SECONDARY_POSITIONS 4096 // 2^12

#define POSITION_STARTPOS 0x0000

void printboard (short pid, short sid){
    int i = 0, j = 0;
    for (int y = 0; y < 3; y++){
        for (int x = 0; x < 3; x++){
            if ((x+y)%2 == 0){
                printf("%d ", (pid >> i) & 0x7);
                i += 3;
            } else {
                printf("%d ", (sid >> j) & 0x7);
                j += 3;
            }
        }
        printf("\n");
    }
}

int log2int(int n){
    int res = 0;
    while (n >>= 1) res++;
    return res;
}

void create_buffers(cl_context ctx, size_t size, 
    cl_mem *parent_table, cl_mem *payload, cl_mem *next_payload, 
    cl_mem *predicate_table, cl_mem *idx_sum_table, cl_mem *temp_sum_table)
{
    cl_int err;
    *parent_table = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, NULL, &err);
    ocl_check(err, "creating parent table buffer");
    *payload = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, NULL, &err);
    ocl_check(err, "creating payload buffer");
    *next_payload = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, NULL, &err);
    ocl_check(err, "creating next payload buffer");
    *predicate_table = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, NULL, &err);
    ocl_check(err, "creating predicate table buffer");
    *idx_sum_table = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, NULL, &err);
    ocl_check(err, "creating idx sum table buffer");
    *temp_sum_table = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, NULL, &err);
    ocl_check(err, "creating temp sum table buffer");
}

void pass_mem_arg(cl_kernel k, int argc, cl_mem *mem){
    cl_int err = clSetKernelArg(k, argc, sizeof(cl_mem), mem);
    ocl_check(err, "setting mem kernel argument");
}

void pass_int_arg(cl_kernel k, int argc, cl_int *i){
    cl_int err = clSetKernelArg(k, argc, sizeof(cl_int), i);
    ocl_check(err, "setting int kernel argument");
}

void pass_short_arg(cl_kernel k, int argc, cl_short *s){
    cl_int err = clSetKernelArg(k, argc, sizeof(cl_short), s);
    ocl_check(err, "setting short kernel argument");
}

int main(int argc, char *argv[]){
    cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);

	/* Compile device-side of the program */
	cl_program prog = create_program("3x3kernels.ocl", ctx, d);

    cl_int err;
    cl_int nels;
    size_t global_size;
    size_t local_size;

    if (argc < 2){
        printf("Usage: %s <local_size>\n", argv[0]);
        exit(1);
    }

    local_size = atoi(argv[1]);

    // Create kernels
    cl_kernel k_setup = clCreateKernel(prog, "setup_tables", &err);
    ocl_check(err, "creating setup kernel");

    // Create the expand primary kernel
    cl_kernel expandp_kernel = clCreateKernel(prog, "expand_primary", &err);
    ocl_check(err, "create kernel expand_primary");
    cl_kernel expands_kernel = clCreateKernel(prog, "expand_secondary", &err);
    ocl_check(err, "create kernel expand_secondary");

    // Predicate neq0 kernel
    cl_kernel predicate_neq0_kernel = clCreateKernel(prog, "predicate_neq0", &err);
    ocl_check(err, "create kernel predicate_neq0");

    // Naive prefix sum kernel
    cl_kernel step_naive_prefix_sum_kernel = clCreateKernel(prog, "step_naive_prefix_sum", &err);
    ocl_check(err, "create kernel step_naive_prefix_sum");

    // Scatter kernel
    cl_kernel scatter_kernel = clCreateKernel(prog, "scatter", &err);
    ocl_check(err, "create kernel scatter");

    // PRIMARY GRID: 2^15
    nels = UPPERBOUND_PRIMARY_POSITIONS;
    global_size = round_mul_up(nels, local_size); // Which is still 2^15, but lets be explicit
    // Create buffers
    cl_mem parent_table, payload, next_payload, predicate_table, idx_sum_table, temp_sum_table;
    create_buffers(ctx, nels * sizeof(short), 
        &parent_table, &payload, &next_payload, &predicate_table, &idx_sum_table, &temp_sum_table);

    // Setup tables
    pass_mem_arg(k_setup, 0, &parent_table);
    pass_mem_arg(k_setup, 1, &payload);
    pass_mem_arg(k_setup, 2, &next_payload);
    pass_mem_arg(k_setup, 3, &predicate_table);
    pass_mem_arg(k_setup, 4, &idx_sum_table);
    pass_mem_arg(k_setup, 5, &temp_sum_table);
    pass_int_arg(k_setup, 6, &nels);

    err = clEnqueueNDRangeKernel(que, k_setup, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    ocl_check(err, "enqueueing setup kernel");

    cl_int log2_nels = log2int(nels);

    float p_expand_times[log2_nels];
    float p_predicate_times[log2_nels];
    float p_prefix_sum_times[log2_nels]; 
    // Initialize to 0 since we will accumulate
    for (int i = 0; i < log2_nels; i++){
        p_prefix_sum_times[i] = 0;
    }
    float p_scatter_times[log2_nels];
    
    // Expand primary cycle is as follows:
    // 1. Expand primary
    // 2. Predicate neq0
    // 3. Prefix sum
    // 4. Scatter next payload to payload
    // 5. repeat 1-4 until no more positions are discovered

    cl_int discovered_positions = 1; // Start with 1, since we know the start position
    cl_int total_positions = 1; // Same as above
    cl_int iteration = 0;

    while (discovered_positions){
        // global size can be reduced to a suitable size
        global_size = round_mul_up(discovered_positions, local_size);

        // Expand primary
        pass_mem_arg(expandp_kernel, 0, &parent_table);
        pass_mem_arg(expandp_kernel, 1, &payload);
        pass_mem_arg(expandp_kernel, 2, &next_payload);
        pass_int_arg(expandp_kernel, 3, &discovered_positions);

        cl_event expandp_event;
        err = clEnqueueNDRangeKernel(que, expandp_kernel, 1, NULL, &global_size, &local_size, 0, NULL, &expandp_event);
        ocl_check(err, "enqueueing expand primary kernel");
        clFinish(que);  
        p_expand_times[iteration] = runtime_ms(expandp_event);

        // Now we don't know which positions are discovered, so we will launch full global size
        global_size = round_mul_up(nels, local_size);

        // Predicate neq0
        pass_mem_arg(predicate_neq0_kernel, 0, &next_payload);
        pass_mem_arg(predicate_neq0_kernel, 1, &predicate_table);
        pass_int_arg(predicate_neq0_kernel, 2, &nels);

        cl_event predicate_neq0_event;
        err = clEnqueueNDRangeKernel(que, predicate_neq0_kernel, 1, NULL, &global_size, &local_size, 0, NULL, &predicate_neq0_event);
        ocl_check(err, "enqueueing predicate neq0 kernel");
        clFinish(que);  
        p_predicate_times[iteration] = runtime_ms(predicate_neq0_event);

        // Copy predicate table to idx_sum_table
        err = clEnqueueCopyBuffer(que, predicate_table, idx_sum_table, 0, 0, nels*sizeof(cl_short), 0, NULL, NULL);
        ocl_check(err, "copying predicate table to idx_sum_table");
        clFinish(que);

        // Use naive prefix sum
        int offset = 1;
        for (int d = log2_nels; d > 0; d--){
            
            // Prefix sum
            pass_mem_arg(step_naive_prefix_sum_kernel, 0, &idx_sum_table);
            pass_mem_arg(step_naive_prefix_sum_kernel, 1, &temp_sum_table);
            pass_int_arg(step_naive_prefix_sum_kernel, 2, &offset);
            pass_int_arg(step_naive_prefix_sum_kernel, 3, &nels);

            cl_event prefix_sum_event;
            err = clEnqueueNDRangeKernel(que, step_naive_prefix_sum_kernel, 1, NULL, &global_size, &local_size, 0, NULL, &prefix_sum_event);
            ocl_check(err, "enqueueing prefix sum kernel");
            clFinish(que);  
            p_prefix_sum_times[iteration] += runtime_ms(prefix_sum_event);

            // Swap idx_sum_table and temp_sum_table
            cl_mem temp = idx_sum_table;
            idx_sum_table = temp_sum_table;
            temp_sum_table = temp;

            // Wait for the previous event to finish
            clFinish(que);

            offset *= 2;            
        }

        // Scatter
        pass_mem_arg(scatter_kernel, 0, &next_payload);
        pass_mem_arg(scatter_kernel, 1, &predicate_table);
        pass_mem_arg(scatter_kernel, 2, &idx_sum_table);
        pass_mem_arg(scatter_kernel, 3, &payload); // By writing to payload, we can reuse the same buffer
        pass_int_arg(scatter_kernel, 4, &nels);

        cl_event scatter_event;
        err = clEnqueueNDRangeKernel(que, scatter_kernel, 1, NULL, &global_size, &local_size, 0, NULL, &scatter_event);
        ocl_check(err, "enqueueing scatter kernel");
        clFinish(que);  
        p_scatter_times[iteration] = runtime_ms(scatter_event);

        // Update discovered positions as the last element in the idx_sum_table
        err = clEnqueueReadBuffer(que, idx_sum_table, CL_TRUE, (nels-1)*sizeof(cl_short), sizeof(cl_short), &discovered_positions, 0, NULL, NULL);
        ocl_check(err, "reading discovered positions");
        clFinish(que);  

        total_positions += discovered_positions;
        
        iteration++;
        short zero = 0;
        // Clear the next payload buffer
        err = clEnqueueFillBuffer(que, next_payload,&zero , sizeof(cl_short), 0, nels*sizeof(cl_short), 0, NULL, NULL);
        ocl_check(err, "clearing next payload buffer");
        clFinish(que);

        printf("Primary Iteration %d\t: %d\t positions discovered (%d\t total)\n", iteration, discovered_positions, total_positions);
    }

    // Free the buffers
    clReleaseMemObject(parent_table);
    clReleaseMemObject(payload);
    clReleaseMemObject(next_payload);
    clReleaseMemObject(predicate_table);
    clReleaseMemObject(idx_sum_table);
    clReleaseMemObject(temp_sum_table);

    printf("Total primary positions discovered: %d\n", total_positions);
    printf("== Primary Time stats ==\n");
    float sum_expand = 0;
    float sum_predicate = 0;
    float sum_prefix_sum = 0;
    float sum_scatter = 0;
    for (int i = 0; i < iteration; i++){
        sum_expand += p_expand_times[i];
        sum_predicate += p_predicate_times[i];
        sum_prefix_sum += p_prefix_sum_times[i];
        sum_scatter += p_scatter_times[i];
    }
    float sum_total = sum_expand + sum_predicate + sum_prefix_sum + sum_scatter;
    printf("Expand: %f ms\n", sum_expand);
    printf("Predicate: %f ms\n", sum_predicate);
    printf("Prefix sum: %f ms\n", sum_prefix_sum);
    printf("Scatter: %f ms\n", sum_scatter);
    printf("Total: %f ms\n", sum_total);
    printf("========================\n");

    // SECONDARY GRID: 2^12
    
    nels = UPPERBOUND_SECONDARY_POSITIONS;
    global_size = round_mul_up(nels, local_size); // Which is still 2^12, but lets be explicit
    // Create buffers
    create_buffers(ctx, nels * sizeof(short), 
        &parent_table, &payload, &next_payload, &predicate_table, &idx_sum_table, &temp_sum_table);

    // Setup tables
    pass_mem_arg(k_setup, 0, &parent_table);
    pass_mem_arg(k_setup, 1, &payload);
    pass_mem_arg(k_setup, 2, &next_payload);
    pass_mem_arg(k_setup, 3, &predicate_table);
    pass_mem_arg(k_setup, 4, &idx_sum_table);
    pass_mem_arg(k_setup, 5, &temp_sum_table);
    pass_int_arg(k_setup, 6, &nels);

    err = clEnqueueNDRangeKernel(que, k_setup, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    ocl_check(err, "enqueueing setup kernel");

    log2_nels = log2int(nels);

    float s_expand_times[log2_nels];
    float s_predicate_times[log2_nels];
    float s_prefix_sum_times[log2_nels]; 

    // Initialize to 0 since we will accumulate
    for (int i = 0; i < log2_nels; i++){
        s_prefix_sum_times[i] = 0;
    }
    float s_scatter_times[log2_nels];
    
    // Expand secondary cycle is as follows:
    // 1. Expand secondary
    // 2. Predicate neq0
    // 3. Prefix sum
    // 4. Scatter next payload to payload
    // 5. repeat 1-4 until no more positions are discovered

    discovered_positions = 1; // Start with 1, since we know the start position
    total_positions = 1; // Same as above
    iteration = 0;
    while (discovered_positions){
        // global size can be reduced to a suitable size
        global_size = round_mul_up(discovered_positions, local_size);

        // Expand primary
        pass_mem_arg(expands_kernel, 0, &parent_table);
        pass_mem_arg(expands_kernel, 1, &payload);
        pass_mem_arg(expands_kernel, 2, &next_payload);
        pass_int_arg(expands_kernel, 3, &discovered_positions);

        cl_event expands_event;
        err = clEnqueueNDRangeKernel(que, expands_kernel, 1, NULL, &global_size, &local_size, 0, NULL, &expands_event);
        ocl_check(err, "enqueueing expand primary kernel");
        clFinish(que);  
        s_expand_times[iteration] = runtime_ms(expands_event);

        // Now we don't know which positions are discovered, so we will launch full global size
        global_size = round_mul_up(nels, local_size);

        // Predicate neq0
        pass_mem_arg(predicate_neq0_kernel, 0, &next_payload);
        pass_mem_arg(predicate_neq0_kernel, 1, &predicate_table);
        pass_int_arg(predicate_neq0_kernel, 2, &nels);

        cl_event predicate_neq0_event;
        err = clEnqueueNDRangeKernel(que, predicate_neq0_kernel, 1, NULL, &global_size, &local_size, 0, NULL, &predicate_neq0_event);
        ocl_check(err, "enqueueing predicate neq0 kernel");
        clFinish(que);  
        s_predicate_times[iteration] = runtime_ms(predicate_neq0_event);

        // Copy predicate table to idx_sum_table
        err = clEnqueueCopyBuffer(que, predicate_table, idx_sum_table, 0, 0, nels*sizeof(cl_short), 0, NULL, NULL);
        ocl_check(err, "copying predicate table to idx_sum_table");
        clFinish(que);

        // Use naive prefix sum
        int offset = 1;
        for (int d = log2_nels; d > 0; d--){
            
            // Prefix sum
            pass_mem_arg(step_naive_prefix_sum_kernel, 0, &idx_sum_table);
            pass_mem_arg(step_naive_prefix_sum_kernel, 1, &temp_sum_table);
            pass_int_arg(step_naive_prefix_sum_kernel, 2, &offset);
            pass_int_arg(step_naive_prefix_sum_kernel, 3, &nels);

            cl_event prefix_sum_event;
            err = clEnqueueNDRangeKernel(que, step_naive_prefix_sum_kernel, 1, NULL, &global_size, &local_size, 0, NULL, &prefix_sum_event);
            ocl_check(err, "enqueueing prefix sum kernel");
            clFinish(que);  
            s_prefix_sum_times[iteration] += runtime_ms(prefix_sum_event);

            // Swap idx_sum_table and temp_sum_table
            cl_mem temp = idx_sum_table;
            idx_sum_table = temp_sum_table;
            temp_sum_table = temp;

            // Wait for the previous event to finish
            clFinish(que);

            offset *= 2;            
        }

        // Scatter
        pass_mem_arg(scatter_kernel, 0, &next_payload);
        pass_mem_arg(scatter_kernel, 1, &predicate_table);
        pass_mem_arg(scatter_kernel, 2, &idx_sum_table);
        pass_mem_arg(scatter_kernel, 3, &payload); // By writing to payload, we can reuse the same buffer
        pass_int_arg(scatter_kernel, 4, &nels);

        cl_event scatter_event;
        err = clEnqueueNDRangeKernel(que, scatter_kernel, 1, NULL, &global_size, &local_size, 0, NULL, &scatter_event);
        ocl_check(err, "enqueueing scatter kernel");
        clFinish(que);  
        s_scatter_times[iteration] = runtime_ms(scatter_event);

        // Update discovered positions as the last element in the idx_sum_table
        err = clEnqueueReadBuffer(que, idx_sum_table, CL_TRUE, (nels-1)*sizeof(cl_short), sizeof(cl_short), &discovered_positions, 0, NULL, NULL);
        ocl_check(err, "reading discovered positions");
        clFinish(que);  

        total_positions += discovered_positions;
        
        iteration++;
        short zero = 0;
        // Clear the next payload buffer
        err = clEnqueueFillBuffer(que, next_payload,&zero , sizeof(cl_short), 0, nels*sizeof(cl_short), 0, NULL, NULL);
        ocl_check(err, "clearing next payload buffer");
        clFinish(que);

        printf("Secondary Iteration %d\t: %d\t positions discovered (%d\t total)\n", iteration, discovered_positions, total_positions);
    }

    // Free the buffers
    clReleaseMemObject(parent_table);
    clReleaseMemObject(payload);
    clReleaseMemObject(next_payload);
    clReleaseMemObject(predicate_table);
    clReleaseMemObject(idx_sum_table);
    clReleaseMemObject(temp_sum_table);

    printf("Total secondary positions discovered: %d\n", total_positions);
    printf("== Secondary Time stats ==\n");
    sum_expand = 0;
    sum_predicate = 0;
    sum_prefix_sum = 0;
    sum_scatter = 0;
    sum_total = 0;
    for (int i = 0; i < iteration; i++){
        sum_expand += s_expand_times[i];
        sum_predicate += s_predicate_times[i];
        sum_prefix_sum += s_prefix_sum_times[i];
        sum_scatter += s_scatter_times[i];
    }
    sum_total = sum_expand + sum_predicate + sum_prefix_sum + sum_scatter;
    printf("Expand: %f ms\n", sum_expand);
    printf("Predicate: %f ms\n", sum_predicate);
    printf("Prefix sum: %f ms\n", sum_prefix_sum);
    printf("Scatter: %f ms\n", sum_scatter);
    printf("Total: %f ms\n", sum_total);
    printf("==========================\n");
}