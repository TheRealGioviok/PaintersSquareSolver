#define CL_TARGET_OPENCL_VERSION 120
#include "oclboiler.h"
#include <stdio.h>
#include <stdlib.h>

void error(const char *err){
	fprintf(stderr, "%s\n", err);
	exit(1);
}

void pass_mem_arg(cl_kernel k, int argc, cl_mem *mem){
    cl_int err = clSetKernelArg(k, argc, sizeof(cl_mem), mem);
    ocl_check(err, "setting mem kernel argument");
}

void pass_int_arg(cl_kernel k, int argc, cl_int *i){
    cl_int err = clSetKernelArg(k, argc, sizeof(cl_int), i);
    ocl_check(err, "setting int kernel argument");
}

void pass_lmem(cl_kernel k, int argc, size_t size){
    cl_int err = clSetKernelArg(k, argc, size, NULL);
    ocl_check(err, "setting lmem kernel argument");
}

void create_buffers(cl_context ctx, size_t size, size_t tails_size,
    cl_mem *parent_table, cl_mem *payload, cl_mem *next_payload, 
    cl_mem *predicate_table, cl_mem *idx_sum_table, cl_mem *tails_table)
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
    *tails_table = clCreateBuffer(ctx, CL_MEM_READ_WRITE, tails_size, NULL, &err);
    ocl_check(err, "creating tails table buffer");
}

cl_event expand(cl_kernel kernel, cl_command_queue que, cl_mem parent_table, cl_mem payload, cl_mem next_payload, size_t lws, cl_int items){
    // global size can be reduced to a suitable size
    size_t gws = round_mul_up(items, lws);

    // printf("Call to expand kernel with args: %p, %p, %p, %d\n", parent_table, payload, next_payload, items);

    // Expand primary
    pass_mem_arg(kernel, 0, &parent_table);
    pass_mem_arg(kernel, 1, &payload);
    pass_mem_arg(kernel, 2, &next_payload);
    pass_int_arg(kernel, 3, &items);

    cl_event expandp_event;
    ocl_check(clEnqueueNDRangeKernel(que, kernel, 1, NULL, &gws, &lws, 0, NULL, &expandp_event), "enqueueing expand kernel");
    
    return expandp_event;
}

cl_event predicate(cl_kernel predicate, cl_command_queue que, cl_mem predicate_table, cl_mem next_payload, size_t lws, cl_int items){
    // global size can be reduced to a suitable size
    size_t gws = round_mul_up(items, lws);

    // Predicate
    pass_mem_arg(predicate, 0, &next_payload);
    pass_mem_arg(predicate, 1, &predicate_table);
    pass_int_arg(predicate, 2, &items);

    cl_event predicate_event;
    ocl_check(clEnqueueNDRangeKernel(que, predicate, 1, NULL, &gws, &lws, 0, NULL, &predicate_event), "enqueueing predicate kernel");

    return predicate_event;
}

cl_event scan(cl_command_queue que, cl_kernel scan_kernel, cl_mem predicates, cl_mem tails, cl_mem out, int nels,cl_int preferred_rounding_scan, int lws_arg, int nwg) {
	cl_int nquarts = nels/4;
	size_t lws[] = { lws_arg };
	size_t gws[] = { lws[0]*nwg };

	cl_int err;
	cl_event ret;

	pass_mem_arg(scan_kernel, 0, &out);
    pass_mem_arg(scan_kernel, 1, &tails);
    pass_mem_arg(scan_kernel, 2, &predicates);
    pass_int_arg(scan_kernel, 3, &nquarts);
    pass_int_arg(scan_kernel, 4, &preferred_rounding_scan);
    pass_lmem(scan_kernel, 5, lws[0]*sizeof(cl_int));
    err = clEnqueueNDRangeKernel(que, scan_kernel, 1,
		NULL, gws, lws,
		0, NULL,  &ret);
	ocl_check(err, "enqueue scan");
	return ret;
}

cl_event fixup(cl_command_queue que, cl_kernel fixup_kernel, cl_mem d_out, cl_mem d_tails, int nels, cl_int preferred_rounding_scan, int lws_arg, int nwg) {
	cl_int nquarts = nels/4;
	size_t lws[] = { lws_arg };
	size_t gws[] = { lws[0]*nwg };

	cl_int err;
	cl_event ret;

    pass_mem_arg(fixup_kernel, 0, &d_out);
    pass_mem_arg(fixup_kernel, 1, &d_tails);
    pass_int_arg(fixup_kernel, 2, &nquarts);
    pass_int_arg(fixup_kernel, 3, &preferred_rounding_scan);

	err = clEnqueueNDRangeKernel(que, fixup_kernel, 1,
		NULL, gws, lws,
		0, NULL,  &ret);
	ocl_check(err, "enqueue fixup");

	return ret;
}

cl_event scatter(cl_command_queue que, cl_kernel scatter_kernel, cl_mem predicate_table, cl_mem sum_table, cl_mem next_payload_table, cl_mem payload_table, size_t lws, cl_int items){
    // global size can be reduced to a suitable size
    size_t gws = round_mul_up(items, lws);
    // Scatter
    pass_mem_arg(scatter_kernel, 0, &predicate_table);
    pass_mem_arg(scatter_kernel, 1, &sum_table);
    pass_mem_arg(scatter_kernel, 2, &next_payload_table);
    pass_mem_arg(scatter_kernel, 3, &payload_table);
    pass_int_arg(scatter_kernel, 4, &items);
    cl_event scatter_event;
    ocl_check(clEnqueueNDRangeKernel(que, scatter_kernel, 1, NULL, &gws, &lws, 0, NULL, &scatter_event), "enqueueing scatter kernel");

    return scatter_event;
}

void await(cl_event e){
    cl_int err = clWaitForEvents(1, &e);
    ocl_check(err, "waiting for event");
}


void printboard (int pid, int sid){
    int i = 0, j = 0;
    for (int y = 0; y < 4; y++){
        for (int x = 0; x < 4; x++){
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

// #define USE_SUBS_WORD_VECTORIZATION // Use bitgroups arithmetic to speedup (yields same results because the board is small)
#define DEVICE 0 // Means GPU 1

#define UPPERBOUND_PRIMARY_POSITIONS (1<<24)

#define POSITION_TODISCOVER (1<<24)
#define POSITION_STARTPOS 0x0

// Arguments:
// 1: local work size
// 2: number of groups for the scan kernel
int main(int argc, char *argv[]){
    if (argc < 3) error("Usage: 4x4 <local work size> <number of groups>");

    int lws = atoi(argv[1]);
    int ngroups, ntails;
    ngroups = ntails = atoi(argv[2]);

    size_t buffer_size = UPPERBOUND_PRIMARY_POSITIONS * sizeof(cl_int);
    size_t tails_buffer_size = ntails * sizeof(cl_int);

    cl_platform_id p = select_platform();
    cl_device_id d = select_device(p); // Passed via ambient variable
    cl_context ctx = create_context(p, d);
    cl_command_queue que = create_queue(ctx, d);
    cl_program prog = create_program("4x4.ocl", ctx, d);

    cl_int err;

    // The index sum kernel. We use the much more efficient grouped scan kernel
    cl_kernel scan_kernel = clCreateKernel(prog, "scan", &err);
    ocl_check(err, "creating scan kernel");
    cl_kernel fixup_kernel = clCreateKernel(prog, "scan_fixup", &err);
    ocl_check(err, "creating fixup kernel");

    // The expand kernel. Since this is a 4x4 board, that means we that the secondary grid is just a flip of the primary grid
    // Therefore, we just calculate the primary grid indices and then flip them to get the secondary grid indices

#ifdef USE_SUBS_WORD_VECTORIZATION
    cl_kernel expand_kernel = clCreateKernel(prog, "expand_subvec", &err);
    ocl_check(err, "creating expand kernel");
#else
    cl_kernel expand_kernel = clCreateKernel(prog, "expand", &err);
    ocl_check(err, "creating expand kernel");
#endif

    // The predicate kernel. It is already extremely fast and takes little to no time in relation to the other kernels
    cl_kernel predicate_kernel = clCreateKernel(prog, "predicate_neq0", &err);
    ocl_check(err, "creating predicate kernel");

    // The scatter kernel.
    cl_kernel scatter_kernel = clCreateKernel(prog, "scatter", &err);
    ocl_check(err, "creating scatter kernel");

    // We will use the notation DTZ(x) to denote the distance to zero of x, that is, the optimal number of moves to reach a zero (start) position
    // Create all the required buffers
    // Parent_table: For each position x, where DTZ(x) = k, it contains a position y such that DTZ(y) = k-1
    // Payload: After iteration i, it contains the number of positions with DTZ(x) = i
    // Next_payload: After iteration i, it contains the number of positions with DTZ(x) = i+1
    // Predicate_table: Used to filter out zeros from the next payload
    // Idx_sum_table: Inclusive sum of predicates. Used to do efficient compaction of the next payload
    // tails_table: Temp buffer, useful for the scan kernel

    cl_mem parent_table, payload, next_payload, predicate_table, idx_sum_table, tails_table;
    create_buffers(ctx, buffer_size, tails_buffer_size, &parent_table, &payload, &next_payload, &predicate_table, &idx_sum_table, &tails_table);

    // Fill parent_table with POSITION_TODISCOVER (except for the first position, which is the start position)
    int passable_position_to_discover = POSITION_TODISCOVER; // Since it needs to be passed by reference
    err = clEnqueueFillBuffer(que, parent_table, &passable_position_to_discover, sizeof(cl_int), sizeof(cl_int), buffer_size - sizeof(cl_int), 0, NULL, NULL);
    ocl_check(err, "filling parent table buffer");

    // Assert that the first position is the start position
    unsigned int zero = 0;
    err = clEnqueueWriteBuffer(que, parent_table, CL_TRUE, 0, sizeof(cl_int), &zero, 0, NULL, NULL);
    ocl_check(err, "writing start position to parent table buffer");


    // Fill every other buffer with zeros
    err = clEnqueueFillBuffer(que, payload, &zero, sizeof(cl_int), 0, buffer_size, 0, NULL, NULL);
    ocl_check(err, "filling payload buffer");
    err = clEnqueueFillBuffer(que, next_payload, &zero, sizeof(cl_int), 0, buffer_size, 0, NULL, NULL);
    ocl_check(err, "filling next payload buffer");
    err = clEnqueueFillBuffer(que, predicate_table, &zero, sizeof(cl_int), 0, buffer_size, 0, NULL, NULL);
    ocl_check(err, "filling predicate table buffer");
    err = clEnqueueFillBuffer(que, idx_sum_table, &zero, sizeof(cl_int), 0, buffer_size, 0, NULL, NULL);
    ocl_check(err, "filling idx sum table buffer");
    err = clEnqueueFillBuffer(que, tails_table, &zero, sizeof(cl_int), 0, tails_buffer_size, 0, NULL, NULL);
    ocl_check(err, "filling temp sum table buffer");

    float expand_time = .0f;
    float predicate_time = .0f;
    float main_scan_time = .0f;
    float tails_scan_time = .0f;
    float fixup_time = .0f;
    float scatter_time = .0f;

    // Expand cycle is as follows:
    // 1. Expand (primary)
    // 2. Predicate neq0
    // 3. Prefix sum
    // 4. Scatter next payload to payload
    // 5. repeat 1-4 until no more positions are discovered

    cl_int discovered_positions = 1; // Start position is already discovered
    cl_int total_positions = 1; // Start position is already discovered
    cl_int iteration = 0;

    size_t preferred_rounding_scan;

    err = clGetKernelWorkGroupInfo(scan_kernel, d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &preferred_rounding_scan, NULL);
    ocl_check(err, "getting preferred rounding for scan kernel");

    // Loop
    while (discovered_positions){
        // Expand
        cl_event expand_event = expand(expand_kernel, que, parent_table, payload, next_payload, lws, discovered_positions);

        // Await expand
        await(expand_event);
        expand_time += runtime_ms(expand_event);

        // Predicate
        cl_event predicate_event = predicate(predicate_kernel, que, predicate_table, next_payload, lws, UPPERBOUND_PRIMARY_POSITIONS);
        await(predicate_event);
        predicate_time += runtime_ms(predicate_event);

        // Scan
        cl_event scan_partial_event = scan(que, scan_kernel, predicate_table, tails_table, idx_sum_table, UPPERBOUND_PRIMARY_POSITIONS, preferred_rounding_scan, lws, ngroups);
        // Await scan
        await(scan_partial_event);
        main_scan_time += runtime_ms(scan_partial_event);

        // Fixup and scan tails
        cl_event fixup_event;
        cl_event scan_tails_event;

        // If there are more than one group, we need to fixup and scan the tails
        if (ngroups > 1){
            scan_tails_event = scan(que, scan_kernel, tails_table, NULL, tails_table, ntails, preferred_rounding_scan, lws, 1);
            // Await scan tails
            await(scan_tails_event);
            tails_scan_time += runtime_ms(scan_tails_event);
            fixup_event = fixup(que, fixup_kernel, idx_sum_table, tails_table, UPPERBOUND_PRIMARY_POSITIONS, preferred_rounding_scan, lws, ngroups);
            // Await fixup
            await(fixup_event);
            fixup_time += runtime_ms(fixup_event);
        }

        // Read the total number of discovered positions as the last element of the idx_sum_table
        err = clEnqueueReadBuffer(que, idx_sum_table, CL_TRUE, buffer_size - sizeof(cl_int), sizeof(cl_int), &discovered_positions, 0, 0, NULL);
        ocl_check(err, "reading total number of discovered positions");

        // Scatter next payload to payload
        cl_event scatter_event = scatter(que, scatter_kernel, predicate_table, idx_sum_table, next_payload, payload, lws, UPPERBOUND_PRIMARY_POSITIONS);

        // Await scatter
        await(scatter_event);
        scatter_time += runtime_ms(scatter_event);

        // Clear next payload
        err = clEnqueueFillBuffer(que, next_payload, &zero, sizeof(cl_int), 0, buffer_size, 0, NULL, NULL);
        ocl_check(err, "filling next payload buffer");

        // Await clear next payload
        clFinish(que);

        // Update total positions
        total_positions += discovered_positions;

        // Print progress
        printf("Iteration %d: %d positions discovered\n", iteration, discovered_positions);

        // Increment iteration
        iteration++;
    }

    // Print results
    printf("Total positions: %d\n", total_positions);

    // Print timings
    printf("===== Timings =====\n");
    printf("Expand: %gms\n", expand_time);
    printf("Predicate: %gms\n", predicate_time);
    printf("Main scan: %gms\n", main_scan_time);
    printf("Tails scan: %gms\n", tails_scan_time);
    printf("Fixup: %gms\n", fixup_time);
    printf("Scatter: %gms\n", scatter_time);

    float total_time = expand_time + predicate_time + main_scan_time + tails_scan_time + fixup_time + scatter_time;
    printf("Total time: %gms\n", total_time);
    printf("===== End Timings =====\n");

    // Cleanup
    clReleaseKernel(expand_kernel);
    clReleaseKernel(predicate_kernel);
    clReleaseKernel(scan_kernel);
    clReleaseKernel(fixup_kernel);
    clReleaseKernel(scatter_kernel);
    clReleaseMemObject(parent_table);
    clReleaseMemObject(payload);
    clReleaseMemObject(next_payload);
    clReleaseMemObject(predicate_table);
    clReleaseMemObject(idx_sum_table);
    clReleaseMemObject(tails_table);
    clReleaseProgram(prog);
    clReleaseCommandQueue(que);
    clReleaseContext(ctx);

    return 0;
}