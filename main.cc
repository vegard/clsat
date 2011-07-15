#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <CL/cl.h>

struct thread_state {
	uint32_t rnd;
	uint32_t nr_sat_clauses;
};

struct gpu_clause {
	uint16_t literals[4];
};

void CL_CALLBACK pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
	fprintf(stderr, "notify: %s\n", errinfo);
}

typedef int literal;
typedef unsigned int variable;
typedef std::map<variable, variable> variable_map;
typedef std::vector<literal> clause;
typedef std::vector<clause> clause_vector;

void read_cnf(const char *filename, variable_map &variables, variable_map &reverse_variables, clause_vector &clauses)
{
	std::ifstream file;

	file.open(filename);
	if (!file) {
		fprintf(stderr, "ifstream::open() failed\n");
		exit(EXIT_FAILURE);
	}

	while (!file.eof()) {
		std::string line;
		getline(file, line);

		if (line.size() == 0)
			continue;

		/* Skip problem line -- we don't use it anyway */
		if (line[0] == 'p')
			continue;

		/* Skip comments */
		if (line[0] == 'c')
			continue;

		/* XOR clauses */
		if (line[0] == 'x') {
			fprintf(stderr, "Cannot read XOR clauses\n");
			exit(EXIT_FAILURE);
		}

		clause c;

		std::stringstream s(line);
		while (!s.eof()) {
			literal l;
			s >> l;

			if (l == 0)
				break;

			variable v = abs(l);

			/* We remap variables to the range [1, n], where n
			 * is the total number of variables */
			variable v2;
			variable_map::iterator it = variables.find(v);
			if (it == variables.end()) {
				v2 = 1 + variables.size();
				variables[v] = v2;
				reverse_variables[v2] = v;
			} else {
				v2 = it->second;
			}

			c.push_back(v2 * (l < 0 ? -1 : 1));
		}

		clauses.push_back(c);
	}

	file.close();
}

int main(int argc, char *argv[])
{
	/* XXX: Use mersenne twister */
	srand(time(NULL));

	variable_map variables;
	variable_map reverse_variables;
	clause_vector clauses;

	for (int i = 1; i < argc; ++i)
		read_cnf(argv[i], variables, reverse_variables, clauses);

	cl_uint num_platforms;
	if (clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS) {
		fprintf(stderr, "clGetPlatformIDs() returned failure\n");
		exit(EXIT_FAILURE);
	}

	if (num_platforms == 0) {
		fprintf(stderr, "No OpenCL platforms available\n");
		exit(EXIT_FAILURE);
	}

	cl_platform_id platforms[num_platforms];
	if (clGetPlatformIDs(num_platforms, platforms, NULL) != CL_SUCCESS) {
		fprintf(stderr, "clGetPlatformIDs() returned failure\n");
		exit(EXIT_FAILURE);
	}

	unsigned int platform;
	for (unsigned int i = 0; i < num_platforms; ++i) {
		size_t size;
		if (clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &size)
			!= CL_SUCCESS)
		{
			fprintf(stderr, "clGetPlatformInfo() returned failure\n");
			continue;
		}

		char name[size];
		if (clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, size, name, 0)
			!= CL_SUCCESS)
		{
			fprintf(stderr, "clGetPlatformInfo() returned failure\n");
			continue;
		}

		printf("platform %u: %s\n", i, name);

		platform = i;
	}

	cl_uint num_devices;
	if (clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices)
		!= CL_SUCCESS)
	{
		fprintf(stderr, "clGetDeviceIDs() returned failure\n");
		exit(EXIT_FAILURE);
	}

	cl_device_id devices[num_devices];
	if (clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL)
		!= CL_SUCCESS)
	{
		fprintf(stderr, "clGetDeviceIDs() returned failure\n");
		exit(EXIT_FAILURE);
	}

	unsigned int device;
	for (unsigned int i = 0; i < num_devices; ++i) {
		size_t size;
		if (clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, NULL, &size)
			!= CL_SUCCESS)
		{
			fprintf(stderr, "clDeviceInfo() returned failure\n");
			continue;
		}

		char name[size];
		if (clGetDeviceInfo(devices[i], CL_DEVICE_NAME, size, name, NULL)
			!= CL_SUCCESS)
		{
			fprintf(stderr, "clDeviceInfo() returned failure\n");
			continue;
		}

		fprintf(stderr, "device %u: %s\n", i, name);

		device = i;
	}

	cl_int err;
	cl_context context = clCreateContext(NULL, 1, &devices[device], &pfn_notify, NULL, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "clCreateContext() failed\n");
		exit(EXIT_FAILURE);
	}

	cl_command_queue queue = clCreateCommandQueue(context, devices[device], 0, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "clCreateCommandQueue() failed\n");
		exit(EXIT_FAILURE);
	}

	int fd = open("kernel.cl", O_RDONLY);
	if (fd == -1) {
		fprintf(stderr, "open() failed\n");
		exit(EXIT_FAILURE);
	}

	struct stat st;
	if (fstat(fd, &st) == -1) {
		fprintf(stderr, "stat() failed\n");
		exit(EXIT_FAILURE);
	}

	void *ptr = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	if (ptr == MAP_FAILED) {
		fprintf(stderr, "mmap() failed\n");
		exit(EXIT_FAILURE);
	}

	const char *string = (char *) ptr;
	const size_t length = st.st_size;

	cl_program program = clCreateProgramWithSource(context, 1, &string, &length, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "clCreateProgramWithSource() failed\n");
		exit(EXIT_FAILURE);
	}

	if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
		fprintf(stderr, "clBuildProgram() failed\n");

		size_t size;
		if (clGetProgramBuildInfo(program, devices[device],
			CL_PROGRAM_BUILD_LOG, 0, NULL, &size) != CL_SUCCESS)
		{
			fprintf(stderr, "clGetProgramBuildInfo() failed\n");
			exit(EXIT_FAILURE);
		}

		char log[size];
		if (clGetProgramBuildInfo(program, devices[device],
			CL_PROGRAM_BUILD_LOG, size, log, NULL) != CL_SUCCESS)
		{
			fprintf(stderr, "clGetProgramBuildInfo() failed\n");
			exit(EXIT_FAILURE);
		}

		fprintf(stderr, "%s\n", log);
		exit(EXIT_FAILURE);
	}

	cl_kernel kernel = clCreateKernel(program, "search", &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "clCreateKernel() failed\n");
		exit(EXIT_FAILURE);
	}

	unsigned int nr_threads = 256;

	struct thread_state host_threads[nr_threads];

	for (unsigned int i = 0; i < nr_threads; ++i) {
		host_threads[i].rnd = i;
		host_threads[i].nr_sat_clauses = 0;
	}

	cl_mem threads = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(host_threads), host_threads, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "clCreateBuffer() failed\n");
		exit(EXIT_FAILURE);
	}

	unsigned int nr_variables = 1 + variables.size();

	uint8_t host_variables[nr_variables][nr_threads];

	/* Variable 0 is always false. */
	for (unsigned int j = 0; j < nr_threads; ++j)
		host_variables[0][j] = 0;

	for (unsigned int i = 1; i < nr_variables; ++i) {
		for (unsigned int j = 0; j < nr_threads; ++j)
			host_variables[i][j] = rand() % 2;
	}

	cl_mem values = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(host_variables), host_variables, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "clCreateBuffer() failed\n");
		exit(EXIT_FAILURE);
	}

	unsigned int nr_clauses = clauses.size() + nr_threads - (clauses.size() % nr_threads);

	gpu_clause host_clauses[nr_clauses];
	for (unsigned int i = 0; i < clauses.size(); ++i) {
		clause &c = clauses[i];

		assert(c.size() >= 1 && c.size() <= 4);
		for (unsigned int j = 0; j < c.size(); ++j) {
			literal l = c[j];
			unsigned int var = abs(l);
			unsigned int sign = (l < 0);

			host_clauses[i].literals[j] = (var << 1) | sign;
		}

		/* Fill the rest of the clause with unsatisfied literals */
		for (unsigned int j = c.size(); j < 4; ++j)
			host_clauses[i].literals[j] = 0;
	}

	/* Create trivially true clauses to fill the remaining clause slots. */
	for (unsigned int i = clauses.size(); i < nr_clauses; ++i) {
		host_clauses[i].literals[0] = 1;
		host_clauses[i].literals[1] = 1;
		host_clauses[i].literals[2] = 1;
		host_clauses[i].literals[3] = 1;
	}

	cl_mem device_clauses = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(host_clauses), host_clauses, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "clCreateBuffer() failed\n");
		exit(EXIT_FAILURE);
	}

	unsigned int clause_i = 0;
	unsigned int n = nr_threads * 256;

	if (clSetKernelArg(kernel, 0, sizeof(threads), &threads) != CL_SUCCESS) {
		fprintf(stderr, "clSetKernelArg() failed\n");
		exit(EXIT_FAILURE);
	}

	if (clSetKernelArg(kernel, 1, sizeof(values), &values) != CL_SUCCESS) {
		fprintf(stderr, "clSetKernelArg() failed\n");
		exit(EXIT_FAILURE);
	}

	/* Number of clauses must be a multiple of nr_threads in order to
	 * facilitate the shared clause cache. */
	assert(nr_clauses % nr_threads == 0);
	if (clSetKernelArg(kernel, 2, sizeof(nr_clauses), &nr_clauses) != CL_SUCCESS) {
		fprintf(stderr, "clSetKernelArg() failed\n");
		exit(EXIT_FAILURE);
	}

	if (clSetKernelArg(kernel, 3, sizeof(device_clauses), &device_clauses) != CL_SUCCESS) {
		fprintf(stderr, "clSetKernelArg() failed\n");
		exit(EXIT_FAILURE);
	}

	/* Number of iterations should be a multiple of nr_threads in order
	 * to make the best use of the shared clause cache. */
	assert(n % nr_threads == 0);
	if (clSetKernelArg(kernel, 5, sizeof(n), &n) != CL_SUCCESS) {
		fprintf(stderr, "clSetKernelArg() failed\n");
		exit(EXIT_FAILURE);
	}

	struct timeval tv_a;
	if (gettimeofday(&tv_a, NULL) == -1) {
		fprintf(stderr, "gettimeofday() failed\n");
		exit(EXIT_FAILURE);
	}

	size_t global_work_offset = 0;
	size_t global_work_size = nr_threads;
	size_t local_work_size = nr_threads;

	while (1) {
		printf("c clause index: %u/%u\n", clause_i, nr_clauses);

		if (clSetKernelArg(kernel, 4, sizeof(clause_i), &clause_i) != CL_SUCCESS) {
			fprintf(stderr, "clSetKernelArg() failed\n");
			exit(EXIT_FAILURE);
		}

		if ((err = clEnqueueNDRangeKernel(queue, kernel, 1,
			&global_work_offset,
			&global_work_size,
			&local_work_size,
			0, NULL, NULL)) != CL_SUCCESS)
		{
			printf("%d\n", err);

			fprintf(stderr, "clEnqueue() failed\n");
			exit(EXIT_FAILURE);
		}

		if (clEnqueueReadBuffer(queue, threads, CL_TRUE, 0,
			sizeof(host_threads), host_threads, 0, NULL, NULL) != CL_SUCCESS)
		{
			fprintf(stderr, "clEnqueueReadBuffer() failed\n");
			exit(EXIT_FAILURE);
		}

		if (clFinish(queue) != CL_SUCCESS) {
			fprintf(stderr, "clFinish() failed\n");
			exit(EXIT_FAILURE);
		}

		bool found_solution = false;
		unsigned int max_nr_sat_clauses = 0;

		for (unsigned int i = 0; i < nr_threads; ++i) {
			unsigned int nr_sat_clauses = host_threads[i].nr_sat_clauses;

			if (nr_sat_clauses > max_nr_sat_clauses)
				max_nr_sat_clauses = nr_sat_clauses;

			if (nr_sat_clauses < nr_clauses)
				continue;

			if (!found_solution) {
				/* Fetch all the threads' current valuations
				 * if we didn't already */
				if (clEnqueueReadBuffer(queue, values, CL_TRUE, 0,
					sizeof(host_variables), host_variables, 0, NULL, NULL) != CL_SUCCESS)
				{
					fprintf(stderr, "clEnqueueReadBuffer() failed\n");
					exit(EXIT_FAILURE);
				}

				if (clFinish(queue) != CL_SUCCESS) {
					fprintf(stderr, "clFinish() failed\n");
					exit(EXIT_FAILURE);
				}
			}

			/* We have a solution! */
			found_solution = true;

			/* Double-check on the host that this is a correct solution. */
			for (unsigned int j = 0; j < clauses.size(); ++j) {
				clause &c = clauses[j];

				bool value = false;
				for (unsigned int k = 0; k < c.size(); ++k) {
					literal l = c[k];
					unsigned int var = abs(l);
					unsigned int sign = (l < 0);

					value |= host_variables[var][i] ^ sign;
				}

				assert(value);
			}

			printf("v");
			for (unsigned int j = 1; j < nr_variables; ++j) {
				unsigned int v = reverse_variables[j];
				printf(" %d", host_variables[j][i] ? v : -v);
			}

			printf("\n");
		}

		if (found_solution)
			break;

		printf("c best: %u\n", max_nr_sat_clauses);

		clause_i = (clause_i + n) % nr_clauses;
	}

	struct timeval tv_b;
	if (gettimeofday(&tv_b, NULL) == -1) {
		fprintf(stderr, "gettimeofday() failed\n");
		exit(EXIT_FAILURE);
	}

	struct timeval tv_delta;
	timersub(&tv_b, &tv_a, &tv_delta);
	fprintf(stderr, "c Wall time: %lu.%06lu\n", tv_delta.tv_sec, tv_delta.tv_usec);

	clReleaseMemObject(threads);
	clReleaseMemObject(values);
	clReleaseMemObject(device_clauses);

	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	return 0;
}
