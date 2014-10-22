
#include "math.h"

#include "stdlib.h"

#include "stdio.h"

#include "string.h"

#include "CL/opencl.h"

#include "assert.h"

#include "float.h"

#include "time.h"

#define MAX_SOURCE_SIZE 0x100000

#define MIN_COST -INFINITY

#define MAX_VAR INFINITY

#define NUM_SIMD_ITEM	4

#define WORK_GROUP_SIZE  256


//A structure used to record the upper and lower boundaries for each variable
struct bound
{
	double upper;
	double lower;
};

//This is the function that contains the first phase of the 2-phase simplex method
cl_double* PhaseI(cl_double* tableau,cl_double* cost,int *var,int &t_col,int &t_row,int num_var,int &num_constraint,int num_artificial_var,int num_col,
cl_kernel &pivot_2,cl_kernel &copy_kr,cl_kernel &pre_update,cl_kernel &find_max,cl_kernel &find_r,cl_context &context, cl_command_queue &queue,int &found_status,
int row_temp_num,cl_mem &t_buff,cl_mem &boundary_buff,cl_mem &result_buff,cl_mem &result_buff_2,cl_mem &basic_var_buff,cl_mem &boundary_limit_buff,cl_mem &float_result_buff,
 int* boundary_active,bound* boundary_original);

//function to create kernels 
void createKernel(char *fileName,char* kernelName_0,char* kernelName_1,char *kernelName_2,char *kernelName_3,char *kernelName_4,
cl_kernel &kernelObject_0, cl_kernel &kernelObject_1,cl_kernel &kernelObject_2,cl_kernel &kernelObject_3,cl_kernel &kernelObject_4,
cl_context &context,cl_device_id &device,int &ret);            

//The function that enqueues the kernel for "pivoting" operation on the tableau
void pivot(cl_double* tableau,int r,int k,int t_col,int t_row,int &ret,cl_kernel &pivot_2,cl_kernel &copy_kr,cl_context &context, cl_command_queue &queue,cl_mem &t_buff);

//This function sets some constant(either a constant value or a constant pointer) arguments for all kernel calls.
void setKernelArg(cl_kernel &find_max,cl_kernel &pivot_2,cl_kernel &find_r,cl_kernel &copy_kr,cl_kernel &pre_update,
cl_mem &t_buff,cl_mem &r_buff,cl_mem &result_buff,cl_mem &result_buff_2,cl_mem &col_k_buff,cl_mem &float_result_buff,
int t_col,int t_row,int b_pos,cl_mem& boundary_buff,int length,
cl_mem &boundary_limit_buff,cl_mem &basic_var_buff,int inPhaseI);

//This function enqueues the kernel that find kth column giving maximum Ck-Zk 
int findMax(cl_double* tableau,int t_col,int t_row,int &ret,cl_kernel &find_max, cl_context &context, cl_command_queue &queue,cl_mem &t_buff,cl_mem &result_buff);

//This function enqueues the kernel computing the r
int findR(cl_double* tableau,int t_col,int t_row,int k,int num_col,int &ret,cl_kernel &find_r, cl_context &context, cl_command_queue &queue,cl_mem &t_buff,cl_mem &result_buff,cl_double ceiling,cl_mem &float_result_buff,bool &increase);

//This function preUpdates the tableau with all entering and leaving variables that does not go down to zero.
void preUpdate(cl_kernel &pre_update, cl_context &context, cl_command_queue &queue,int k,int var,cl_double entering_mult,cl_double leaving_mult,int t_row);

int main(int argc,char* argv[]){ 
	
	//A variable used to record the return value for each cl_function call
	cl_int ret;
	
	//**Creating OpenCL Infrastructure**
	//***Creating Platform***
	cl_uint num_platforms;
	ret = clGetPlatformIDs(0, NULL, &num_platforms);
	assert(ret==CL_SUCCESS);
	cl_platform_id* platform_id = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	cl_platform_id platform = NULL;
	clGetPlatformIDs(num_platforms, platform_id, &num_platforms);

	unsigned int i,j;

	for(i = 0; i < num_platforms; ++i){
		char pbuff[100];
		clGetPlatformInfo(platform_id[i],CL_PLATFORM_VENDOR,sizeof(pbuff),pbuff,NULL);
		platform = platform_id[i];
		//if(!strcmp(pbuff, "Altera Corporation")){break;}
		}
			platform = platform_id[0];
	//***Creating Context***
	cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
	cl_context context = clCreateContextFromType(cps, CL_DEVICE_TYPE_ALL, NULL, NULL, &ret);
	assert(ret==CL_SUCCESS);

	//***Creating Device***
	size_t deviceListSize;
	ret = clGetContextInfo(context,CL_CONTEXT_DEVICES,0, NULL,&deviceListSize);
	assert(ret==CL_SUCCESS);
	cl_device_id *devices = (cl_device_id *)malloc(deviceListSize);
	clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceListSize,devices,NULL);
	cl_device_id device = devices[0];
	char* value;
    size_t valueSize;
	//Temporary file used to save debugging info
	FILE *debug_info=fopen("debug_info.txt","w");
	
	//**Creating OpenCL Command Queue**
	cl_command_queue command_queue = clCreateCommandQueue(context, device, 0, &ret);
	assert(ret==CL_SUCCESS);
	/*load problem description from file*/
	printf("Which problem set do you wish to run?\n");
	int num_var,num_constraint,num_limit;
	//open file
	FILE *fp_problem;
	char *problemFile=(char*)malloc(20*sizeof(char));
	scanf("%s",problemFile);
	printf("%s\n",problemFile);
	fp_problem=fopen(problemFile, "r");
		if (!fp_problem) {
				fprintf(stderr, "Failed to load problem file.\n");
				exit(1);
			}
	//Reading the number of variables, the number of constraints and the number of variables that has boundary conditions.
	fscanf(fp_problem,"%d %d %d",&num_var,&num_constraint,&num_limit);
	printf("%d %d %d\n",num_var,num_constraint,num_limit);
	
	//The num_col is the width of the constraint matrix including slack variable
	int num_col=num_var+num_constraint;

	/*
	*the following lines defines the container that are reused to hold problem description
	*/
	
	cl_double *cost=(cl_double*)malloc(num_col*sizeof(cl_double));
	cl_double *constraint= (cl_double*)malloc(num_col*num_constraint*sizeof(cl_double));
	cl_double *b=(cl_double*)malloc(num_constraint*sizeof(cl_double));
	bound *boundary=(bound*)malloc((num_col+1)*sizeof(bound));
	/*
	*This container holds the information of the sign of each constraint
	* 0 means Ax<=b; 1 means Ax>=b; 2 means Ax=b
	*/
	int *lessThan=(int*)malloc(num_constraint*sizeof(int));
	
	//A variable recording the number of artificial variables introduced into the tableau
	int num_artificial_var=0;
	//Reading the objective function and initializing boundary conditions to be non-negative as default.
	for (i=0;i<num_var;i++)
		{
			fscanf(fp_problem,"%lf",&cost[i]);
			boundary[i+1].lower=0;
			boundary[i+1].upper=MAX_VAR;
		}
	for (i=num_var;i<num_col;i++)
		{
			boundary[i+1].lower=0;
			boundary[i+1].upper=MAX_VAR;
		}
	//Slack variables initially has zero coefficient in objective function
	for (i=num_var;i<num_col;i++)
			cost[i]=0;
	//Reading the constraint matrix Ax(<=or>=or=)b 
	for (i=0;i<num_constraint;i++)
	{
		for (j=0;j<num_var;j++)
		{
			fscanf(fp_problem,"%lf",&constraint[i*num_col+j]);
		}
		fscanf(fp_problem,"%lf",&b[i]);
		fscanf(fp_problem,"%d",&lessThan[i]);
	}
	
	//reading in the additional boundary condition for each variable;
	for (i=0;i<num_limit;i++)
	{
		int var_num;
		fscanf(fp_problem,"%d",&var_num);
		fscanf(fp_problem,"%lf %lf",&boundary[var_num].lower,&boundary[var_num].upper);
	}

	/*
	*A variable to keep track of the number of slack variables that need to be introduced into the tableau
	*if Ax<=b(Lessthan[i]==0), a slack variable is introduced
	*IF aX>=B(Lessthan[i]==1), a slack variable and an artificial variable are introduced
	*if Ax=b (Lessthan[i]==2), only an artificial variable is introduced 
	*/
	
	int row_temp_num=0;
	for (i=0;i<num_constraint;i++)
			{ 
				for (j=0;j<num_constraint;j++)
					constraint[i*num_col+num_var+j]=0;
				if (lessThan[i]==0)
					{
						constraint[i*num_col+num_var+row_temp_num]=1;
						row_temp_num++;
					}
					else if (lessThan[i]==1)
						{
							constraint[i*num_col+num_var+row_temp_num]=-1;
							num_artificial_var++;
							row_temp_num++;
						}
						else num_artificial_var++;
			}
			
		//save the number of columns before resizing the matrix
		int num_col_old=num_col;
		num_col=num_var+row_temp_num;

	fclose(fp_problem);
	
	/*
	*Phase 1
	*building the tableau with introduced artificial variable
	*/
	
	int t_col=num_col+2+num_artificial_var;
	int divisor=WORK_GROUP_SIZE;
	//To guarantee the tableau width is multiples of the workgroup size
	if (t_col%divisor!=0) t_col+=divisor-t_col%divisor;
	int t_row=num_constraint+1;
	//Creating the tableau container
	cl_double *tableau=(cl_double*)malloc(t_col*(t_row+1)*sizeof(cl_double));
	tableau[0]=1;
	
	//initialize the redundant row left to keep track of the real cost function
	for (i=0;i<t_col;i++)
		tableau[t_row*t_col+i]=0;

	/*
	*this loop copies copy objective function to the last row of the tableau 
	*The first row is filled with -1 for artificial variables and 0 for other variables
	*/
	for (i=0;i<num_col+num_artificial_var;i++)
			if (i>=num_col) 
				tableau[i+1]=-1; 
			else 
				{
					tableau[i+1]=0;
					if (cost[i]!=0) 
						tableau[t_row*t_col+i+1]=-cost[i]; 
				}

	/*
	* In this section the b vector is introduced into the tableau
	* At the start, all variables are at their lower ground.
	* The variable "diff" is used to initialize the basic variables' values when all variables are at their lower ground
	*/
	cl_double diff=0;			
	for (j=0;j<num_var;j++)
				diff+=cost[j]*boundary[j+1].lower;

	tableau[t_row*t_col+num_col+num_artificial_var+1]=diff;
	tableau[num_col+num_artificial_var+1]=0;

	for (i=0;i<num_constraint;i++)
	{
		tableau[(i+1)*t_col]=0;
		diff=0;
		for (j=0;j<num_var;j++)
				diff-=boundary[j+1].lower*constraint[i*num_col_old+j];
		tableau[(i+1)*t_col+num_col+num_artificial_var+1]=b[i]+diff;
		//A loop doing zero padding 
		for (j=0;j<(t_col-num_col-num_artificial_var-2);j++)	
			tableau[(i+1)*t_col-1-j]=0;	
	}
	
	//zero padding to make the tableau multiple of workgroup size
	for (j=0;j<(t_col-num_col-num_artificial_var-2);j++)	
			tableau[(i+1)*t_col-1-j]=0;

	//A variable used to keep track of the position of last artificial variable inserted into the tableau
	int assigned_var=0;
	//A variable used to keep track of the position of last slack variable introduced
	int slack_var_pos=1;
	//An array recording all the variables currently in the basis
	int *var=(int*)malloc(num_constraint*sizeof(int));
	
	//copy the constraints table into the tableau
	for (i=0;i<num_constraint;i++)
	{
		for (j=0;j<num_col+num_artificial_var;j++)
			if (j<num_col)	
				tableau[(i+1)*t_col+j+1]=constraint[i*num_col_old+j];
			else
				tableau[(i+1)*t_col+j+1]=0;
		if (lessThan[i]!=0) 
				{
					tableau[(i+1)*t_col+num_col+assigned_var+1]=1;
					var[i]=num_col+assigned_var+1;
					assigned_var++;
					
				}
				else
					var[i]=num_var+slack_var_pos;
		if (lessThan[i]!=2) slack_var_pos++;
	}

	//fill the entries for artificial variables
	for (int i=1;i<num_col+num_artificial_var+2;i++)
		for (int j=0;j<num_constraint;j++)
			if (lessThan[j]!=0) tableau[i]+=tableau[(j+1)*t_col+i];
	//declaring the k column and r row which defines the point of pivoting
	int k,r;
	
	/*
	*below is the declaration of all memory object used in this method
	*t_buff is used to copy the tableau
	*result_buff are used to read result from the two kernels finding k and r
	*boundary_buff is used to pass the boundary conditions to the kernel 
	*basic_var_buff is used to pass the var array which records the current basic variables
	*/
	
	cl_mem r_buff = clCreateBuffer(context, CL_MEM_READ_WRITE,t_col*sizeof(cl_double),NULL,&ret);
	assert(ret==CL_SUCCESS);	

	cl_mem t_buff = clCreateBuffer(context, CL_MEM_READ_WRITE,t_col*(t_row+1)*sizeof(cl_double),NULL,&ret);
	assert(ret==CL_SUCCESS);	

	cl_mem result_buff=clCreateBuffer(context, CL_MEM_READ_ONLY,sizeof(cl_int),NULL,&ret);
	assert(ret==CL_SUCCESS);

	cl_mem result_buff_2=clCreateBuffer(context, CL_MEM_READ_ONLY,sizeof(cl_int),NULL,&ret);
	assert(ret==CL_SUCCESS);

	cl_mem float_result_buff=clCreateBuffer(context, CL_MEM_READ_ONLY,sizeof(cl_double),NULL,&ret);
	assert(ret==CL_SUCCESS);
	
	cl_mem boundary_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY,(num_col+num_artificial_var+1)*sizeof(int),NULL,&ret);
	assert(ret==CL_SUCCESS);	
	
	cl_mem boundary_limit_buff=clCreateBuffer(context, CL_MEM_WRITE_ONLY,(num_col+num_artificial_var+1)*sizeof(bound),NULL,&ret);
	assert(ret==CL_SUCCESS);
	
	cl_mem basic_var_buff = clCreateBuffer(context, CL_MEM_READ_WRITE,num_constraint*sizeof(int),NULL,&ret);
	assert(ret==CL_SUCCESS);
	
	cl_mem col_k_buff = clCreateBuffer(context, CL_MEM_READ_WRITE,t_row*sizeof(cl_double),NULL,&ret);
	assert(ret==CL_SUCCESS);	


	clock_t ts,te,t1,t2;
	ts=clock();
	// 1/divider to eliminate division in kernel so as to minimize delay 
	cl_double dividerInv;
	
	//creating kernels by calling the CreatKernel function
	cl_kernel find_max;
	cl_kernel kernel_pivot_2;
	cl_kernel find_r;
	cl_kernel copy_kr;
	cl_kernel pre_update;
	createKernel("./pivot_2.cl","find_max","pivot_2","find_r","copy_kr","pre_update",find_max,kernel_pivot_2,find_r,copy_kr,pre_update,context,device,ret);
	printf("setting kernel arguments\n");
	int *boundary_active=(int*)malloc((num_col+num_artificial_var+1)*sizeof(int));
	for (i=0;i<num_col+num_artificial_var+1;i++) 
		boundary_active[i]=0;
	//set the kernel argument provided the argument is a pointer or constant
	setKernelArg(find_max,kernel_pivot_2,find_r,copy_kr,pre_update,t_buff,r_buff,result_buff,result_buff_2,col_k_buff,float_result_buff,t_col,t_row+1,
	num_col+num_artificial_var+1,boundary_buff,num_col+num_artificial_var+2,boundary_limit_buff,basic_var_buff,1);

	/*
	*step 2 of phase I
	*solving the method for artificial variable
	*/
	ret = clEnqueueWriteBuffer(command_queue, t_buff, CL_TRUE, 0, t_col*(t_row+1)*sizeof(cl_double), tableau, 0, NULL, NULL);
	assert(ret==CL_SUCCESS);
	int it=0;
	
	
	
	//initialize variable record for variables in the basis

	//a variable keeping track of whether a solution is found. 0 = not found, 1 = found optimal solution 2=unbounded LP
	int found_status=0;

	printf("artificial var: %d\n",num_artificial_var);
				
	if (num_artificial_var==0) found_status=1;
	tableau=PhaseI(tableau,cost,var,t_col,t_row,num_var,num_constraint,num_artificial_var,num_col,kernel_pivot_2,
	copy_kr,pre_update,find_max,find_r,context,command_queue,found_status,row_temp_num,t_buff,boundary_buff,result_buff,result_buff_2,basic_var_buff,boundary_limit_buff,float_result_buff,
	boundary_active,boundary);
	//if phase I is able to reach a feasible solution, reset status flag for phase II
	if(found_status==1)	found_status=0; 
		

	//A boolean array recording whether the current variable is at its lower or upper boundary,0=lower and 1=upper
	boundary_active=(int*)malloc((num_col+1)*sizeof(int));
	for (i=1;i<num_col+1;i++) 
		boundary_active[i]=0;
		
	/*
	* The code below is simply resizing all the buffers to fit the new tableau for Phase II
	*/
		
	r_buff = clCreateBuffer(context, CL_MEM_READ_WRITE,t_col*sizeof(cl_double),NULL,&ret);
	assert(ret==CL_SUCCESS);	

	t_buff = clCreateBuffer(context, CL_MEM_READ_WRITE,t_col*t_row*sizeof(cl_double),NULL,&ret);
	assert(ret==CL_SUCCESS);	

	boundary_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY,(num_col+1)*sizeof(int),NULL,&ret);
	assert(ret==CL_SUCCESS);	

	//set the kernel argument provided the argument is a pointer or constant
	setKernelArg(find_max,kernel_pivot_2,find_r,copy_kr,pre_update,t_buff,r_buff,result_buff,result_buff_2,col_k_buff,float_result_buff,t_col,t_row,num_col+1,boundary_buff,num_col+2,boundary_limit_buff,basic_var_buff,0);
	/*
	*Below is the iteration loop which move from one basis to another having lower cost function value until a solution is found or no solution exists
	*/
	cl_event t_buff_event[1];
	ret = clEnqueueWriteBuffer(command_queue, t_buff, CL_TRUE, 0, t_col*t_row*sizeof(cl_double), tableau, 0, NULL, t_buff_event);
	assert(ret==CL_SUCCESS);

	while (found_status==0)
	{
		cl_double MaxCost=MIN_COST;
		ret = clEnqueueWriteBuffer(command_queue, boundary_buff, CL_TRUE, 0, (num_col+1)*sizeof(int), boundary_active, 0, NULL, NULL);
		assert(ret==CL_SUCCESS);	
		k=findMax(tableau,num_col+2,t_row,ret,find_max, context,command_queue,t_buff,result_buff);
		int k_ref=0;
		/*
		*This following loop looks for k, the column of maximum rate of reduction in objective function.
		*the cost is negated if the current variable is at upper boundary, which means the variable are to be decreased.
		*/
		//a variable keep tracking of whether variable k is to be increased or decreased
		bool increase=false;
		//a variable keep tracking of whether there is entering and leaving of variables to and from the basis
		bool no_basic_var_change=false;
		//if all direction cannot improve the objective function any more, optimal solution is found,else find row r
		if (k==-1) found_status=1;
		else 
			{
				//upper bound- lower bound gives the maximum possible increase in the entering variable
				cl_double ceiling=boundary[k].upper-boundary[k].lower;
				ret = clEnqueueWriteBuffer(command_queue, boundary_limit_buff, CL_TRUE, 0, (num_col+1)*sizeof(bound), boundary, 0, NULL, NULL);
				assert(ret==CL_SUCCESS);
				ret = clEnqueueWriteBuffer(command_queue, basic_var_buff, CL_TRUE, 0, num_constraint*sizeof(cl_int), var, 0, NULL, NULL);
				assert(ret==CL_SUCCESS);
				cl_double min_xk=MAX_VAR;
				//Call the function that enques the kernel computing the value of r
				r=findR(tableau,t_col,t_row,k,num_col,ret,find_r,context,command_queue,t_buff,result_buff_2,ceiling,float_result_buff,increase);

			}
			//If unbounded, exit the loop
			if (r==0) found_status=2;	
		//if solution is not found or LP is not considered unbounded, do pivoting
		if (found_status==0) 
			{	
				//determining the multiplier for entering variable column and leaving variable column that to be added to the RHS
				cl_double entering_mult,leaving_mult;
				if (boundary_active[k]==0)
					entering_mult=boundary[k].lower;
				else
					entering_mult=boundary[k].upper;
				if (increase)
				{
					leaving_mult=boundary[var[r-1]].upper;
					boundary_active[var[r-1]]=true;
				}
				else 
				{
					leaving_mult=boundary[var[r-1]].lower;
					boundary_active[var[r-1]]=false;
				}
				//pre-updating the RHS for bounded simplex method

				preUpdate(pre_update, context, command_queue ,k,var[r-1],entering_mult,leaving_mult,t_row);
				//if basic variable will change, do pivoting
				if (!no_basic_var_change)
				{
					pivot(tableau,r,k,t_col,t_row,ret,kernel_pivot_2,copy_kr, context, command_queue,t_buff);
					var[r-1]=k;
				}
			}
			it++;
	}
/*
*After the loop, determine whether a solution is found
*If one is found, print the variables and the objective function value at optimal point
*/
	te=clock();
	ret = clEnqueueReadBuffer(command_queue, t_buff, CL_TRUE, 0,t_col*t_row*sizeof(cl_double),tableau, 0, NULL,NULL);
	assert(ret==CL_SUCCESS);
	if (found_status==1)
	{
		printf("found an optimal solution!\n");
		for (i=0;i<num_constraint;i++)
			printf("Variable %d has value: %8.2f\n",var[i],tableau[(i+1)*t_col+num_col+1]);
		printf("objective function value is %.10f\n",tableau[num_col+1]);
	}
	else if( found_status==2)
		printf("The Linear Problem is unbounded! no finite optimal solution!\n");
	printf("running time is: %d\n",te-ts);
	//***Releasing OpenCL Memory Objects***
	ret = clReleaseKernel(kernel_pivot_2);
	assert(ret==CL_SUCCESS);
	
	ret = clReleaseKernel(find_max);
	assert(ret==CL_SUCCESS);
	
	ret = clReleaseKernel(find_r);
	assert(ret==CL_SUCCESS);

	ret = clReleaseCommandQueue(command_queue);
	assert(ret==CL_SUCCESS);

	ret = clReleaseContext(context);
	assert(ret==CL_SUCCESS);
	
	ret=clReleaseMemObject(r_buff);
	assert(ret==CL_SUCCESS);

	ret=clReleaseMemObject(t_buff);
	assert(ret==CL_SUCCESS);

	ret=clReleaseMemObject(result_buff);
	assert(ret==CL_SUCCESS);

	ret=clReleaseMemObject(result_buff_2);
	assert(ret==CL_SUCCESS);
	
	ret=clReleaseMemObject(boundary_buff);
	assert(ret==CL_SUCCESS);
	
	ret=clReleaseMemObject(basic_var_buff);
	assert(ret==CL_SUCCESS);
	free(tableau);
	free(constraint);
	free(cost);
	free(b);
	fclose(debug_info);
	return 0;
}	


void createKernel(char *fileName,char* kernelName_0,char* kernelName_1,char *kernelName_2,char *kernelName_3,char *kernelName_4,
cl_kernel &kernelObject_0, cl_kernel &kernelObject_1,cl_kernel &kernelObject_2,cl_kernel &kernelObject_3,cl_kernel &kernelObject_4,
cl_context &context,cl_device_id &device,int &ret)
{
FILE *fp;
	char *source_str;
	size_t source_size;
	/* Load kernel code */
	printf("creating kernel for %s\n",fileName);
	fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel \n");
		exit(1);
		}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);
	assert(ret==CL_SUCCESS);	
	/*create program*/
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str,(const size_t *)&source_size, &ret);
	ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	assert(ret==CL_SUCCESS);

	/*Creating Kernel Object*/
	kernelObject_0 = clCreateKernel(program,kernelName_0,&ret);
	kernelObject_1 = clCreateKernel(program,kernelName_1,&ret);
	kernelObject_2 = clCreateKernel(program,kernelName_2,&ret);
	kernelObject_3 = clCreateKernel(program,kernelName_3,&ret);
	kernelObject_4 = clCreateKernel(program,kernelName_4,&ret);
	/*clean up memory object*/
	assert(ret==CL_SUCCESS);
	ret=clReleaseProgram(program);
	assert(ret==CL_SUCCESS);
	printf("kernel created\n");
}

void pivot(cl_double* tableau,int r,int k,int t_col,int t_row,int &ret,cl_kernel &pivot_2,cl_kernel &copy_kr, cl_context &context, cl_command_queue &queue,cl_mem &t_buff)
{
	//loop control variables
	int i,j;
	cl_double dividerInv=1/tableau[r*t_col+k];
	//setting kernel arguments that changes at while running

	ret = clSetKernelArg(pivot_2, 1, sizeof(cl_double), (void*)&dividerInv);
	assert(ret==CL_SUCCESS);

	ret = clSetKernelArg(pivot_2, 2, sizeof(cl_int), (void*)&k);
	assert(ret==CL_SUCCESS);

	ret = clSetKernelArg(pivot_2, 4, sizeof(cl_int), (void*)&r);
	assert(ret==CL_SUCCESS);

	ret = clSetKernelArg(copy_kr, 5, sizeof(cl_int), (void*)&k);
	assert(ret==CL_SUCCESS);

	ret = clSetKernelArg(copy_kr, 6, sizeof(cl_int), (void*)&r);
	assert(ret==CL_SUCCESS);

	/*create opencl events for synchronization*/

	cl_event write_events[1];
	cl_event kernel_2_event[1];
	cl_event kernel_1_event[1];
	cl_event buffer_events[1];

	/*the width of the tableau must be evenly divisible by the work group size*/
	int global_width=t_col-t_col%WORK_GROUP_SIZE;
	if (t_col%WORK_GROUP_SIZE!=0) global_width+=WORK_GROUP_SIZE;
	size_t copy_kr_globalws=global_width;
	size_t copy_kr_localws=WORK_GROUP_SIZE;
	
	/*Enqueue the kernel of copying column k and row r to another location so no racing condition will occur*/
	ret = clEnqueueNDRangeKernel(queue,copy_kr,(cl_uint)1,0, &copy_kr_globalws,&copy_kr_localws,0,NULL,kernel_1_event);
	assert(ret==CL_SUCCESS);

	//printf("global_width %d t_col %d t_row %d\n",global_width,t_col,t_row);
	size_t pivot_2_globalws[2]={t_row,global_width};
	size_t pivot_2_localws[2]={1,WORK_GROUP_SIZE};
	cl_uint dimension=2;

	/*Enqueue the kernel of pivoting*/
	ret = clEnqueueNDRangeKernel(queue,pivot_2,(cl_uint)dimension,0, pivot_2_globalws,pivot_2_localws,1,kernel_1_event,kernel_2_event);
	assert(ret==CL_SUCCESS);
		
}

cl_double* PhaseI(cl_double* tableau,cl_double* cost,int *var,int &t_col,int &t_row,int num_var,int &num_constraint,int num_artificial_var,int num_col,
cl_kernel &pivot_2,cl_kernel &copy_kr,cl_kernel &pre_update,cl_kernel &find_max,cl_kernel &find_r,cl_context &context, cl_command_queue &queue,int &found_status,
int row_temp_num,cl_mem &t_buff,cl_mem &boundary_buff,cl_mem &result_buff,cl_mem &result_buff_2,cl_mem &basic_var_buff,cl_mem &boundary_limit_buff,cl_mem &float_result_buff,
 int* boundary_active,bound* boundary_original)
{
	int i,j,k,r;
	int it=0;
	int ret;
	/*
	*enlarge the size of the boundary arrays for artificial variables
	*All artificial variables are initialized with lower bound as 0 and upper bound as infinity
	*/
	bound* boundary=(bound*)malloc((num_col+num_artificial_var+1)*sizeof(bound));
	for (i=0;i<num_col+num_artificial_var+1;i++)
	{
		if (i<num_col+1)
			boundary[i]=boundary_original[i];
		else
			{
				boundary[i].lower=0;
				boundary[i].upper=MAX_VAR;
			}
	}
	//solution loop for each movement from one basis to another
	
	while (found_status==0)
	{
		cl_double MaxCost=MIN_COST;
		ret = clEnqueueWriteBuffer(queue, boundary_buff, CL_TRUE, 0, (num_col+num_artificial_var+1)*sizeof(int), boundary_active, 0, NULL, NULL);
		assert(ret==CL_SUCCESS);	
		k=findMax(tableau,num_col+num_artificial_var+2,t_row,ret,find_max, context,queue,t_buff,result_buff);
		int k_ref=0;
		/*
		*This following loop looks for k, the column of maximum rate of reduction in objective function.
		*the cost is negated if the current variable is at upper boundary, which means the variable are to be decreased.
		*/
		//a variable keep tracking of whether variable k is to be increased or decreased
		bool increase=false;
		//a variable keep tracking of whether there is entering and leaving of variables to and from the basis
		bool no_basic_var_change=false;
		//if all direction cannot improve the objective function any more, optimal solution is found,else find row r
		if (k==-1) found_status=1;
		else 
			{
				//upper bound- lower bound gives the maximum possible increase in the entering variable
				cl_double ceiling=boundary[k].upper-boundary[k].lower;
				ret = clEnqueueWriteBuffer(queue, boundary_limit_buff, CL_TRUE, 0, (num_col+num_artificial_var+1)*sizeof(bound), boundary, 0, NULL, NULL);
				assert(ret==CL_SUCCESS);
				ret = clEnqueueWriteBuffer(queue, basic_var_buff, CL_TRUE, 0, num_constraint*sizeof(cl_int), var, 0, NULL, NULL);
				assert(ret==CL_SUCCESS);
				cl_double min_xk=MAX_VAR;
				r=findR(tableau,t_col,t_row,k,num_col+num_artificial_var,ret,find_r,context,queue,t_buff,result_buff_2,ceiling,float_result_buff,increase);

			}
			if (r==0) found_status=2;	
		if (found_status==0) 
			{	
				//determining the multiplier for entering variable column and leaving variable column that to be added to the RHS
				cl_double entering_mult,leaving_mult;
				if (boundary_active[k]==0) 
					entering_mult=boundary[k].lower;
				else
					entering_mult=boundary[k].upper;
				if (increase)
				{
					leaving_mult=boundary[var[r-1]].upper;
					boundary_active[var[r-1]]=true;
				}
				else 
				{
					leaving_mult=boundary[var[r-1]].lower;
					boundary_active[var[r-1]]=false;
				}
				//pre-updating the RHS for bounded simplex method
				preUpdate(pre_update, context,queue ,k,var[r-1],entering_mult,leaving_mult,t_row);
				//if basic variable will change, do pivoting
				if (!no_basic_var_change)
				{
					pivot(tableau,r,k,t_col,t_row+1,ret,pivot_2,copy_kr, context,queue,t_buff);
					var[r-1]=k;
				}
			}
		it++;
	}

	ret = clEnqueueReadBuffer(queue, t_buff, CL_TRUE, 0,t_col*(t_row+1)*sizeof(cl_double),tableau, 0, NULL,NULL);
	assert(ret==CL_SUCCESS);
	/*
	*end of phase I solution loop
	*Now starts to remove artificial variables to prepare for phase II
	*/
	for (int i=0;i<num_constraint;i++)
	{
		double val=tableau[(i+1)*t_col+num_col+num_artificial_var+1];
	}
	printf("finished Phase I iteration loop, start analyzing intermediate result!\n");
	//check if all artificial variables has left the basis, if not, check if Xa=0
	for (int i=0;i<num_constraint;i++)
		if (var[i]>num_var+row_temp_num)
		{
			//if Xa!=0 then no feasible solution exists
			if ((tableau[(i+1)*t_col+num_col+num_artificial_var+1]<=-0.001)||(tableau[(i+1)*t_col+num_col+num_artificial_var+1]>=0.001))
				{
					printf("No optimal solution as artificial variable cannot be eliminated\n");
					found_status=3;
					break;
				}
			else 
/*
*if Xa=0 proceeds with the tableau or eliminate artificial varibale from basis if there is one
*The array var_state is used to record whether a basic variable is artificial
*if var[j] is non-artificial variable, its corresponding position in var_state will be marked true
*This 
*/
				{
					printf("Eliminating basic artificial variables from the basis\n");
					bool *var_state=(bool*)malloc((num_var+1+row_temp_num)*sizeof(bool));
					for (j=1;j<=num_var+row_temp_num;j++)
						var_state[j]=false;
					for (j=0;j<num_constraint;j++)
						if (var[j]<=num_var) var_state[var[j]]=true;
					int non_basic_var=0;
					//Find the position of non_basic_variable that will enter the basis
					for (j=1;j<=num_var+row_temp_num;j++)
						if (!var_state[j])
						{
							non_basic_var=j;
							break;
						}
					//if no non_basic_variable can enter, delete the row of zero_basic_artificial_variable
					if (tableau[(i+1)*t_col+non_basic_var]==0)
						{
							for (j=i+1;j<t_row;j++)
								{
									for (int k=0;k<t_col;k++)
										tableau[j*t_col+k]=tableau[(j+1)*t_col+k];
									var[j-1]=var[j];
								}
							t_row--;
							num_constraint--;
						}

					else
					//If one non-artificial non-basic variable is found, enter the variable into the basis
						if (non_basic_var!=0)
							{
								pivot(tableau,i+1,non_basic_var,t_col,t_row+1,ret,pivot_2,copy_kr,context, queue,t_buff);
								var[i]=non_basic_var;
							}
						//Eliminate the basic artificial variable by introducing non-artificial variable into the basis
					
				}
		}
	//copy the problem cost function to the first row of tableau
	for (i=1;i<num_col+num_artificial_var+2;i++)
		tableau[i]=tableau[t_row*t_col+i];

	/*
	*eliminate the artificial variable from the tableau 
	*a new tableau is built by excluding the columns of artificial variables
	*/
	int t_col_old=t_col;
	t_col=num_col+2;
	//The width of the new tableau must also be evenly divisible by work group size
	int divisor=WORK_GROUP_SIZE;
	if (t_col%divisor!=0) t_col+=divisor-t_col%divisor;
	t_row=num_constraint+1;
	cl_double*tableau_temp=tableau;
	tableau=(cl_double*)malloc(t_col*t_row*sizeof(cl_double));
	for (i=0;i<t_row;i++)
		for (j=0;j<t_col;j++)
			if (j<num_col+1) 
				tableau[i*t_col+j]=tableau_temp[i*t_col_old+j];
			else
				tableau[i*t_col+j]=tableau_temp[i*t_col_old+num_artificial_var+j];
	free(tableau_temp);
	//fclose(tableau_out);
	return tableau;
}

/*
*This function set the invariant arguments for all kernels
*Invariant arguments can be constant variables or constant pointers
*/
void setKernelArg(cl_kernel &find_max,cl_kernel &pivot_2,cl_kernel &find_r,cl_kernel &copy_kr,cl_kernel &pre_update,
cl_mem &t_buff,cl_mem &r_buff,cl_mem &result_buff,cl_mem &result_buff_2,cl_mem &col_k_buff,cl_mem &float_result_buff,
int t_col,int t_row,int b_pos,cl_mem& boundary_buff,int length,
cl_mem &boundary_limit_buff,cl_mem &basic_var_buff,int inPhaseI)
{
	int ret;
	ret = clSetKernelArg(pivot_2, 0, sizeof(cl_mem), (void *)&t_buff);
	assert(ret==CL_SUCCESS);
	
	ret = clSetKernelArg(pivot_2, 3, sizeof(cl_int), (void*)&t_col);
	assert(ret==CL_SUCCESS);
	
	ret = clSetKernelArg(pivot_2, 5, sizeof(cl_int), (void *)&t_row);
	assert(ret==CL_SUCCESS);
	
	ret = clSetKernelArg(pivot_2, 6, sizeof(cl_mem), (void *)&r_buff);
	assert(ret==CL_SUCCESS);
	
	ret = clSetKernelArg(pivot_2, 7, sizeof(cl_mem), (void *)&col_k_buff);
	assert(ret==CL_SUCCESS);
	
	ret = clSetKernelArg(find_max, 0, sizeof(cl_mem), (void *)&t_buff);
	assert(ret==CL_SUCCESS);

	ret = clSetKernelArg(find_max, 3, sizeof(cl_int), (void *)&length);
	assert(ret==CL_SUCCESS);

	ret = clSetKernelArg(find_max, 4, sizeof(cl_mem), (void *)&boundary_buff);
	assert(ret==CL_SUCCESS);
	
	ret = clSetKernelArg(find_max, 5, sizeof(cl_mem), (void *)&result_buff);
	assert(ret==CL_SUCCESS);
	
	ret = clSetKernelArg(find_r, 0, sizeof(cl_mem), (void *)&t_buff);
	assert(ret==CL_SUCCESS);

	ret = clSetKernelArg(find_r, 4, sizeof(cl_int), (void *)&b_pos);
	assert(ret==CL_SUCCESS);
	
	ret = clSetKernelArg(find_r, 5, sizeof(cl_int), (void *)&t_col);
	assert(ret==CL_SUCCESS);
	
	int length_r=t_row-inPhaseI;	
	ret = clSetKernelArg(find_r, 6, sizeof(cl_int), (void *)&length_r);
	assert(ret==CL_SUCCESS);

	ret = clSetKernelArg(find_r, 7, sizeof(cl_mem), (void *)&result_buff_2);
	assert(ret==CL_SUCCESS);
	
	ret = clSetKernelArg(find_r, 8, sizeof(cl_mem), (void *)&boundary_buff);
	assert(ret==CL_SUCCESS);
	
	ret = clSetKernelArg(find_r, 9, sizeof(cl_mem), (void *)&boundary_limit_buff);
	assert(ret==CL_SUCCESS);
	
	ret = clSetKernelArg(find_r, 10, sizeof(cl_mem), (void *)&basic_var_buff);
	assert(ret==CL_SUCCESS);

	ret = clSetKernelArg(find_r, 12, sizeof(cl_mem), (void *)&float_result_buff);
	assert(ret==CL_SUCCESS);
	
	ret = clSetKernelArg(copy_kr, 0, sizeof(cl_mem), (void *)&t_buff);
	assert(ret==CL_SUCCESS);
	
	ret = clSetKernelArg(copy_kr, 1, sizeof(cl_mem), (void *)&r_buff);
	assert(ret==CL_SUCCESS);
	
	ret = clSetKernelArg(copy_kr, 2, sizeof(cl_mem), (void *)&col_k_buff);
	assert(ret==CL_SUCCESS);
	
	ret = clSetKernelArg(copy_kr, 3, sizeof(cl_int), (void *)&t_col);
	assert(ret==CL_SUCCESS);
	
	ret = clSetKernelArg(copy_kr, 4, sizeof(cl_int), (void *)&t_row);
	assert(ret==CL_SUCCESS);
	
	ret = clSetKernelArg(pre_update, 0, sizeof(cl_mem), (void *)&t_buff);
	assert(ret==CL_SUCCESS);
	
	ret = clSetKernelArg(pre_update, 5, sizeof(cl_int), (void *)&b_pos);
	assert(ret==CL_SUCCESS);
	
	ret = clSetKernelArg(pre_update, 6, sizeof(cl_int), (void *)&t_col);
	assert(ret==CL_SUCCESS);
	
}

/*
*This function enqueues the findMax kernel which looks for the k value
*/
int findMax(cl_double* tableau,int t_col,int t_row,int &ret,cl_kernel &find_max, cl_context &context, cl_command_queue &queue,cl_mem &t_buff,cl_mem &result_buff)
{
	cl_event write_events[1];
	cl_event kernel_event[1];
	cl_event read_events[2];
	//A buffer that are used to point to the first row of tableau

	//to make the length of the array 2^n so as to ensure correct computation of offset
	int two_power=2;
	int size=t_col;
	while (size>=2) 
		{
			
			size/=2;
			two_power*=2;
		}	
	//initializing the local array of the kernel
	ret = clSetKernelArg(find_max, 1, two_power*sizeof(cl_double), NULL);
	assert(ret==CL_SUCCESS);
	ret = clSetKernelArg(find_max, 2, two_power*sizeof(cl_int), NULL);
	assert(ret==CL_SUCCESS);
	const size_t kernel_paths = two_power;
	const size_t local_kernel_paths = two_power;
	//Enqueuing the kernel of find_max
	ret = clEnqueueNDRangeKernel(queue, find_max, (cl_uint) 1, 0, &kernel_paths, &local_kernel_paths, 0, NULL, kernel_event);
	assert(ret==CL_SUCCESS);
	
	cl_int *z = (cl_int*) malloc(sizeof(cl_int));
	//**Reading the results**
	ret = clEnqueueReadBuffer(queue, result_buff, CL_TRUE, 0, sizeof(cl_int),z, 1, kernel_event, &read_events[0]);
	assert(ret==CL_SUCCESS);
	
	ret=clWaitForEvents (1, read_events);
	assert(ret==CL_SUCCESS);
	int k=z[0];	
	free(z);
	return k;
}

int findR(cl_double* tableau,int t_col,int t_row,int k,int num_col,int &ret,cl_kernel &find_r, cl_context &context, cl_command_queue &queue,cl_mem &t_buff,cl_mem &result_buff,cl_double ceiling,cl_mem &float_result_buff,bool &increase)
{
	
	cl_event write_events[1];
	cl_event kernel_event[1];
	cl_event read_events[1];
	ret = clSetKernelArg(find_r, 3, sizeof(cl_int), (void *)&k);
	assert(ret==CL_SUCCESS);
	ret = clSetKernelArg(find_r, 11, sizeof(cl_double), (void *)&ceiling);
	assert(ret==CL_SUCCESS);

	//to make the length of the array 2^n so as to ensure correct computation of offset
	int two_power=2;
	int size=t_row;
	while (size>=2) 
		{
			size/=2;
			two_power*=2;
		}	
	//initializing local array of find_r kernel
	ret = clSetKernelArg(find_r, 1,  two_power*sizeof(cl_double), NULL);
	assert(ret==CL_SUCCESS);
	ret = clSetKernelArg(find_r, 2, two_power*sizeof(cl_int), NULL);
	assert(ret==CL_SUCCESS);

	const size_t kernel_paths = two_power;
	const size_t local_kernel_paths = two_power;
	//enqueuing the kernel
	ret = clEnqueueNDRangeKernel(queue, find_r, (cl_uint) 1, 0, &kernel_paths, &local_kernel_paths, 0, NULL, kernel_event);
	assert(ret==CL_SUCCESS);
	
	cl_int *z = (cl_int*) malloc(sizeof(cl_int));
	cl_double *roc=(cl_double*) malloc(sizeof(cl_double));
	//**Reading the results**
	ret = clEnqueueReadBuffer(queue, result_buff, CL_TRUE, 0, sizeof(cl_int),z, 1, kernel_event, &read_events[0]);
	assert(ret==CL_SUCCESS);
	ret = clEnqueueReadBuffer(queue, float_result_buff, CL_TRUE, 0, sizeof(cl_double),roc, 1, &read_events[0],NULL );
	assert(ret==CL_SUCCESS);
	//roc[0] location records a boolean value telling whether the entering variable is increasing or decreasing 
	increase=(roc[0]<0);
	int r=z[0];
	free(z);
	return r;
}

/*
*This function enqueues the pre_update kernel which updates the tableau for any leaving variable that are not eventually 0
*and for any entering variable that are not initially 0, before pivoting
*/
void preUpdate(cl_kernel &pre_update, cl_context &context, cl_command_queue &queue,int k,int var,cl_double entering_mult,cl_double leaving_mult,int t_row)
{
	int ret;
	ret = clSetKernelArg(pre_update, 1, sizeof(cl_double), (void *)&entering_mult);
	assert(ret==CL_SUCCESS);
	
	ret = clSetKernelArg(pre_update, 2, sizeof(cl_double), (void *)&leaving_mult);
	assert(ret==CL_SUCCESS);
	
	ret = clSetKernelArg(pre_update, 3, sizeof(cl_int), (void *)&k);
	assert(ret==CL_SUCCESS);
	
	ret = clSetKernelArg(pre_update, 4, sizeof(cl_int), (void *)&var);
	assert(ret==CL_SUCCESS);
	
	int global_height=t_row-t_row%WORK_GROUP_SIZE;
	if (t_row%WORK_GROUP_SIZE!=0) global_height+=WORK_GROUP_SIZE;
	
	const size_t kernel_paths = global_height;
	const size_t local_kernel_paths = WORK_GROUP_SIZE;
	ret = clEnqueueNDRangeKernel(queue, pre_update, (cl_uint) 1, 0, &kernel_paths, &local_kernel_paths, 0, NULL, NULL);
	assert(ret==CL_SUCCESS);
}
