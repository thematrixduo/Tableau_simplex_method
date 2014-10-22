
//The struct that records the upper and lower bound for each variable
typedef struct tag_bound
{
	double upper;
	double lower;
} bound;

/*
*This is the kernel for pivoting
*/
__kernel
void pivot_2(
__global double *tableau,
double dividerInv,
int k,
int t_col,
int r,
int t_row,
__global double *row_r,
__global double *col_k
)
{	
	//Get the row and col number of the cell
	int row=get_global_id(0);
	int col=get_global_id(1);
	double dividerInversed=1/row_r[k];
	//fetch the kth_item of this row and the col item of row r
	double kth_item=col_k[row]; 
	double row_r_item=row_r[col];
	barrier(CLK_LOCAL_MEM_FENCE);
	//Pivoting operation according to tableau simplex method
	if ((row<t_row)&&(col<t_col))
		if (row!=r) 
			tableau[row*t_col+col]-=kth_item*row_r_item*dividerInversed;
		else 
			tableau[r*t_col+col]=row_r_item*dividerInversed;
	
}

/*
*This kernel takes the first row of the tableau and find the column k that gives 
*the maximum ck-zk for increasing cases or minimum ck-zk for decreasing cases
*/
__kernel
void find_max(
            	__global double* buffer,
            	__local double* scratch,
		__local int* index,
            	__const int length,
		__global int* boundary_active,
            	__global int* result
		) {

  int global_index = get_global_id(0);
  int local_index = get_local_id(0);
  // initializing the pointer array
  index[local_index]=local_index;
  // Load data into local memory	
  if ((global_index>0)&&(global_index < length-1)) {
		if	(boundary_active[global_index])
				scratch[local_index] = -buffer[global_index];
		else
				scratch[local_index] = buffer[global_index];
  } else {
    // Infinity is the identity element for the max operation
		scratch[local_index] = -INFINITY; 
  }  	
	//result[local_index]=scratch[local_index];
  barrier(CLK_LOCAL_MEM_FENCE);	
/*
*Comparing each element with the element at location of its location plus "offset"
*If the latter is larger, let the corresponding location of the pointer array points
*to the latter location, otherwise the pointer points to this location 
*/
  for(int offset = get_local_size(0)/2;offset > 0;offset >>= 1) 
	{
		if (local_index < offset) 
		{
			int this_index=index[local_index];
			int other_index=index[local_index + offset];
			double other = scratch[other_index];
			double mine = scratch[this_index]; 	
			index[local_index]=(mine>=other)? this_index:other_index;
		}	
    barrier(CLK_LOCAL_MEM_FENCE);
	}
   int index_result=index[0];
	//if (local_index!=0) result[local_index]=scratch[index[local_index-1]];
	if (local_index==0)
   		if (scratch[index_result]>0.0001)
			result[0]=index_result;
   		else
			result[0]=-1;
}


/*
*This kernel finds the r that gives the leaving variables
*The algorithm is the same as kernel find_max but the data loaded into local array 
*and the conditional statements are different, please refer to reports for further
*explanation
*/

__kernel
void find_r(
            __global double* buffer,
            __local double* scratch,
	    __local int* index,
	    __const int k,
	    __const int b_pos,
	    __const int t_col,
            __const int length,
            __global int* result,
	    __global int* boundary_active,
	    __global bound* boundary,
	    __global int* basic_var,
	    __const double ceiling,
	    __global double* float_result
	) {

  int global_index = get_global_id(0);
  int local_index = get_local_id(0);
  int factor;
  // Load data into local memory
  double k_value=buffer[global_index*t_col+k];
  	index[local_index]=local_index;
	
  if ((global_index>0)&&(global_index < length)&&((k_value>0.0001)||(k_value<-0.0001))) {
	int var=basic_var[global_index-1];
	factor=(boundary_active[var]==1)? -1:1;
	double boundary_item=(factor*k_value>0)? boundary[var].lower:boundary[var].upper;
	//double numerator=boundary_item;
	//result[local_index]=var;
    scratch[local_index] = factor*(buffer[global_index*t_col+b_pos]-boundary_item)/k_value;
  } else {
    // Infinity is the identity element for the min operation
    scratch[local_index] = INFINITY;
	//result[local_index] = 88888;
  }    
  barrier(CLK_LOCAL_MEM_FENCE);	
	
  for(int offset = get_local_size(0)/2;offset > 0;offset >>= 1) 
	{
		if (local_index < offset) 
		{
			int this_index=index[local_index];
			int other_index=index[local_index + offset];
			double other = scratch[other_index];
			double mine = scratch[this_index]; 
			index[local_index]=(other<mine)? other_index:this_index;
		}		
    barrier(CLK_LOCAL_MEM_FENCE);
  }
	result[0]=(scratch[index[0]]<ceiling)? index[0]:-1;
	if (local_index==index[0]) float_result[0]=factor*k_value;
}


/*
*This kernel duplicates the column k and row r so that they can be considered in pivot 2 kernel
*This is because column k and row r itself will be changed during the pivoting, in order to make sure
*all work items get consistent column k and row r value. Global synchronization before pivoting 
*is needed.
*/
__kernel
void copy_kr(
__global double* tableau,
__global double* row_r,
__global double* col_k,
int t_col,
int t_row,
int k,
int r
)
{
	int global_id=get_global_id(0);
	if (global_id<t_col)
		{
			row_r[global_id]=tableau[r*t_col+global_id];
			if (global_id<t_row)
				col_k[global_id]=tableau[global_id*t_col+k];
		}
}


/*
*This kernel pre-updates the tableau with entering variable not initialy at zero
*or leaving variable that does not go down to zero, refer to report or book for details.
*/
__kernel
void pre_update(
__global double*tableau,
double entering_mult,
double leaving_mult,
int entering_pos,
int leaving_pos,
int b_pos,
int t_col)
{
	int local_index=get_local_id(0);
	tableau[local_index*t_col+b_pos]+=entering_mult*tableau[local_index*t_col+entering_pos]-leaving_mult*tableau[local_index*t_col+leaving_pos];
}

