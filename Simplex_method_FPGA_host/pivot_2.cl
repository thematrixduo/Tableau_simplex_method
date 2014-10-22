#ifndef NUM_SIMD_ITEM
	#define NUM_SIMD_ITEM 4
#endif

#ifndef WORK_GROUP_SIZE
	#define WORK_GROUP_SIZE	256
#endif

typedef struct tag_bound
{
	double upper;
	double lower;
} bound;


//__attribute__((num_simd_work_items(4)))
__attribute__((reqd_work_group_size(1,WORK_GROUP_SIZE,0)))
__kernel
void pivot_2(
__global double * restrict tableau,
double dividerInv,
int k,
int t_col,
int r,
int t_row,
__global double *restrict row_r,
__global double *restrict col_k
)
{	
	//Get the row and col number of the cell
	int row=get_global_id(0);
	int col=get_global_id(1);
	//fetch the kth_item of this row and the col item of row r
	double kth_item=col_k[row]; 
	double row_r_item=row_r[col];
	barrier(CLK_LOCAL_MEM_FENCE);
	//Pivoting operation according to tableau simplex method
	if ((row<t_row)&&(col<t_col))
		if (row!=r) 
			tableau[row*t_col+col]-=kth_item*row_r_item*dividerInv;
		else 
			tableau[r*t_col+col]=row_r_item*dividerInv;
	
}


/*
*This kernel duplicates the column k and row r so that they can be considered in pivot 2 kernel
*This is because column k and row r itself will be changed during the pivoting, in order to make sure
*all work items get consistent column k and row r value. Global synchronization before pivoting 
*is needed.
*/
__kernel
void copy_kr(
__global double* restrict tableau,
__global double* restrict row_r,
__global double* restrict col_k,
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
__global double* tableau,
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
