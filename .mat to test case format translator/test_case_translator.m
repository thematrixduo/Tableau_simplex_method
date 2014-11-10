A_full=full(A);
infinity=1.00000000000000e+100;
num_row=9074;
num_var=3439;
num_constraint=0;
for i=1:num_row
    if (LHS(i)>(-infinity))
        num_constraint=num_constraint+1;
    end
    if (RHS(i)<infinity)
        num_constraint=num_constraint+1;
    end
end
k=1;
num_constraint
problem_file = fopen('problem_x.txt','W');
fprintf(problem_file,'%d %d %d\n',num_var,num_constraint,num_var);
fclose(problem_file);
c_t=c.';
dlmwrite('problem_x.txt',c_t,'-append','delimiter',' ');
T=zeros(num_constraint,num_var+2);
for i=1:num_row
    if (LHS(i)>(-infinity))
        T(k,1:num_var)=A_full(i,1:num_var);
        T(k,num_var+1)=LHS(i);
        T(k,num_var+2)=1;
        k=k+1;
    end
    if (RHS(i)<infinity)
        T(k,1:num_var)=A_full(i,1:num_var);
        T(k,num_var+1)=RHS(i);
        T(k,num_var+2)=0; 
        k=k+1;
    end
    i
end
dlmwrite('problem_x.txt',T,'-append','delimiter',' ');
bound=zeros(num_var,3);
bound(:,2:3)=boundary(:,1:2);
i=1:1:num_var;
bound(i,1)=i;
dlmwrite('problem_x.txt',bound,'-append','delimiter',' ');