clear all
clc
A{1}=[4 7
   5 6
   7 5
   8 4
   9 2];

A{2}=[4 7
   5 6
   7 5
   8 4];

A{3}=[6 8
   7 7
   8 6
   9 5
   10 4];

%P=A4
A{4}=[1 3
   2 2
   3 1];

for i=1:4
    for j=1:4
        fprintf('A%d x A%d\n',i,j);
        eps_ind_mult(A{i},A{j})        
    end
end
