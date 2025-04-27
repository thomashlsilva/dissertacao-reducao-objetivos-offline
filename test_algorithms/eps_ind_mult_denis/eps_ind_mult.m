function s = eps_ind_mult(A, B)    
    clear aux
	for i=1:size(B,1)
        aux(i)=min(max(A./repmat(B(i,:),size(A,1),1),[],2));
    end
    s=max(aux);
end
