clear
close all
clc

x = [9893, 564, 17689; 
    8776, 389, 17359; 
    13572, 1103, 18597; 
    6455, 743, 8745;
 5129, 203, 14397; 
 5432, 215, 3467; 
 3807, 385, 4679; 
 3423, 187, 6754;
 3708, 127, 2275; 
 3294, 297, 6754; 5433, 432, 5589; 6287, 451, 8972];


% Calcula a matriz de correlação do DataFrame
corr_matrix = corr(x);
disp(corr_matrix);

% Cálculo dos Autovalores e Autovetores
[e_vectors, e_values, e_vectors_left] = eig(corr_matrix);

% Ordena os autovalores e autovetores em ordem decrescente
[~, idx] = sort(diag(e_values), 'descend');

e_values = e_values(idx, idx);
disp(e_values);

e_vectors = e_vectors(:, idx);
disp(e_vectors);

% Calcula a razão de variância explicada para cada componente principal
e_values_sum = sum(diag(e_values));
explained_variance_ratio = diag(e_values) / e_values_sum;
disp(explained_variance_ratio);

% Transpor a matriz de autovetores para manter consistência com Python
eigenvectors = e_vectors';

csvwrite('./pca_test_expvar_ratio.csv', explained_variance_ratio);
csvwrite('./pca_test_eigenvectors.csv', eigenvectors);