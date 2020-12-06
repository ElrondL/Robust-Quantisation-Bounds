function qmatrix = quantize_matrix_Binary(matrix, level)

size_matrix = size(matrix);
qmatrix = zeros(size_matrix(1), size_matrix(2));

for i = 1: size_matrix(1)
    for j = 1: size_matrix(2)
        qmatrix(i, j) = fix(matrix(i, j)*2^level)/(2^level);
    end
end

end
