function qcell = quantize_cell_Binary(inputcell, level)

size_cell = size(inputcell);
qcell = cell(size_cell(1), size_cell(2));

for i = 1: size_cell(1)
    for j = 1: size_cell(2)
        matrix = inputcell{i, j};
        size_matrix = size(matrix);
        qmatrix = zeros(size_matrix(1), size_matrix(2));
            for k = 1:size_matrix(1)
                for l = 1:size_matrix(2)
                    qmatrix(k, l) = fix(matrix(k, l)*2^level)/(2^level);
                end
            end
        qcell{i, j} = qmatrix;
    end
end

end
