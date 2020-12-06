function compNR = complexity(nx, nf, n,l)

compNR = 2*nx*n(1);
for i = 1:l-1
    compNR = compNR + 2*n(i)*n(i+1);
end
compNR = compNR + 2*n(l)*nf;
end

