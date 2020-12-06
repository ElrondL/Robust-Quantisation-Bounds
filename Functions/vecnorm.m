function norm_vec = vecnorm(M)

    a = size(M);

    norm_vec = zeros(1, a(2));

    for i = 1:a(2)
        norm_vec(1, i) = norm(M(:,i));
    end
    
end