function errorregion = check_negative(a)

[m, n]=size(a);
count = 0;
errorregion = [];
for i=1:m
    for j=1:n
        if a(i,j)<-1e-8
         count = count + 1;
         errorregion = [errorregion; i,j];
        end
    end
end

if count > 0
    count
    disp('Bound partially covers error surface');
%     check_M = chol(Mval)
else
    disp('Bound secure');
    mean_tightness = mean(mean(a))
    widest_gap = max(max(a))
    smallest_gap = min(min(a))
%     check_M = chol(Mval)
end
end