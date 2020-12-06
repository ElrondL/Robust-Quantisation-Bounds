function phi = ReLU(y)

sizey = size(y);

phi= zeros(sizey(1,1),sizey(1,2));
for i = 1:max(size(y))
    if y(i) >= 0
        phi(i) = y(i);
    else
        phi(i) = 0;
    end
end

end