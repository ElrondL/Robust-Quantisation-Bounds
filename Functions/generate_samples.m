function samp_space = generate_samples(x_up, x_lower, Nsamples, n_x)

count = 0;
samp_space = [];

while count < n_x
%     a = x_lower + (x_up-x_lower).*rand(1, Nsamples);
%     samp_space = [samp_space; sort(a)];
    samp_space = [samp_space; linspace(x_lower, x_up, Nsamples)];
    count = count + 1;
end