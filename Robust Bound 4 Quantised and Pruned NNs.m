clear; close all; clc;
%%ANN model: MLP

% Dec 06 2020
% Jiaqi Li, Ross Drummond
% Department of Engineering, University of Oxford

% ***** if your MATLAB has vecnorm function, simply disable the vecnorm function script

% Compare NN mode: set qc for quantization multiplier to zero in M, set input to be
% different random vectors, set two random NNs

% Self-compare NN mode: set qc for quantization multiplier to zero in M, set input to be
% different random vectors, set one random NN and set the second NN to equal the first
% **Activate line 46 and 47, Deactivate line 44

% Quantization Comparison mode: set one random NN and set the second NN to
% equal the first, set second input to be quantized version of the first,
% set qc for quantization multiplier to one in M
% **Activate line 47 and 49
% **Set line 35 quantization_mode = 1

% Prune mode: run prune_Universal.m on a saved set of NN1 weights and save it as NN2's weights
% load all weight and biases parameters for NN1, pruned weight parameters for NN2, and let the biases = NN1's biases
% **Activate line 47, 50 and 51


%% Set up part
n = [10,10,10,10]; % number of neurons per hidden layer. 
l = 5; % number of weights and biases sets (number of transitions)
n_x = 1; %dimension of the input data.
nf = 1; %dimension of the output
compNR = complexity(n_x, nf, n, l-1) %number of computations
q_level = 3; %quantize to 0.001
quantization_mode = 0; %set to 1 when on quantization mode, set back to zero for comparison mode

N = sum(n); %total number of nonlinearities 
Nin = n_x + sum(n(1:l-1)); % dimensions of the "arguements" of the activation functions.
Nout = sum(n(1:l-1));
N_out = Nout;

number= 2*n_x+2*N_out+1; %Size of the zeta matrix. Contains both x_1 and x_2, all the nonlinearities of both neural networks and a constant term.

[W_1,Wc_1,b_1,bc_1,Wl_1,bl_1,Wx_1,bx_1] = compute_weights(n,l,n_x,nf); % Generate random weights for the first neural network.
[W_2,Wc_2,b_2,bc_2,Wl_2,bl_2,Wx_2,bx_2] = compute_weights(n,l,n_x,nf); % Generate random weights for the second neural network.

%load('W_save_2BPruned.mat'); % SAVED AN INTERESTING SET OF WEIGHTS AND BIASES
%W_2 = W_1;b_2 =b_1;Wc_2 = Wc_1;bc_2 = bc_1;Wl_2 = Wl_1;bl_2 = bl_1;Wx_2= Wx_1;bx_2 = bx_1;
%W_2 = quantize_cell_Binary(W_1, q_level); Wc_2 = quantize_matrix_Binary(Wc_1, q_level);b_2 =quantize_cell_Binary(b_1, q_level); bc_2 = quantize_matrix_Binary(bc_1, q_level);Wl_2 = quantize_matrix_Binary(Wl_1, q_level);bl_2 = quantize_matrix_Binary(bl_1, q_level);Wx_2= quantize_matrix_Binary(Wx_1, q_level);bx_2 = quantize_matrix_Binary(bx_1, q_level);
%b_2 =b_1;bc_2 = bc_1;bl_2 = bl_1;bx_2 = bx_1;%pruning mode
%load('W_save_pruned.mat')



% savefile = 'W_save_1.mat';
% save('W_save.mat','W_1','Wc_1','b_1','bc_1','Wl_1','bl_1','Wx_1','bx_1');
% savefile = 'W_save_2.mat';
% save('W_save.mat','W_2','Wc_2','b_2','bc_2','Wl_2','bl_2','Wx_2','bx_2');

% b1_1 = 0; b1_2 = 0; %Set the output biases to zero.
% bl_1 = 0; bl_2 = 0; %Set the output biases to zero.

%% ANN part
Nsamples = 1e2 + quantization_mode*1; % number of samples of the input space
x_up = 1e0; %upper limit for the input x, x< x_up
x_lower = -1e0;
samp_space_1 = generate_samples(x_up, x_lower, Nsamples, n_x);
samp_space_2 = generate_samples(x_up, x_lower, Nsamples, n_x);
%samp_space_1 = quantize_matrix_Binary(samp_space_1,q_level);
%samp_space_2 = quantize_matrix_Binary(samp_space_1,q_level);
%samp_space_2 = quantize_matrix_Binary(samp_space_2,q_level);
f_1 = zeros(nf, Nsamples);
f_2 = zeros(nf, Nsamples);

for i = 1:Nsamples % produces f_1(x) for various values of x_1
    x1 = samp_space_1(:,i);
    x2 = samp_space_2(:,i);
    y1 = Wx_1*x1+bx_1;
    y2 = Wx_2*x2+bx_2;
    %%Modify this equation to change script to work for other NNs e.g. RNN
    for j =2:l-1
        phi1 = ReLU(y1);
        phi2 = ReLU(y2);
        %phi2 = ReLU(quantize_matrix_Binary(y2,q_level));
        y1 = W_1{1,j}*phi1+b_1{1,j};
        y2 = W_2{1,j}*phi2+b_2{1,j};
        %         y1 = W_1{1,2}*phi1+b_1{1,2};
        %         y2 = W_2{1,2}*phi2+b_2{1,2};
    end
    phi1 = ReLU(y1);
    phi2 = ReLU(y2);
    %phi2 = ReLU(quantize_matrix_Binary(y2,q_level));
    f_1(:,i) = Wl_1*phi1+bl_1;
    f_2(:,i) = Wl_2*phi2+bl_2;
    %f_2(:,i) = quantize_matrix_Binary(Wl_2*phi2+bl_2,q_level);
end

%% Modelling part
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Condition part
%x1_mat = blkdiag(eye(n_x),zeros(n_x,n_x),zeros(N_out),zeros(N_out),0); % Defines the input of the first NN being x_1
x1_mat = [eye(n_x),zeros(n_x,n_x),zeros(n_x,N_out),zeros(n_x,N_out),zeros(n_x,1)]; % Defines the input of the first NN being x_1
q_x1_mat = x1_mat; %quantization of x_1 (notation difference), if x2 = q(x1) use this to set the condition
%x2_mat = blkdiag(zeros(n_x,n_x),eye(n_x),zeros(N_out),zeros(N_out),0); % Defines the input of the second NN being x_2
x2_mat = [zeros(n_x,n_x),eye(n_x),zeros(n_x,N_out),zeros(n_x,N_out),zeros(n_x,1)]; % Defines the input of the second NN being x_2
q_x2_mat = x2_mat; %quantization of x_2 (notation difference)
one_x2_mat = [zeros(n_x,n_x),eye(n_x),zeros(n_x,N_out),zeros(n_x,N_out),zeros(n_x,1)];

xi1_mat = [Wx_1,zeros(n(1),n_x), zeros(n(1),2*N_out),bx_1; zeros(N_out-n(1),2*n_x), Wc_1, zeros(N_out-n(1),n(l-1)), zeros(N_out-n(1),N_out), bc_1]; % Defines the arguement of the first NN being \xi_1
xi2_mat = [zeros(n(1),n_x),Wx_2, zeros(n(1),2*N_out),bx_2; zeros(N_out-n(1),2*n_x), zeros(N_out-n(1),N_out), Wc_2, zeros(N_out-n(1),n(l-1)),bc_2]; % Defines the arguement of the second NN being \xi_2
one_xi2_mat = [zeros(n(1),n_x),ones(size(Wx_2)), zeros(n(1),2*N_out),ones(size(bx_2)); zeros(N_out-n(1),2*n_x), zeros(N_out-n(1),N_out), ones(size(Wc_2)), zeros(N_out-n(1),n(l-1)),ones(size(bc_2))]; % Defines the arguement of the second NN being \xi_2

phi1_mat = [zeros(N_out,2*n_x),eye(N_out),zeros(N_out,N_out), zeros(N_out,1)]; % Defines the nonlinearities of the first NN
phi2_mat = [zeros(N_out,2*n_x),zeros(N_out,N_out),eye(N_out), zeros(N_out,1)]; % Defines the nonlinearities of the second NN

one_mat = [zeros(N_out,2*n_x+2*N_out),ones(N_out,1)]; % Defines the vector of ones.
one_mat_x = [zeros(n_x,2*n_x+2*N_out),ones(n_x,1)]; % Defines another vector of ones.

f_1_mat = [zeros(nf,2*n_x + N_out-n(l-1)),Wl_1, zeros(nf,N_out),bl_1]; % Defines the output of the first NN
f_2_mat = [zeros(nf,2*n_x + 2*N_out-n(l-1)),Wl_2 ,bl_2]; % Defines the output of the second NN

%generate the lambdas
tau_slope = sdpvar(N_out,N_out);
tau_slope_1 = sdpvar(N_out,N_out);
tau_slope_2 = sdpvar(N_out,N_out); %Slope restricted
tau_pos_1 = sdpvar(N_out,N_out);
tau_pos_2 = sdpvar(N_out,N_out); %Positive
tau_cpos_1 = sdpvar(N_out,N_out);
tau_cpos_2 = sdpvar(N_out,N_out); %Positive Complement
tau_comp_1 = sdpvar(N_out,N_out,'diagonal');
tau_comp_2 = sdpvar(N_out,N_out,'diagonal'); %Complementary condition
tau_crx = sdpvar(N_out,N_out,'diagonal');
tau_crx_1 = sdpvar(N_out,N_out,'diagonal');
tau_crx_2 = sdpvar(N_out,N_out,'diagonal'); %Cross terms i
tau_crx_phi = sdpvar(N_out,N_out,'diagonal');
tau_crx_phi_1 = sdpvar(N_out,N_out,'diagonal');
tau_crx_phi_2 = sdpvar(N_out,N_out,'diagonal'); %Cross terms ii
tau_xi_2_quantize = sdpvar(N_out,N_out,'diagonal');
tau_phi_2_quantize = sdpvar(N_out,N_out,'diagonal');
tau_f_2_quantize = sdpvar(nf,nf,'diagonal');

tau_x_1 = sdpvar(n_x,n_x,'diagonal');
tau_x_2 = sdpvar(n_x,n_x,'diagonal');
tau_x_2_quantize = sdpvar(n_x,n_x,'diagonal');
tau_x_j = sdpvar(n_x,n_x,'diagonal');
tau_x_j_m = sdpvar(n_x,n_x,'diagonal');


gamma = sdpvar;
gamma_affine = sdpvar;

%Conditions

slope = zeros(2*n_x+2*N_out+1,2*n_x+2*N_out+1);
slope_1 = zeros(2*n_x+2*N_out+1,2*n_x+2*N_out+1);
slope_2 = zeros(2*n_x+2*N_out+1,2*n_x+2*N_out+1); %Initialise the slope conditions matrix
for i = 1:N_out
    for j = 1:N_out
        phi_diff = phi2_mat(i,:)-phi1_mat(j,:);xi_diff = xi2_mat(i,:)-xi1_mat(j,:);
        slope = slope  + 1*(xi_diff-phi_diff)'*tau_slope(i,j)*phi_diff;
        %joint slope conditions for the first and second  NN
        
        phi_diff_1 = phi1_mat(i,:)-phi1_mat(j,:);y_diff_1 = xi1_mat(i,:)-xi1_mat(j,:);
        slope_1 = slope_1  + (y_diff_1-phi_diff_1)'*tau_slope_1(i,j)*phi_diff_1;
        %slope conditions for the first NN
        
        phi_diff_2 = phi2_mat(i,:)-phi2_mat(j,:);y_diff_2 = xi2_mat(i,:)-xi2_mat(j,:);
        slope_2 = slope_2  + (y_diff_2-phi_diff_2)'*tau_slope_2(i,j)*phi_diff_2; 
        %slope conditions for the second NN
    end
end

pos_1 =  one_mat'*tau_pos_1'*phi1_mat; % Multiplied this by 1.
pos_2 =  one_mat'*tau_pos_2'*phi2_mat;  % Multiplied this by 1.

cpos_1 = one_mat'*tau_cpos_1'*(phi1_mat - xi1_mat); % Multiplied this by 1.
cpos_2 = one_mat'*tau_cpos_2'*(phi2_mat - xi2_mat); % Multiplied this by 1.


comp_1 = (phi1_mat - xi1_mat)'*tau_comp_1*phi1_mat; % Complementary condition for the 1st NN
comp_2 = (phi2_mat - xi2_mat)'*tau_comp_2*phi2_mat; % Complementary condition for the 2nd NN
crx_1i = phi1_mat'*tau_crx_1*(phi1_mat - xi1_mat);
crx_2i = phi2_mat'*tau_crx_2*(phi2_mat - xi2_mat);
crx12 =  phi1_mat'*tau_crx*(phi2_mat - xi2_mat);
crx21 =  phi2_mat'*tau_crx*(phi1_mat - xi1_mat);
crx_1ii = phi1_mat'*tau_crx_phi_1*phi1_mat;
crx_2ii = phi2_mat'*tau_crx_phi_2*phi2_mat;
crx_phi12 = phi1_mat'*tau_crx_phi*phi2_mat;
crx_phi21 = phi2_mat'*tau_crx_phi*phi1_mat;



%Input constraints
x_upper = x_up;%x_lower = -x_upper; % set the upper and lower bounds for the inputs x_1 and x_2
c = 0;
for i = 1:q_level
    c = c + 2^-i; % set quantization constant
end
c = 2^-q_level;
xcon_1 = (x_upper*one_mat_x-x1_mat)'*tau_x_1*(x1_mat-x_lower*one_mat_x); %upper and lower bounds for x_1 of the first NN
xcon_2 = (x_upper*one_mat_x-x2_mat)'*tau_x_2*(x2_mat-x_lower*one_mat_x); %upper and lower bounds for x_2 of the second NN

one_mat_q = blkdiag(zeros(Nout-1,2*n_x+2*Nout),1);
xquantize_2 = (x2_mat - (x1_mat-one_mat_x*c))'*tau_x_2_quantize*((x1_mat+one_mat_x*c) - x2_mat); %quantization bound for x_2 of the second NN
xiquantize_2 = (xi2_mat - (xi1_mat-one_mat*c))'*tau_xi_2_quantize*((xi1_mat+one_mat*c) - xi2_mat); %quantization bound for xi_2 of the second NN
phiquantize_2 = (xi2_mat - (phi2_mat- c))'*tau_phi_2_quantize*((phi2_mat+c) - xi2_mat); %quantization bound for xi_2 of the second NN
outputquantize_2 = (f_2_mat - (f_1_mat-one_mat_x*c))'*tau_f_2_quantize*((f_1_mat+one_mat_x*c) - f_2_mat); %quantization bound for xi_2 of the second NN

% xquantize_2 = zeros(2*n_x+2*N_out+1,2*n_x+2*N_out+1); %Initialise the quantization condition matrix
% for i = 1:N_out
%     for j = 1:N_out
%         xquantize_2 = (x2_mat(i,j) - (x1_mat(i,j)-one_mat_x(1,j)*c))*tau_x_2_quantize*((x1_mat(j,i)+one_mat_x(1,i)*c) - x2_mat(j,i)); %quantization bound for x_2 of the second NN
%     end
% end


xcon_joint = (2*x_upper*one_mat_x-(x2_mat+x1_mat))'*tau_x_j*((x2_mat+x1_mat)-2*x_lower*one_mat_x); % A joint constraint between the inputs of the two NNs
xcon_joint_min = ((x_upper-x_lower)*one_mat_x-(x2_mat-x1_mat))'*tau_x_j_m*((x2_mat-x1_mat)-(x_lower-x_upper)*one_mat_x); % Another joint constraint between the inputs of the two NNs

%Output constraints
x1_m_x2 = x2_mat-x1_mat; %Difference between the first and second inputs x_1-x_2
f1_m_f2 = f_2_mat-f_1_mat; % what we care about is the difference between the outputs of the first and second NNs f_1(x_1)-f_2(x_2). Doesn't matter if it the difference of the second and first.

One_mat = blkdiag(zeros(2*n_x+2*Nout,2*n_x+2*Nout),1);
output = f1_m_f2'*f1_m_f2-gamma*(x1_m_x2'*x1_m_x2)-1*gamma_affine*One_mat;
output_q = f1_m_f2'*f1_m_f2-gamma*(x1_mat'*x1_mat)-1*gamma_affine*One_mat;


%This bit basically makes all the matrices deifned above symmettric.
% sec_1  = 0.5*(sec_1+sec_1');
% sec_2  = 0.5*(sec_2+sec_2');
slope  = 0.5*(slope+slope');
slope_1  = 0.5*(slope_1+slope_1');
slope_2  = 0.5*(slope_2+slope_2');
pos_1 = 0.5*(pos_1'+pos_1); % THIS WAS MULTIPLIED, MADE IT A BMI (AS I TWO MATRIX VARIABLES MULTIPLIER TOGETHER). You want the matrix to always be LINEAR in their variables.
pos_2 = 0.5*(pos_2'+pos_2);  % THIS WAS MULTIPLIED, MADE IT A BMI (AS I TWO MATRIX VARIABLES MULTIPLIER TOGETHER). You want the matrix to always be LINEAR in their variables.
cpos_1 = 0.5*(cpos_1'+cpos_1);  % THIS WAS MULTIPLIED, MADE IT A BMI (AS I TWO MATRIX VARIABLES MULTIPLIER TOGETHER). You want the matrix to always be LINEAR in their variables.
cpos_2 = 0.5*(cpos_2'+cpos_2);  % THIS WAS MULTIPLIED, MADE IT A BMI (AS I TWO MATRIX VARIABLES MULTIPLIER TOGETHER). You want the matrix to always be LINEAR in their variables.
crx12 = 0.5*(crx12+crx12');
crx21 = 0.5*(crx21+crx21');
crx_phi12 = 0.5*(crx_phi12+crx_phi12');
crx_phi21 = 0.5*(crx_phi21+crx_phi21');
crx_1i = 0.5*(crx_1i+crx_1i');
crx_1ii = 0.5*(crx_1ii+crx_1ii');
crx_2i = 0.5*(crx_2i+crx_2i');
crx_2ii = 0.5*(crx_2ii+crx_2ii');
comp_1 = 0.5*(comp_1+comp_1');
comp_2 = 0.5*(comp_2+comp_2');
xiquantize_2 = 0.5*(xiquantize_2+xiquantize_2');
phiquantize_2 = 0.5*(phiquantize_2+phiquantize_2');
outputquantize_2 = 0.5*(outputquantize_2+outputquantize_2');

xcon_1  = 0.5*(xcon_1+xcon_1');
xcon_2  = 0.5*(xcon_2+xcon_2');
xquantize_2 = 0.5*(xquantize_2+xquantize_2');
xcon_joint  = 0.5*(xcon_joint+xcon_joint');
xcon_joint_min  = 0.5*(xcon_joint_min+xcon_joint_min');
output  = 0.5*(output+output');
output_q  = 0.5*(output_q+output_q');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%SDP LMI part
x_constraints= xcon_1+xcon_2+ 1*xcon_joint +1*xcon_joint_min + quantization_mode*(xquantize_2 + 0*xiquantize_2 + 0*phiquantize_2 + 0*outputquantize_2);
sector_slope = slope+slope_1+slope_2;
positivity = pos_1+pos_2;
comp_positivity = cpos_1+cpos_2;
cross_terms = 1*crx_1i+1*crx_1ii+1*crx_2i+1*crx_2ii+0*crx12+0*crx21+0*crx_phi12+0*crx_phi21;
complimentarity = comp_1+comp_2;

if quantization_mode == 0

    M = output + sector_slope + x_constraints+1*positivity+1*comp_positivity+1*cross_terms+1*complimentarity;

else
    
    M = output_q + sector_slope + x_constraints+1*positivity+1*comp_positivity+1*cross_terms+1*complimentarity;
    
end

%%The matrix that constains all of the inequalities above. This is the main guy.

%
tol = 1e-8;
%tol = 0;
%tol = 0.5; %tolerance of the solver.
%F = [];

F = [-M >= tol*eye(number)]; % Includes all the positive constraints. This requires that matrix M is positive definite.
%F = [F, -output >= tol*eye(number)];

tol1 = tol;

F = [F,  tau_crx_phi >= tol1];
F = [F,  tau_crx_phi_1 >= tol1];
F = [F,  tau_crx_phi_2 >= tol1];
F = [F,  tau_crx >=tol1];
F = [F,  tau_crx_1 >=tol1];
F = [F,  tau_crx_2 >=tol1];

F = [F,  tau_slope(:) >=tol1]; %This is just a way to make that a matrix full of positive elements
F = [F,  tau_slope_1(:) >=tol1];
F = [F,  tau_slope_2(:) >=tol1];
F = [F,  tau_pos_1(:) >=tol1];
F = [F,  tau_pos_2(:) >=tol1];
F = [F,  tau_cpos_1(:) >=tol1];
F = [F,  tau_cpos_2(:) >=tol1];

F = [F, tau_x_1 >=tol1 ];
F = [F, tau_x_2 >=tol1 ];
F = [F, tau_x_2_quantize >=tol1 ];
F = [F, tau_xi_2_quantize >=tol1 ];
F = [F, tau_phi_2_quantize >=tol1 ];
% F = [F, tau_f_2_quantize >=tol1 ];
F = [F, tau_x_j >=tol1 ];
F = [F, tau_x_j_m >=tol1 ];

tol2 = 1e-8;
% tol2 = 1e1;


F = [F, gamma >= tol2, ];
F = [F, gamma_affine >= tol2]; %We want our bounds to be positive.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%solution generation
w_x = 1e0; w_const = 1e0; %weights for the objective function
obj = w_x*gamma+ w_const*gamma_affine; %the objective function.
%obj = 1*norm(output, Inf)+ w_x*gamma + w_const*gamma_affine; %the output cost function.
% obj = []; %no objective function.
sol2 = optimize(F,obj,sdpsettings('solver','mosek')); %run the optimisation
sol = sol2.problem
Mval = value(M);
gamma_val = value(gamma) %this converts the sdp variable into a real one
gamma_affine_val = value(gamma_affine) %this converts the sdp variable into a real one


% Graphing part
close all;
f_size = 20;

e_f = (f_1 - f_2).^2;
e_con = gamma_val*(samp_space_1-samp_space_2).^2 + gamma_affine_val;
samp_space_index = [1:Nsamples];


if quantization_mode == 1
    
    samp_space_index = [1:Nsamples];
    error = zeros(1,Nsamples); bound = zeros(1,Nsamples);
    for i=1:Nsamples
            error(i) = (f_1(i)-f_2(i)).^2;
            bound(i) = gamma_val*(samp_space_1(i)).^2+1*gamma_affine_val; %+ (l-1)*compNR*(c/2);
            %bound(i) = gamma_val*(samp_space_1(i)-samp_space_2(i)).^2+gamma_affine_val;
    end
    
    figure;
    semilogy(samp_space_1, error, 'k', 'linewidth', 2); hold on; 
    semilogy(samp_space_1, bound, 'r', 'linewidth', 2); hold on;
    semilogy(samp_space_2, 10^-6, '+k', 'linewidth', 2); hold on;
    xlabel('x','interpreter','latex','fontsize',f_size)
    %ylabel('value','interpreter','latex','fontsize',f_size)
    leg = legend('ln($(f_1(x_1)-f_2(x_2))^2)$','ln$(\gamma_x (x_1)^2 + \gamma)$');
    set(leg,'fontsize',f_size,'interpreter','latex','location','best')
    grid on; 
    difference = error - bound;
    
 
    
else
    error = zeros(Nsamples,Nsamples); bound = zeros(Nsamples,Nsamples);

    if n_x == 1 && nf == 1

        for i=1:Nsamples
            for j = 1:Nsamples
                error(i,j) = (f_1(i)-f_2(j)).^2;
                bound(i,j) = gamma_val*(samp_space_1(i)-samp_space_2(j)).^2+gamma_affine_val;
                bound(i,j) = gamma_val*(samp_space_1(i)).^2+gamma_affine_val;
            end
        end


        difference = error - bound;

        [X,Y] = meshgrid(samp_space_1,samp_space_2);

        fig_surf = figure;
        s2 = surf(X,Y,log(error),'FaceAlpha',0.7); hold on;
        s = surf(X,Y,log(bound),'FaceAlpha',0.5,'FaceColor','black'); hold on;
        % s3 = surf(X,Y,log(bound)-log(error),'FaceAlpha',0.5,'FaceColor','magenta'); hold on;
        xlabel('$x_1$','interpreter','latex','fontsize',f_size)
        ylabel('$x_2$','interpreter','latex','fontsize',f_size)
        grid on
        leg = legend('ln($(f_1(x_1)-f_2(x_2))^2)$','ln$(\gamma_x (x_1-x_2)^2 + \gamma)$');
        set(leg,'fontsize',f_size,'interpreter','latex','location','best')
        box
        s.EdgeColor = 'none'; s2.EdgeColor = 'none';

        fig_surf_error = figure;
        s3 = surf(X,Y,log(error./bound),'FaceAlpha',1,'FaceColor','magenta'); hold on;
        xlabel('$x_1$','interpreter','latex','fontsize',f_size)
        ylabel('$x_2$','interpreter','latex','fontsize',f_size)
        grid on
        leg = legend('ln\Bigg($\frac{(f_1(x_1)-f_2(x_2))^2}{\gamma_x (x_1-x_2)^2 + \gamma}$\Bigg)');
        set(leg,'fontsize',f_size,'interpreter','latex','location','best')
        box
        s3.EdgeColor = 'cyan';

        fig_all = figure;
        s = surf(X,Y,log(bound),'FaceAlpha',0.5,'FaceColor','black'); hold on;
        s2 = surf(X,Y,log(error)); hold on;
        s3 = surf(X,Y,log(error./bound),'FaceAlpha',1,'FaceColor','magenta'); hold on;
        xlabel('$x_1$','interpreter','latex','fontsize',f_size)
        ylabel('$x_2$','interpreter','latex','fontsize',f_size)
        grid on
        leg = legend('ln($(f_1(x_1)-f_2(x_2))^2)$','ln$(\gamma_x (x_1-x_2)^2 + \gamma)$', 'ln\Bigg($\frac{(f(x_1)-f(x_2))^2}{\gamma (x_1-x_2)^2 + \gamma_{affine}}$\Bigg)');
        set(leg,'fontsize',f_size,'interpreter','latex','location','best')
        box
        s.EdgeColor = 'none'; s2.EdgeColor = 'none'; s3.EdgeColor = 'cyan';

    else

        for i=1:Nsamples
            for j = 1:Nsamples
                error(i,j) = norm(f_1(:,i)-f_2(:,j));
                bound(i,j) = gamma_val*(norm(samp_space_1(:,i)-samp_space_2(:,j)))+gamma_affine_val;
            end
        end

        difference = error - bound;

        norm_samp_space_1 = vecnorm(samp_space_1);
        norm_samp_space_2 = vecnorm(samp_space_2); %If you are using MATLAB version > 2019, disable vecnorm.m in your working directory

        [X,Y] = meshgrid(norm_samp_space_1,norm_samp_space_2);

        fig_surf = figure;
        s2 = surf(X,Y,log(error)); hold on;
        s = surf(X,Y,log(bound),'FaceAlpha',0.5,'FaceColor','black'); hold on;
        % s3 = surf(X,Y,log(bound)-log(error),'FaceAlpha',0.5,'FaceColor','magenta'); hold on;
        xlabel('$norm(x_1)$','interpreter','latex','fontsize',f_size)
        ylabel('$norm(x_2)$','interpreter','latex','fontsize',f_size)
        grid on
        leg = legend('ln($(f(x_1)-f(x_2))^2)$','ln$(\gamma (x_1-x_2)^2 + \gamma_{affine})$');
        set(leg,'fontsize',f_size,'interpreter','latex','location','best')
        box
        s.EdgeColor = 'none'; s2.EdgeColor = 'none';

        fig_surf_error = figure;
        s3 = surf(X,Y,log(error./bound),'FaceAlpha',1,'FaceColor','magenta'); hold on;
        xlabel('$norm(x_1)$','interpreter','latex','fontsize',f_size)
        ylabel('$norm(x_2)$','interpreter','latex','fontsize',f_size)
        grid on
        leg = legend('ln\Bigg($\frac{(f(x_1)-f(x_2))^2}{\gamma (x_1-x_2)^2 + \gamma_{affine}}$\Bigg)');
        set(leg,'fontsize',f_size,'interpreter','latex','location','best')
        box
        s3.EdgeColor = 'cyan';

        fig_all = figure;
        s = surf(X,Y,log(bound),'FaceAlpha',0.5,'FaceColor','black'); hold on;
        s2 = surf(X,Y,log(error)); hold on;
        s3 = surf(X,Y,log(error./bound),'FaceAlpha',1,'FaceColor','magenta'); hold on;
        xlabel('$norm(x_1)$','interpreter','latex','fontsize',f_size)
        ylabel('$norm(x_2)$','interpreter','latex','fontsize',f_size)
        grid on
        leg = legend('ln($(f(x_1)-f(x_2))^2)$','ln$(\gamma (x_1-x_2)^2 + \gamma_{affine})$', 'ln\Bigg($\frac{(f(x_1)-f(x_2))^2}{\gamma (x_1-x_2)^2 + \gamma_{affine}}$\Bigg)');
        set(leg,'fontsize',f_size,'interpreter','latex','location','best')
        box
        s.EdgeColor = 'none'; s2.EdgeColor = 'none'; s3.EdgeColor = 'cyan';

    end
    
end


error_region = check_negative((log(bound)-log(error)));







%% Check Error Part
slope_val = value(slope);
slope_1_val = value(slope_1);
slope_2_val = value(slope_2);
pos_1_val = value(pos_1);
pos_2_val = value(pos_2);
cpos_1_val = value(cpos_1);
cpos_2_val = value(cpos_2);

crx12_val = value(crx12);
crx21_val = value(crx21);
crx_phi12_val = value(crx_phi12);
crx_phi21_val = value(crx_phi21);
crx_1i_val = value(crx_1i);
crx_1ii_val = value(crx_1ii);
crx_2i_val = value(crx_2i);

crx_2ii_val =value(crx_2ii);
comp_1_val = value(comp_1);
comp_2_val = value(comp_2 );
xcon_1_val = value(xcon_1);
xcon_2_val = value(xcon_2);
xquantize_2_val = value(xquantize_2);
xiquantize_2_val = value(xiquantize_2);
xcon_joint_val = value(xcon_joint );
output_val = value(output);

for i = 1:Nsamples % produces f_1(x) for various values of x_1
    for g =1:Nsamples
        
        zetacolumn = zeros(2*n_x+2*Nout+1,1);
        x1 = samp_space_1(i);
        x2 = samp_space_2(g);
        
        zetacolumn = [x1;x2];
        
        y1 = Wx_1*x1+bx_1;
        %%Modify this equation to change script to work for other NNs e.g. RNN
        for j =2:l-1
            phi1 = ReLU(y1);
            zetacolumn = [zetacolumn; phi1];
            y1 = W_1{1,j}*phi1+b_1{1,j};
        end
        phi1 = ReLU(y1);
        zetacolumn = [zetacolumn; phi1];
        %     f_1(i) = Wl_1*phi1+bl_1;
        
        %     zetacolumn = [zetacolumn; x2];
        y2 = Wx_2*x2+bx_2;
        %%Modify this equation to change script to work for other NNs e.g.
        %     RNN
        for j =2:l-1
            phi2 = ReLU(y2);
            zetacolumn = [zetacolumn; phi2];
            y2 = W_2{1,j}*phi2+b_2{1,j};
        end
        phi2 = ReLU(y2);
        zetacolumn = [zetacolumn; phi2];
        %     f_2(i) = Wl_2*phi2+bl_2;
        zetacolumn = [zetacolumn; 1];
        
        %%
        check_slope(i,g) = zetacolumn'*(slope_val)*zetacolumn;
        check_slope_1(i,g) = zetacolumn'*(slope_1_val)*zetacolumn;
        check_slope_2(i,g) = zetacolumn'*(slope_2_val)*zetacolumn;
        check_pos_1(i,g) = zetacolumn'*(pos_1_val)*zetacolumn;
        check_pos_2(i,g) = zetacolumn'*(pos_2_val)*zetacolumn;
        check_cpos_1(i,g) = zetacolumn'*(cpos_1_val)*zetacolumn;
        check_cpos_2(i,g) = zetacolumn'*(cpos_2_val)*zetacolumn;
        
        check_crx12(i,g) = zetacolumn'*(crx12_val)*zetacolumn;
        check_crx21_1(i,g) = zetacolumn'*(crx21_val)*zetacolumn;
        check_crx_phi12(i,g) = zetacolumn'*(crx_phi12_val)*zetacolumn;
        check_crx_phi21(i,g) = zetacolumn'*(crx_phi21_val)*zetacolumn;
        check_crx_1i(i,g) = zetacolumn'*(crx_1i_val)*zetacolumn;
        check_crx_1ii(i,g) = zetacolumn'*(crx_1ii_val)*zetacolumn;
        check_crx_2i(i,g) = zetacolumn'*(crx_2i_val)*zetacolumn;
        
        check_crx_2ii(i,g) = zetacolumn'*(crx_2ii_val)*zetacolumn;
        check_comp_1(i,g) = zetacolumn'*(comp_1_val)*zetacolumn;
        check_comp_2(i,g) = zetacolumn'*(comp_2_val )*zetacolumn;
        check_xcon_1(i,g) = zetacolumn'*(xcon_1_val)*zetacolumn;
        check_xcon_2(i,g) = zetacolumn'*(xcon_2_val)*zetacolumn;
        check_xquantize_2(i,g) = zetacolumn'*(xquantize_2_val)*zetacolumn;
        check_xiquantize_2(i,g) = zetacolumn'*(xiquantize_2_val)*zetacolumn;
        check_xcon_joint(i,g) = zetacolumn'*(xcon_joint_val )*zetacolumn;
        
        check_output(i,g) = zetacolumn'*(output_val)*zetacolumn;
        
    end
end
% zeta

disp('check_slope');

check_negative(check_slope);

disp('check_slope_1');

check_negative(check_slope_1);

disp('check_slope_2');

check_negative(check_slope_2);

disp('check_pos_1');

check_negative(check_pos_1);

disp('check_pos_2');

check_negative(check_pos_2);

disp('check_cpos_1');

check_negative(check_cpos_1);

disp('check_cpos_2');

check_negative(check_cpos_2);

disp('check_crx_1i');

check_negative(check_crx_1i);

disp('check_crx_1ii');

check_negative(check_crx_1ii);

disp('check_crx_2i');

check_negative(check_crx_2i);

disp('check_crx_2ii');

check_negative(check_crx_2ii);

disp('check_comp_1');

check_negative(check_comp_1);

disp('check_comp_2');

check_negative(check_comp_2);

disp('check_xcon_1');

check_negative(check_xcon_1);

disp('check_xcon_2');

check_negative(check_xcon_2);

disp('check_xcon_joint');

check_negative(check_xcon_joint);

disp('check_xquantize_2');

check_negative(check_xquantize_2);

disp('check_xiquantize_2');

check_negative(check_xiquantize_2);

disp('check_output');

check_negative(check_output);

output_inspect = difference - check_output;

mean(mean(output_inspect))

% end
% 
% save('W_save_BinaryQuant.mat','W_1','Wc_1','b_1','bc_1','Wl_1','bl_1','Wx_1','bx_1');

