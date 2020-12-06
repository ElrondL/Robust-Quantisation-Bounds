function [W,Wc,b,bc,Wl,bl,Wx,bx]= compute_weights(n,l,nx,nf)

W = cell(1,l);
b = cell(1,l);

W{1, 1} = randn(n(1),nx);
b{1, 1} = randn(n(1),1);
Wc = [];
bc = [];

for i = 1:l-2
    W{1, i+1} = randn(n(i+1),n(i));
    b{1, i+1} = randn(n(i+1),1);
    Wc = blkdiag(Wc,W{1, i+1});
    bc = [bc;b{1, i+1}];
%     if i == l-2
%         Wc = blkdiag(Wc,W{1, i+1});
%         bc = [bc;b{1, i+1}];
%     end
    
end
W{1,l} = randn(nf,n(l-1));
b{1,l} = randn(nf,1);

Wx = W{1, 1};
bx = b{1, 1};
Wl = W{1,l};
bl = b{1,l};

end



% % scaling = 1e0;
% Wx = randn(n,nx) 
% bx = randn(n,1)
% W = randn(n,n) 
% b = randn(n,1)
% for i = 1:l-2
%     W = blkdiag(W,rand(n,n))
%     b = [b;randn(n,1)]
% end
% % bother = [rand(n,1);rand(n,1)];
% 
% % W = scaling*blkdiag(Wx,Wother); 
% % b = scaling*[bx;bother];
% 
% Wl = randn(nf,n)
% 
% 
% %%
% % scaling = 1e0;
% % Wx = rand(n,nx); bx = rand(n,1);
% % Wother = rand(n,n); bother = rand(n,1);
% % for i = 1:l-2
% %     Wother = blkdiag(Wother,rand(n,n));
% %     bother = [bother;rand(n,1)];
% % end
% % % bother = [rand(n,1);rand(n,1)];
% % 
% % W = scaling*blkdiag(Wx,Wother); 
% % b = scaling*[bx;bother];
% % 
% % Wl = rand(nf,n);

%end
