%function to take in a random NN generated in robust_NN_guarantees, prune
%it, and return the pruned weights, l is number of transitions
%Pruning is only applied to hidden layers, pruneDepth is number of neurons
%to be pruned

function [W_2,Wc_2,Wl_2,Wx_2] = pruneUniversal(W_1,l,pruneDepth)

score = [];

W_2 = W_1;
Wc_2 = [];


for i = 2:l-1
    score = [score,vecnorm(W_1{i}')]; %evaluate the norm of each neuron weight(s)
end

[scoreRank,Index] = sort(score) %sort the score to find the weights with smallest norms

for i = 1:pruneDepth %pick the eight smallest neuron weights
    if floor(Index(i)/10)*10 ~= Index(i)
        id = Index(i)+10; %pick the index
        cellNum = floor(id/10)+1 %find out the cell that contains weak weights
        rowNum = id - (cellNum-1)*10 %find out the row number for the weak weights
    else
        id = Index(i)+10;
        cellNum = id/10
        rowNum = 10
    end
    W_2{cellNum}(rowNum,:) = 0 %set the weak weights to zero
    
end

Wx_2 = W_2{1, 1};
Wl_2 = W_2{1, l};

for i = 1:l-2
    Wc_2 = blkdiag(Wc_2,W_2{1, i+1});   %set into right formats
end

save('W_save_pruned.mat','W_2','Wc_2','Wl_2','Wx_2'); %save

end

    

