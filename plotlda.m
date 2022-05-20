%% PCA+LDA Analysis i.e. Most Discriminant Feature Method
% Last update: 07/03/22 (needs to be checked but does run and give sensible output)

load("monkeydata_training.mat")
noDirections = 8;
group = 20;
win = 50;
noTrain =  60;
trialProcess =  bin_and_sqrt(trial, group, 1);
trialFinal = get_firing_rates(trialProcess,group,win);
[trainData,testData] = split_test_train(trialFinal,noTrain);
%all_rates = combine_rates(train_data,500);
reachAngles = [30 70 110 150 190 230 310 350]; % given in degrees
trimmer = 560/group; % make the trajectories the same length
firingData = zeros([size(trainData(1,1).rates,1)*trimmer,noTrain*noDirections]);
noNeurons = size(trainData(1,1).rates,1);

% need to get (neurons x time)x trial
for i = 1: noDirections
    for j = 1: noTrain
        for k = 1: trimmer
            firingData(noNeurons*(k-1)+1:noNeurons*k,noTrain*(i-1)+j) = trainData(j,i).rates(:,k);     
        end
    end
end


% The aim of the next section is to identify the reaching direction
% associated with each trial of this monkey's session

% supervised labelling for Linear Discrminant Analysis
dirLabels = [1*ones(1,noTrain),2*ones(1,noTrain),3*ones(1,noTrain),4*ones(1,noTrain),5*ones(1,noTrain),6*ones(1,noTrain),7*ones(1,noTrain),8*ones(1,noTrain)];

% implement Principal Component Analysis 
[princComp,eVals]= getPCA(firingData);

%%
% https://www.doc.ic.ac.uk/~dfg/ProbabilisticInference/old_IDAPILecture15.pdf

% use Linear Discriminant Analysis on Reduced Dimension Firing Data
matBetween = zeros(size(firingData,1),noDirections);
% for LDA need to get the across-class and within-class scatter matrices
for i = 1: noDirections
    matBetween(:,i) =  mean(firingData(:,noTrain*(i-1)+1:i*noTrain),2);
end
scatBetween = (matBetween - mean(firingData,2))*(matBetween - mean(firingData,2))';
scatGrand =  (firingData - mean(firingData,2))*(firingData - mean(firingData,2))';
scatWithin = scatGrand - scatBetween; % as per pdf above
%^^ double checking - done

% with this need to optimise the Fisher's criterion - in particular the
% most discriminant feature method (i.e. PCA --> LDA to avoid issues with
% low trial: neuron ratios)
pcaDim = 30;
[eVectsLDA, eValsLDA] = eig(((princComp(:,1:pcaDim)'*scatWithin*princComp(:,1:pcaDim))^-1 )*(princComp(:,1:pcaDim)'*scatBetween*princComp(:,1:pcaDim)));
[~,sortIdx] = sort(diag(eValsLDA),'descend');
% optimum output
optimOut = princComp(:,1:pcaDim)*eVectsLDA(:,sortIdx(1:2));
% optimum projection from the Most Discriminant Feature Method....!
W = optimOut'*(firingData - mean(firingData,2));

%%
colors = {[1 0 0],[0 1 1],[1 1 0],[0 0 0],[0 0.75 0.75],[1 0 1],[0 1 0],[1 0.50 0.25]};
figure
hold on
for i=1:noDirections
    plot(W(1,noTrain*(i-1)+1:i*noTrain),W(2,noTrain*(i-1)+1:i*noTrain),'o','Color',colors{i},'MarkerFaceColor',colors{i},'MarkerEdgeColor','k')
    hold on
end

legend('30 degrees','70 degrees','110 degrees','150 degrees','190 degrees','230 degrees','310 degrees','350 degrees');
%reachAngles = [30 70 110 150 190 230 310 350];
set(gcf,'color','w')
grid on
grid minor
xlabel('LDA Component 1','fontsize',16)
ylabel('LDA Component 2','fontsize',16)
title('Supervised PCA-LDA Classifier for Reaching Angle','fontsize',20)

%%
function [prinComp,evals,sortIdx,ev]= getPCA(data)
    % subtract the cross-trial mean
    dataCT = data - mean(data,2);
    % calculate the covariance matrix
    covMat = dataCT'*dataCT/size(data,2);
    % get eigenvalues and eigenvectors
    [evects, evals] = eig(covMat);
    % sort both eigenvalues and eigenvectors from largest to smallest weighting
    [~,sortIdx] = sort(diag(evals),'descend');
    evects = evects(:,sortIdx);
    % project firing rate data onto the newly derived basis
    prinComp = dataCT*evects;
    % normalisation
    prinComp = prinComp./sqrt(sum(prinComp.^2));
    % just getting the eigenvalues and not all the other zeros of the diagonal
    % matrix
    evalsDiag = diag(evals);
    evals = diag(evalsDiag(sortIdx));
end
function trialProcessed = bin_and_sqrt(trial, group, to_sqrt)

% Use to re-bin to different resolutions and to sqrt binned spikes (is used
% to reduce the effects of any significantly higher-firing neurons, which
% could bias dimensionality reduction)

% trial = the given struct
% group = new binning resolution - note the current resolution is 1ms
lag =0;% to_sqrt = binary , 1 -> sqrt spikes, 0 -> leave

    trialProcessed = struct;
    

    for i = 1: size(trial,2)
        for j = 1: size(trial,1)

            all_spikes = trial(j,i).spikes; % spikes is no neurons x no time points
            no_neurons = size(all_spikes,1);
            no_points = size(all_spikes,2);
            all_spikes = [zeros(no_neurons,lag/group),all_spikes(:,1:end-(lag/group))];
            t_new = 1: group : no_points +1; % because it might not round add a 1 
            spikes = zeros(no_neurons,numel(t_new)-1);

            for k = 1 : numel(t_new) - 1 % get rid of the paddded bin
                spikes(:,k) = sum(all_spikes(:,t_new(k):t_new(k+1)-1),2);
            end

            if to_sqrt
                spikes = sqrt(spikes);
            end

            trialProcessed(j,i).spikes = spikes;
            trialProcessed(j,i).handPos = trial(j,i).handPos(1:2,:);
            trialProcessed(j,i).bin_size = group; % recorded in ms
        end
    end
    
end

function trialFinal = get_firing_rates(trialProcessed,group,scale_window)

% trial = struct , preferably the struct which has been appropaitely binned
% and had low-firing neurons removed if needed
% group = binning resolution - depends on whether you have changed it with
% the bin_and_sqrt function
% scale_window = a scaling parameter for the Gaussian kernel - am
% setting at 50 now but feel free to mess around with it

    trialFinal = struct;
    win = 10*(scale_window/group);
    normstd = scale_window/group;
    alpha = (win-1)/(2*normstd);
    temp1 = -(win-1)/2 : (win-1)/2;
    gausstemp = exp((-1/2) * (alpha * temp1/((win-1)/2)) .^ 2)';
    gaussian_window = gausstemp/sum(gausstemp);
    
    for i = 1: size(trialProcessed,2)

        for j = 1:size(trialProcessed,1)
            
            hold_rates = zeros(size(trialProcessed(j,i).spikes,1),size(trialProcessed(j,i).spikes,2));
            
            for k = 1: size(trialProcessed(j,i).spikes,1)
                
                hold_rates(k,:) = conv(trialProcessed(j,i).spikes(k,:),gaussian_window,'same')/(group/1000);
            end
            
            trialFinal(j,i).rates = hold_rates;
            trialFinal(j,i).handPos = trialProcessed(j,i).handPos;
            trialFinal(j,i).bin_size = trialProcessed(j,i).bin_size; % recorded in ms
        end
    end

end