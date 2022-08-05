%% This section creates toy data.
%
% It should be replaced by actual experimental data. The data should be
% joined in three arrays of the following sizes (for the Romo-like task):
%
% trialNum: N x S x D
% firingRates: N x S x D x T x maxTrialNum
% firingRatesAverage: N x S x D x T
%
% N is the number of neurons
% S is the number of stimuli conditions (F1 frequencies in Romo's task)
% D is the number of decisions (D=2)
% T is the number of time-points (note that all the trials should have the
% same length in time!)
%
% trialNum -- number of trials for each neuron in each S,D condition (is
% usually different for different conditions and different sessions)
%
% firingRates -- all single-trial data together, massive array. Here
% maxTrialNum is the maximum value in trialNum. E.g. if the number of
% trials per condition varied between 1 and 20, then maxTrialNum = 20. For
% the neurons and conditions with less trials, fill remaining entries in
% firingRates with zeros or nans.
%
% firingRatesAverage -- average of firingRates over trials (5th dimension).
% If the firingRates is filled up with nans, then it's simply
%    firingRatesAverage = nanmean(firingRates,5)
% If it's filled up with zeros (as is convenient if it's stored on hard 
% drive as a sparse matrix), then 
%    firingRatesAverage = bsxfun(@times, mean(firingRates,5), size(firingRates,5)./trialNum)

% load('stressbuffer_2019_seitzman_ignoreSubjs.mat');
% load('stressbuffer_2019_seitzman_ignoreSubjs_perspectiveVsSpatial.mat');
% load('stressbuffer_2019_seitzman_noIgnoreSubjs_otherVSelf.mat');

% load('stressbuffer_2019_seitzmanSelectedChannelsN40_ignoreSubjs_otherVSelf.mat');
% load('stressbuffer_2019_seitzman_ignoreSubjs_otherVSelf.mat');
% load('stressbuffer_2019_seitzmanSelectedChannels_ignoreSubjs_otherVSelf.mat');

% load('stressbuffer_20192022_seitzmanSelectedChannelsN40_ignoreSubjs_otherVSelf.mat');
% load('stressbuffer_20192022_schaefer200_ignoreSubjs_otherVSelf.mat');
load('stressbuffer_20192022_schaeferCor-HOSub214channels_ignoreSubjs_otherVSelf.mat');

data = squeeze(data);
trialCount = squeeze(trialCount(:,:,1));
% trialCount = [];
% trialCount(1) = sum(~isnan(data(1,1,1,:)));
% trialCount(2) = sum(~isnan(data(1,2,1,:)));

N = size(data,1);   % number of neurons
T = size(data,3);     % number of time points
S = size(data,2);       % number of stimuli
E = size(data,4);     % maximal number of trial repetitions

time = (1:T);

trialNum = trialCount;
firingRates = data;
firingRatesAverage = nanmean(firingRates, 4);

subjIDs = nan(size(data,2), size(data,4));
conditionCount = [1 1];
for i = 1:length(subjsList)
    conditionID = conditionList(i)+1;
    subjIDs(conditionID, conditionCount(conditionID)) = subjsList(i);
    
    conditionCount(conditionID) = conditionCount(conditionID) + 1;
end



%% Define parameter grouping

% *** Don't change this if you don't know what you are doing! ***
% firingRates array has [N S D T E] size; herewe ignore the 1st dimension 
% (neurons), i.e. we have the following parameters:
%    1 - stimulus 
%    2 - decision
%    3 - time
% There are three pairwise interactions:
%    [1 3] - stimulus/time interaction
%    [2 3] - decision/time interaction
%    [1 2] - stimulus/decision interaction
% And one three-way interaction:
%    [1 2 3] - rest
% As explained in the eLife paper, we group stimulus with stimulus/time interaction etc.:

combinedParams = {{1, [1 2]}, {2}};
margNames = {'Condition', 'Condition-independent'};
margColours = [23 100 171; 187 20 25]/256;

% For two parameters (e.g. stimulus and time, but no decision), we would have
% firingRates array of [N S T E] size (one dimension less, and only the following
% possible marginalizations:
%    1 - stimulus
%    2 - time
%    [1 2] - stimulus/time interaction
% They could be grouped as follows: 
%    combinedParams = {{1, [1 2]}, {2}};

% Time events of interest (e.g. stimulus onset/offset, cues etc.)
% They are marked on the plots with vertical lines
timeEvents = [];

% check consistency between trialNum and firingRates
% for n = 1:size(firingRates,1)
%     for s = 1:size(firingRates,2)
%         for d = 1:size(firingRates,3)
%             assert(isempty(find(isnan(firingRates(n,s,d,:,1:trialNum(n,s,d))), 1)), 'Something is wrong!')
%         end
%     end
% end

%% Step 1: PCA of the dataset

X = firingRatesAverage(:,:);
X = bsxfun(@minus, X, mean(X,2));

[W,~,~] = svd(X, 'econ');
W = W(:,1:20);

% minimal plotting
dpca_plot(firingRatesAverage, W, W, @dpca_plot_default);

% computing explained variance
explVar = dpca_explainedVariance(firingRatesAverage, W, W, ...
    'combinedParams', combinedParams);

% a bit more informative plotting
dpca_plot(firingRatesAverage, W, W, @dpca_plot_default, ...
    'explainedVar', explVar, ...
    'time', time,                        ...
    'timeEvents', timeEvents,               ...
    'marginalizationNames', margNames, ...
    'marginalizationColours', margColours);


%% Step 2: PCA in each marginalization separately

dpca_perMarginalization(firingRatesAverage, @dpca_plot_default, ...
   'combinedParams', combinedParams);

%% Step 3: dPCA without regularization and ignoring noise covariance

% This is the core function.
% W is the decoder, V is the encoder (ordered by explained variance),
% whichMarg is an array that tells you which component comes from which
% marginalization

tic
[W,V,whichMarg] = dpca(firingRatesAverage, 20, ...
    'combinedParams', combinedParams);
toc

explVar = dpca_explainedVariance(firingRatesAverage, W, V, ...
    'combinedParams', combinedParams);

dpca_plot(firingRatesAverage, W, V, @dpca_plot_default, ...
    'explainedVar', explVar, ...
    'marginalizationNames', margNames, ...
    'marginalizationColours', margColours, ...
    'whichMarg', whichMarg,                 ...
    'time', time,                        ...
    'timeEvents', timeEvents,               ...
    'timeMarginalization', 3, ...
    'legendSubplot', 16);


%% Step 4: dPCA with regularization

% This function takes some minutes to run. It will save the computations 
% in a .mat file with a given name. Once computed, you can simply load 
% lambdas out of this file:
%   load('tmp_optimalLambdas.mat', 'optimalLambda')

% Please note that this now includes noise covariance matrix Cnoise which
% tends to provide substantial regularization by itself (even with lambda set
% to zero).

catData = [];
for i = 1:size(data,2)
    for j = 1:size(data,4)
        catData = [catData squeeze(data(:,i,:,4))];
    end
end
catData(isnan(catData)) = [];

[pcaBasis, ~, ~, ~, explained] = pca(catData');
useComponents = find(cumsum(explained) > 95, 1);
useComponents = 100;

pcaData = zeros(useComponents, size(data,2), size(data,3), max(trialCount(1,:)));
for i = 1:size(pcaData,2)
    for j = 1:size(pcaData,3)
        pcaData(:,i,j,:) = pcaBasis(:,1:useComponents)' * squeeze(data(:,i,j,1:max(trialCount(1,:))));
    end
end



pcaData = data;
useComponents = 300;


%%
% TRAIN_PERCENT = 0.9;

pcaData = data;

allMethodBetas = {};
allMethodProjections = {};
allMethodVarianceExplained = {};
allMethodF2s = [];
allMethodConfusions = [];

allMethodNames = {};
allMethodAccuracies = [];
% load('allMethodAccuracies.mat');
currentRuns = size(allMethodAccuracies,3);
currentRuns = 2;
if currentRuns > 1
    currentRuns = currentRuns - 1;
else
    currentRuns = 1;
end
for runNum = currentRuns:1000
    

    data = squeeze(data);
    trialCount = squeeze(trialCount(:,:,1));

    N = size(data,1);   % number of neurons
    T = size(data,3);     % number of time points
    S = size(data,2);       % number of stimuli
    E = size(data,4);     % maximal number of trial repetitions

    time = (1:T);

    trialNum = trialCount;
    firingRates = data;
    firingRatesAverage = nanmean(firingRates, 4);

    ifSimultaneousRecording = true;

    kFold = 10;

    TRIAL_LEVEL_BOOTSTRAP = 0;
    if TRIAL_LEVEL_BOOTSTRAP
        kthTrainIndices = {};
        kthValidateIndices = {};
        conditionIndices = {};
        for i = 1:size(pcaData,2)
            conditionIndices{i} = randsample(1:trialCount(1,i), trialCount(1,i), true);
        end
        for k = 1:kFold
            for i = 1:size(pcaData,2)
                trials = length(conditionIndices{i});

                startTrial = round(1 + (k-1) * trials/kFold);
                endTrial = round(k * trials/kFold);

                kthValidateIndices{k, i} = conditionIndices{i}(startTrial:endTrial);
                kthTrainIndices{k,i} = setdiff(1:trials, kthValidateIndices{k, i});
            end
        end
    else
        allSubjs = unique(subjsList);
        
        kthTrainIndices = {};
        kthValidateIndices = {};
        randomizedSubjIDs =  randsample(1:length(allSubjs), length(allSubjs), true);
        numSubjs = length(allSubjs);
        
        for k = 1:kFold
            for i = 1:size(pcaData,2)
                startTrial = round(1 + (k-1) * numSubjs/kFold);
                endTrial = round(k * numSubjs/kFold);
                
                validateIDs = randomizedSubjIDs(startTrial:endTrial);
                trainSubjs = allSubjs(setdiff(1:numSubjs, validateIDs));
                validateSubjs = allSubjs(validateIDs);
                
                validateIDs = [];
                for j = 1:length(validateSubjs)
                    validateIDs = [validateIDs find(subjIDs(i,:) == validateSubjs(j))];
                end
                
                trainIDs = [];
                for j = 1:length(trainSubjs)
                    trainIDs = [trainIDs find(subjIDs(i,:) == trainSubjs(j))];
                end
                kthValidateIndices{k, i} = validateIDs;
                kthTrainIndices{k,i} = trainIDs;
            end
        end
    end
            
    for processingIndex = 1:3
        for meanTypeIndex = 1:3        
        

            allConditionVectors = [];
            allDPCAData = [];
            allTrainIndicies = {};
            allTestIndicies = {};
            allCategoryAccuracies = [];
            allCategoryF2s = [];
            allCategoryBetas = [];
            allCategoryConfusions = [];
            allCategoryProjections = [];
            allCategoryVarianceExplained = [];
            for trainTest = 1:kFold

                pcaData = data;
                useComponents = size(pcaData,1);

                validateIndices = {};
                for i = 1:size(pcaData,2)
                    trainIndices{i} = kthTrainIndices{trainTest,i};
                    validateIndices{i} = kthValidateIndices{trainTest,i};
                end

                % TRAIN_PERCENT = 0.9;
                % 
                % trainIndices = {};
                % validateIndices = {};
                % for i = 1:size(pcaData,2)
                %     trainIndices{i} = randsample(1:trialCount(1,i), floor(TRAIN_PERCENT*trialCount(1,i)));
                %     validateIndices{i} = setdiff(1:trialCount(1,i), trainIndices{i});
                % end

                trialNum = floor(trialCount(1:useComponents,:) * (kFold-1) / kFold);
                dataSize = size(pcaData);
                dataSize(4) = max(trialNum(1,:));
                trainingFiringRates = nan(dataSize);
                validationFiringRates = nan(dataSize);

                allFiringRates = pcaData;

                for i = 1:size(pcaData,2)
                    for j = 1:size(pcaData,3)
                        trainingFiringRates(:, i, j, 1:length(trainIndices{i})) = pcaData(:,i,j,trainIndices{i});
                        validationFiringRates(:, i, j, 1:length(validateIndices{i})) = pcaData(:,i,j,validateIndices{i});
                    end
                end



                firingRatesAverage = nanmean(trainingFiringRates, 4);

                processingType = processingIndex;

                if processingType == 2 %No DR                                        
                    MAX_CHANNELS = size(X,1);
                    conditionVectors = eye(MAX_CHANNELS);
                    
                    processingTypeString = "None";
                    
                    X = firingRatesAverage(:,:);
                    X = bsxfun(@minus, X, mean(X,2));
                    
                    correlationMatrix = X * X';
                    
                    explainedVaraince = sum(correlationMatrix.^2) ./ sum(sum(correlationMatrix.^2));
                    
                elseif processingType == 1 %PCA
                    MAX_CHANNELS = 40;
                    X = firingRatesAverage(:,:);
                    X = bsxfun(@minus, X, mean(X,2));

                    [W,S,~] = svd(X, 'econ');
                    S = diag(S).^2 ./ sum(trace(S.^2));
                    W = W(:,1:MAX_CHANNELS);
                    conditionVectors = W;
                    
                    explainedVaraince = S';
                    
                    processingTypeString = "PCA";
                else %dPCA
                    MAX_CHANNELS = 6;
                    
                    tries = 0;
                    while tries < 10
                        try
                            optimalLambda = dpca_optimizeLambda(firingRatesAverage, trainingFiringRates, trialNum, ...
                                'combinedParams', combinedParams, ...
                                'simultaneous', ifSimultaneousRecording, ...
                                'numRep', 10, ...  % increase this number to ~10 for better accuracy
                                'filename', 'tmp_optimalLambdas.mat', 'display', 'no');
                        catch
                            tries = tries + 1;
                            disp('dPCA had issues -- retrying');
                        end
                        
                        tries = tries + 100;
                    end
                    
                    if tries == 10
                        disp('dPCA failed. Giving up.');
                        return;
                    end
                    
                    Cnoise = dpca_getNoiseCovariance(firingRatesAverage, ...
                        trainingFiringRates, trialNum, 'simultaneous', ifSimultaneousRecording);

                    [W,V,whichMarg] = dpca(firingRatesAverage, 20, ...
                        'combinedParams', combinedParams, ...
                        'lambda', optimalLambda, ...
                        'Cnoise', Cnoise);

                    explVar = dpca_explainedVariance(firingRatesAverage, W, V, ...
                        'combinedParams', combinedParams);

%                     dpca_plot(firingRatesAverage, W, V, @dpca_plot_default, ...
%                         'explainedVar', explVar, ...
%                         'marginalizationNames', margNames, ...
%                         'marginalizationColours', margColours, ...
%                         'whichMarg', whichMarg,                 ...
%                         'time', time,                        ...
%                         'timeEvents', timeEvents,               ...
%                         'timeMarginalization', 3,           ...
%                         'legendSubplot', 16);

                    explainedVaraince = explVar.componentVar;

                    conditionVectors = W(:,whichMarg==1);
                    % save("conditionVectors.mat","conditionVectors");
                    
                    processingTypeString = "dPCA";
                end
                
                useConditionVectors = conditionVectors(:,1:MAX_CHANNELS);

                % Classification
                pcaTraces = zeros(size(conditionVectors,2), size(pcaData,2), size(pcaData,3), max(trialCount(1,:)));
                for i = 1:size(pcaTraces,2)
                    for j = 1:size(pcaTraces,3)
                        pcaTraces(:,i,j,:) = conditionVectors' * squeeze(allFiringRates(:,i,j,1:max(trialCount(1,:))));
                    end
                end

                thisTrainIndicies = [];
                thisTestIndicies = [];
                catData = [];
                catYs = [];
                for i = 1:size(pcaTraces,2)
                    catData = cat(3, catData, squeeze(pcaTraces(1:MAX_CHANNELS,i,:,1:trialCount(1,i))));
                    catYs = [catYs; ones(trialCount(1,i),1) * i];
                    if i == 1
                        thisTrainIndicies = trainIndices{i};
                        thisTestIndicies = validateIndices{i};
                    else
                        thisTrainIndicies = [thisTrainIndicies, trainIndices{i} + trialCount(1,1)];
                        thisTestIndicies = [thisTestIndicies, validateIndices{i} + trialCount(1,1)];
                    end
                end
                allYs = catYs;
%                 allDPCAData(trainTest,:,:,:) = catData;
% 
%                 allTrainIndicies{trainTest} = thisTrainIndicies;
%                 allTestIndicies{trainTest} = thisTestIndicies;

              

                meanType = meanTypeIndex;

                if meanType == 2 %no means
                    projections = nan(size(pcaTraces,2),MAX_CHANNELS*size(pcaTraces,3),size(pcaTraces,4));
                    for i = 1:size(pcaTraces,2)
                        projections(i,:,:) = reshape(pcaTraces(1:MAX_CHANNELS,i,:,:), [MAX_CHANNELS * size(pcaTraces,3), size(pcaTraces,4)]);
                    end
                    mergedProjections = projections;
                    
                    meanTypeString = "(none)";
                elseif meanType == 1 %mean across time
                    projections = nan(size(pcaTraces,2),MAX_CHANNELS,size(pcaTraces,4));
                    for i = 1:size(pcaTraces,2)
                        for j = 1:MAX_CHANNELS
                            % Use train indices to calculate the means
                            projections(i,j,:,:) = squeeze(nanmean(pcaTraces(j,i,:,:),3));
                        end
                    end
                    mergedProjections = projections;
                    
                    meanTypeString = "(time)";                    
                else %correlate with mean trajectory
                    projections = nan(size(pcaTraces,2),2,MAX_CHANNELS,size(pcaTraces,4));
                    for i = 1:size(pcaTraces,2)
                        for j = 1:MAX_CHANNELS
                            % Use train indices to calculate the means
                            mean1 = squeeze(nanmean(pcaTraces(j,1,:,trainIndices{1}),4));
                            mean2 = squeeze(nanmean(pcaTraces(j,2,:,trainIndices{2}),4));

                            for k = 1:trialCount(1,i)
                    %             projections(i,1,j,k) = squeeze(pcaTraces(j,i,:,k))' * mean1;
                    %             projections(i,2,j,k) = squeeze(pcaTraces(j,i,:,k))' * mean2;
                                projections(i,1,j,k) = corr(squeeze(pcaTraces(j,i,:,k)), mean1);
                                projections(i,2,j,k) = corr(squeeze(pcaTraces(j,i,:,k)), mean2);
                            end
                        end
                    end
                    mergedProjections = reshape(projections, size(pcaTraces,2),2*MAX_CHANNELS,size(pcaTraces,4));
                
                    meanTypeString = "(traj)";     
                end

                PLOT_FIGURE = 0;
                if PLOT_FIGURE == 1
                    means = [];
                    lower = [];
                    upper = [];
                    for i = 1:size(mergedProjections,1)
                        for j = 1:size(mergedProjections,2)
                            means(i,j,:) = squeeze(nanmean(mergedProjections(i,j,:),3));        
                            lower(i,j) = quantile(mergedProjections(i,j,:),0.025,3);
                            upper(i,j) = quantile(mergedProjections(i,j,:),0.975,3);
                        end
                    end

                    figure(1);
                    clf;
                    hold on;

                    condition1 = reshape(squeeze(mergedProjections(1,:,:)), [size(mergedProjections,2)*size(mergedProjections,3),1]);
                    condition2 = reshape(squeeze(mergedProjections(2,:,:)), [size(mergedProjections,2)*size(mergedProjections,3),1]);
                    indicies = reshape(repmat(1:size(mergedProjections,2), [size(mergedProjections,3), 1])', [size(mergedProjections,2)*size(mergedProjections,3),1]);

                    scatter(indicies + normrnd(0,0.1,size(indicies)), condition1, 'r.');
                    scatter(indicies + normrnd(0,0.1,size(indicies)), condition2, 'b.');

                    condition1Validate = kron((validateIndices{1} - 1) * size(mergedProjections,2), ones(1, size(mergedProjections,2))) + repmat((1:size(mergedProjections,2)), [1, length(validateIndices{1})]);
                    condition2Validate = kron((validateIndices{2} - 1) * size(mergedProjections,2), ones(1, size(mergedProjections,2))) + repmat((1:size(mergedProjections,2)), [1, length(validateIndices{2})]);

                    scatter(indicies(condition1Validate) + normrnd(0,0.1,size(indicies(condition1Validate))), condition1(condition1Validate), 72, 'rx');
                    scatter(indicies(condition2Validate) + normrnd(0,0.1,size(indicies(condition2Validate))), condition2(condition2Validate), 72, 'bx');

                    plotShadedCI(means(1,:), [lower(1,:); upper(1,:)], 1:size(mergedProjections,2), 'r');
                    plotShadedCI(means(2,:), [lower(2,:); upper(2,:)], 1:size(mergedProjections,2), 'b');
                end
                
                % Build classifier

                allData = [];
                allClasses = [];
                for i = 1:2
                    for j = 1:trialCount(1,i)
                %         allData(:,end+1) = mergedProjections(i,:,j) / sqrt(sum(mergedProjections(i,:,j).^2));
                        allData(:,end+1) = mergedProjections(i,:,j);
                        allClasses(end+1) = i;
                    end
                end

                % minTrials = min(length(trainIndices{1}), length(trainIndices{2}));
                % resampledTrainIndicies = {};
                % for i = 1:size(data,2)
                %     resampledTrainIndicies{i} = randsample(1:length(trainIndices{i}), minTrials);
                % end

                % trainIndicies = [resampledTrainIndicies{1}, trialCount(1,1) + resampledTrainIndicies{2}];
                trainIndicies = [trainIndices{1}, trialCount(1,1) + trainIndices{2}];
                validateIndicies = [validateIndices{1}, trialCount(1,1) + validateIndices{2}];

                trainData = allData(:,trainIndicies)';
                trainClasses = allClasses(trainIndicies)';
                validateData = allData(:,validateIndicies)';
                validateClasses = allClasses(validateIndicies)';
                
                meanTrainData = mean(trainData);
                stdTrainData = std(trainData);
                
                trainData = (trainData - meanTrainData) ./ stdTrainData;
                validateData = (validateData - meanTrainData) ./ stdTrainData;

                trainClasses = trainClasses - 1;
                validateClasses = validateClasses - 1;

                Mdl = fitclinear(trainData,trainClasses, 'Prior', [1 1], 'Learner', 'logistic');%, 'Regularization', 'lasso');
%                 Mdl = fitcsvm(trainData, trainClasses, 'Weights', weights);

%                 Mdl = fitclinear(trainData,trainClasses, 'Prior', [1 1], 'Regularization', 'lasso');
%                 Mdl = fitcdiscr(trainData,trainClasses, 'Prior', [1 1]);
            %     Mdl = fitglm(trainData,trainClasses,'Distribution','binomial');
            
                trainAccuracy = sum(trainClasses == predict(Mdl, trainData))/length(trainClasses);
                correctPrediction = double(validateClasses == predict(Mdl, validateData));
                categoryAccuracies = [mean(correctPrediction(validateClasses == 0)), mean(correctPrediction(validateClasses == 1))];
                
                confusion = confusionmat(validateClasses,predict(Mdl, validateData));
                
                allCategoryConfusions(trainTest,:,:) = confusion;
                
                precision = confusion(1,1) / sum(confusion(:,1));
                recall = confusion(1,1) / sum(confusion(1,:));
                class1F2 = 5*precision*recall/(4*precision + recall);
                
                precision = confusion(2,2) / sum(confusion(:,2));
                recall = confusion(2,2) / sum(confusion(2,:));
                class2F2 = 5*precision*recall/(4*precision + recall);

                allCategoryF2s(trainTest,:) = (class1F2 + class2F2) / 2;
                
                allCategoryBetas(trainTest,:) = Mdl.Beta;
                
                allCategoryProjections(trainTest,:,:) = useConditionVectors;
                
                % validateAccuracy = mean(categoryAccuracies)

                allCategoryAccuracies(trainTest,:) = categoryAccuracies;
                
                allCategoryVarianceExplained(trainTest,:) = explainedVaraince;
            end
            
            %meanCategoryAccuracies = mean(allCategoryAccuracies,1)
                     
            allMethodBetas{processingIndex,meanTypeIndex,runNum} = allCategoryBetas;
            allMethodProjections{processingIndex,meanTypeIndex,runNum} = allCategoryProjections;
            allMethodVarianceExplained{processingIndex,meanTypeIndex,runNum} = allCategoryVarianceExplained;
            
            allMethodF2s(processingIndex,meanTypeIndex,runNum,:) = allCategoryF2s;
            
            meanConfusions = mean(allCategoryConfusions,1);
            allMethodConfusions(processingIndex,meanTypeIndex,runNum,:,:) = meanConfusions;            
            
            allFoldAccuracies = mean(allCategoryAccuracies,2);            
            allMethodAccuracies(processingIndex,meanTypeIndex,runNum,:) = allFoldAccuracies;
            
            allMethodNames{processingIndex,meanTypeIndex} = strcat(processingTypeString,meanTypeString);
            
            disp(strcat("Model run ", num2str(runNum), ": ", processingTypeString, meanTypeString, " -> ", num2str(mean(allFoldAccuracies)), " F2: ", num2str(mean(allCategoryF2s))));
            pause(0.1);
        end
    end
    
    allMeanAccuracies = reshape(nanmean(allMethodAccuracies,4), [size(allMethodAccuracies,1)*size(allMethodAccuracies,2),size(allMethodAccuracies,3)]);
    allMeansF2s = reshape(nanmean(allMethodF2s,4), [size(allMethodF2s,1)*size(allMethodF2s,2),size(allMethodF2s,3)]);
    allMeansBetas = reshape(allMethodBetas, [size(allMethodBetas,1)*size(allMethodBetas,2),size(allMethodBetas,3)]);
    
    allMeansProjections = reshape(allMethodProjections, [size(allMethodProjections,1)*size(allMethodProjections,2),size(allMethodProjections,3)]);
    allMeansVarianceExplained = reshape(allMethodVarianceExplained, [size(allMethodVarianceExplained,1)*size(allMethodVarianceExplained,2),size(allMethodVarianceExplained,3)]);
    
    allMeansConfusions = reshape(allMethodConfusions, [size(allMethodConfusions,1)*size(allMethodConfusions,2),size(allMethodConfusions,3),size(allMethodConfusions,4),size(allMethodConfusions,5)]);
    
%     useMetric = allMeansF2s;
    useMetric = allMeanAccuracies;
    
    allPValues = [];
    allPValueStrings = {};
    for i = 1:size(useMetric,1)
        for j = 1:size(useMetric,1)
            allPValues(i,j) = sum(useMetric(i,:) >= useMetric(j,:))/size(useMetric,2);
            allPValueStrings{i,j} = num2str(allPValues(i,j),3);
            
            if i == j
                allPValueStrings{i,j} = '';
            end
            if allPValues(i,j) > 0.1
                allPValueStrings{i,j} = '';
            end
            if allPValues(i,j) == 0
                allPValueStrings{i,j} = ['<' num2str(1/size(useMetric,2),2)];
            end
        end
    end
    
    methodNames = reshape(allMethodNames, [size(allMethodNames,1)*size(allMethodNames,2),1]);
    
    allMeans = nanmean(useMetric,2);
    meanStrings = {};
    for i = 1:length(allMeans)
        meanStrings{i} = num2str(allMeans(i),3);
    end
    
    [textX, textY] = meshgrid(1:size(useMetric,1));
    
    figure(1);
    clf;
    colormap(parula);
    imagesc(-log(allPValues')/log(20));
    colorbar;
    xticks(1:size(useMetric,1))
    yticks(1:size(useMetric,1))
    xticklabels(methodNames);
    yticklabels(methodNames);
    xtickangle(45);
    ytickangle(45);
    title(strcat("Comparison after ", num2str(size(useMetric,2)), " runs"));
    text(1:size(useMetric,1),1:size(useMetric,1),meanStrings,'Color','white','HorizontalAlignment','center');
    text(textY(:),textX(:),allPValueStrings(:),'HorizontalAlignment','center');
    
    save('runData.mat', 'allMeanAccuracies', 'allMeansF2s', 'allMeansBetas', 'allMeansProjections', 'allMeansProjections', 'allMeansVarianceExplained', 'allMeansConfusions', 'methodNames');
end



%quantile(mean(allCategoryAccuracies,2), 0.05)

%mean(mean(allCategoryAccuracies,2))

% reducedROIAccuracies
% reduced40ROIAccuracies
% reduced40ROIPCAAccuracies

%% Optional: estimating "signal variance"

explVar = dpca_explainedVariance(firingRatesAverage, W, V, ...
    'combinedParams', combinedParams, ...
    'Cnoise', Cnoise, 'numOfTrials', trialNum);

% Note how the pie chart changes relative to the previous figure.
% That is because it is displaying percentages of (estimated) signal PSTH
% variances, not total PSTH variances. See paper for more details.

dpca_plot(firingRatesAverage, W, V, @dpca_plot_default, ...
    'explainedVar', explVar, ...
    'marginalizationNames', margNames, ...
    'marginalizationColours', margColours, ...
    'whichMarg', whichMarg,                 ...
    'time', time,                        ...
    'timeEvents', timeEvents,               ...
    'timeMarginalization', 3,           ...
    'legendSubplot', 16);

%% Optional: decoding

decodingClasses = {[(1:S)' (1:S)'], repmat([1:2], [S 1]), [], [(1:S)' (S+(1:S))']};

accuracy = dpca_classificationAccuracy(firingRatesAverage, firingRates, trialNum, ...
    'lambda', optimalLambda, ...
    'combinedParams', combinedParams, ...
    'decodingClasses', decodingClasses, ...
    'simultaneous', ifSimultaneousRecording, ...
    'numRep', 5, ...        % increase to 100
    'filename', 'tmp_classification_accuracy.mat');

dpca_classificationPlot(accuracy, [], [], [], decodingClasses)

accuracyShuffle = dpca_classificationShuffled(firingRates, trialNum, ...
    'lambda', optimalLambda, ...
    'combinedParams', combinedParams, ...
    'decodingClasses', decodingClasses, ...
    'simultaneous', ifSimultaneousRecording, ...
    'numRep', 5, ...        % increase to 100
    'numShuffles', 20, ...  % increase to 100 (takes a lot of time)
    'filename', 'tmp_classification_accuracy.mat');

dpca_classificationPlot(accuracy, [], accuracyShuffle, [], decodingClasses)

componentsSignif = dpca_signifComponents(accuracy, accuracyShuffle, whichMarg);

dpca_plot(firingRatesAverage, W, V, @dpca_plot_default, ...
    'explainedVar', explVar, ...
    'marginalizationNames', margNames, ...
    'marginalizationColours', margColours, ...
    'whichMarg', whichMarg,                 ...
    'time', time,                        ...
    'timeEvents', timeEvents,               ...
    'timeMarginalization', 3,           ...
    'legendSubplot', 16,                ...
    'componentsSignif', componentsSignif);
