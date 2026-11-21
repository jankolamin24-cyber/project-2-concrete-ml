%% Project 2 - Concrete Compressive Strength Prediction
% Student: Muhammed L. Janko
% Requires Statistics and Machine Learning Toolbox.
% Download Concrete_Data.xls from https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
% Put the dataset in the data folder (or beside this script).
clear; clc; close all;
rng(42,'twister');
root = fileparts(mfilename('fullpath'));
if isempty(root), root = pwd; end
figDir = fullfile(root,'figures');
if ~exist(figDir,'dir'), mkdir(figDir); end
names = {'Concrete_Data.xls','Concrete_Data.xlsx','Concrete_Data.csv'};
filename = '';
for k=1:numel(names)
    p = fullfile(root,'data',names{k});
    if isfile(p), filename=p; break; end
    p = fullfile(root,names{k});
    if isfile(p), filename=p; break; end
end
if isempty(filename)
    error('Dataset missing. Download Concrete_Data.xls from UCI and place it in the data folder.');
end
%% Part A: Inspect
T = readtable(filename);
disp('First five rows:'); disp(head(T,5));
[nRows,nCols] = size(T);
fprintf('Rows: %d; Columns: %d\n',nRows,nCols);
disp('Original variable names:'); disp(T.Properties.VariableNames');
disp('Variable types:');
for j=1:width(T), fprintf('%s: %s\n',T.Properties.VariableNames{j},class(T{:,j})); end
fprintf('Total missing entries: %d\n',sum(ismissing(T),'all'));
summary(T);
assert(nCols==9,'Expected eight features and one target column.');
assert(all(varfun(@isnumeric,T,'OutputFormat','uniform')),'Expected numeric dataset columns.');
assert(~any(ismissing(T),'all'),'Missing values found; inspect before modeling.');
% The UCI dataset has eight predictors followed by strength.
A = T{:,:}; X = A(:,1:8); y = A(:,9);
featureNames = {'Cement','Blast Furnace Slag','Fly Ash','Water', ...
    'Superplasticizer','Coarse Aggregate','Fine Aggregate','Age'};
%% Part B: Exploratory figures
f=figure('Visible','off'); histogram(y,25); xlabel('Strength (MPa)'); ylabel('Frequency'); title('Concrete Compressive Strength'); grid on;
saveas(f,fullfile(figDir,'strength_histogram.png')); close(f);
f=figure('Visible','off'); scatter(X(:,8),y,14,'filled'); xlabel('Age (days)'); ylabel('Strength (MPa)'); title('Age vs Strength'); grid on;
saveas(f,fullfile(figDir,'age_vs_strength.png')); close(f);
f=figure('Visible','off'); scatter(X(:,1),y,14,'filled'); xlabel('Cement (kg/m^3)'); ylabel('Strength (MPa)'); title('Cement vs Strength'); grid on;
saveas(f,fullfile(figDir,'cement_vs_strength.png')); close(f);
C = corrcoef(A);
f=figure('Visible','off'); imagesc(C); colorbar; clim([-1 1]); axis square;
labels = [featureNames, {'Strength'}];
set(gca,'XTick',1:9,'XTickLabel',labels,'XTickLabelRotation',45,'YTick',1:9,'YTickLabel',labels);
title('Correlation Matrix');
saveas(f,fullfile(figDir,'correlation_heatmap.png')); close(f);
%% Part C: Holdout split (same test set for all models)
p = cvpartition(numel(y),'HoldOut',0.20);
trainIdx=training(p); testIdx=test(p);
Xtrain=X(trainIdx,:); Xtest=X(testIdx,:);
ytrain=y(trainIdx); ytest=y(testIdx);
% No scaling: tree models are scale insensitive and linear regression
% works in the original engineering units.
%% Part D: Train models
linearModel=fitlm(Xtrain,ytrain);
treeModel=fitrtree(Xtrain,ytrain,'PredictorNames',featureNames,'MinLeafSize',5);
forestModel=TreeBagger(150,Xtrain,ytrain,'Method','regression', ...
    'PredictorNames',featureNames,'OOBPredictorImportance','on','MinLeafSize',5);
%% Part E: Evaluate models
predLinear=predict(linearModel,Xtest);
predTree=predict(treeModel,Xtest);
predForest=predict(forestModel,Xtest);
[mae1,rmse1,r21]=regressionMetrics(ytest,predLinear);
[mae2,rmse2,r22]=regressionMetrics(ytest,predTree);
[mae3,rmse3,r23]=regressionMetrics(ytest,predForest);
results=table({'Linear Regression';'Decision Tree';'Bagged Trees'}, ...
    [mae1;mae2;mae3],[rmse1;rmse2;rmse3],[r21;r22;r23], ...
    'VariableNames',{'Model','MAE','RMSE','R2'});
disp('Test-set results:'); disp(results);
writetable(results,fullfile(root,'model_comparison.csv'));
[~,bestIdx]=min(results.RMSE);
preds={predLinear,predTree,predForest}; bestPred=preds{bestIdx};
fprintf('Best model by test RMSE: %s\n',results.Model{bestIdx});
%% Part F: Prediction plots
f=figure('Visible','off'); scatter(ytest,bestPred,18,'filled'); hold on;
lo=min([ytest;bestPred]); hi=max([ytest;bestPred]); plot([lo hi],[lo hi],'k--','LineWidth',1.5);
xlabel('Actual Strength (MPa)'); ylabel('Predicted Strength (MPa)');
title(['Predicted vs Actual: ',results.Model{bestIdx}]); grid on;
saveas(f,fullfile(figDir,'predicted_vs_actual.png')); close(f);
residual=ytest-bestPred;
f=figure('Visible','off'); scatter(bestPred,residual,18,'filled'); hold on; yline(0,'k--');
xlabel('Predicted Strength (MPa)'); ylabel('Residual: Actual - Predicted (MPa)');
title('Prediction Residuals'); grid on;
saveas(f,fullfile(figDir,'residuals.png')); close(f);
importance=forestModel.OOBPermutedPredictorDeltaError;
f=figure('Visible','off'); bar(importance); set(gca,'XTick',1:8,'XTickLabel',featureNames,'XTickLabelRotation',45);
ylabel('OOB permutation importance'); title('Bagged-Tree Predictor Importance'); grid on;
saveas(f,fullfile(figDir,'feature_importance.png')); close(f);
[~,idx]=sort(importance,'descend');
fprintf('Most important forest features: %s; %s\n',featureNames{idx(1)},featureNames{idx(2)});
fprintf('Mean residual: %.3f MPa (positive = underprediction)\n',mean(residual));
disp('Review plots and model_comparison.csv; describe actual observations in your report.');
%% Local user-defined function (required by assignment)
function [mae,rmse,r2] = regressionMetrics(actual,predicted)
    actual=actual(:); predicted=predicted(:);
    mae=mean(abs(actual-predicted));
    rmse=sqrt(mean((actual-predicted).^2));
    r2=1-sum((actual-predicted).^2)/sum((actual-mean(actual)).^2);
end


