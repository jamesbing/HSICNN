function setRF(FileName)
    % 建立随机森林分类器
    % FileName: 特征文件名称（不带扩展名）
    %
    clc;
    
    % 建立数据文件名称
    DataFile = [FileName '.mat'];
    ResFile = [FileName '_RF.mat'];
    
    % 读入特征数据文件
    load(DataFile);
    if (size(TrD,1) ~= size(CIdTr,1)) || (size(TeD,1) ~= size(CIdTe,1))
        fprintf('Error: Different dimensions of training data and its labels.\n');
        return
    end
    
    % 获得特征维数
    NumFeature = size(TrD,2);

    % 获得类别数目
    diffLabel = unique(CIdTr);
    NumClass = size(diffLabel,1);

    % 建立计算参数
    MTry = round(sqrt(NumFeature));   % 随机选择的特征数目。一般选择特征维数的平方根，对决策结果不太敏感。
    NumTree = 500;    % 树分类器数目
    minNodeSize = 5;  % 最小叶节点样本数目

    % 初始化伪随机数发生器
    rng('shuffle');
    
    % 建立记录数组
    fp = fopen(ResFile,'r');
    if fp == -1
        StTree = 1;
        PerfMat = zeros(NumTree,3);
    else
        fclose(fp);
        load(ResFile);
        StTree = size(binTree,2) + 1;
    end

    % 建立随机森林
    for loop = StTree : NumTree
        % 建立随机森林
        fprintf('\n\nNumber of tree = %d\n',loop);
        % 从原始训练样本集合中获得自举样本集合
        [TrainSet,TrainLabel,OOB,OOBLabel] = boostSample(TrD, CIdTr);
    
        % 使用TrainSet建立一棵决策树
        oneTree = setupTree(TrainSet, TrainLabel, MTry, minNodeSize);
        
        % 转换决策树结构
        binTree{loop} = transTree(oneTree);
        clear oneTree;

        % 测试该决策树
        fprintf('\n===== Testing one tree =====\n');
        succRatioTrain = testTree(TrainSet,TrainLabel,binTree{loop});
        fprintf('Success ratio of training samples = %f\n',succRatioTrain * 100);
        PerfMat(loop,1) = succRatioTrain;
        succRatioTest = testTree(OOB,OOBLabel,binTree{loop});
        fprintf('Success ratio of test samples = %f\n',succRatioTest * 100);
        PerfMat(loop,2) = succRatioTest;
        
        % 保存一次结果
        save(ResFile,'binTree','PerfMat','NumClass');
    end
end

% -------------------------------------
% 转换决策树结构
% -------------------------------------
function binTree = transTree(oneTree)
    % 定义决策树结构
    binTree = struct('FeaID', 0, 'Value', 0, 'LLink', 0, 'RLink', 0);
    
    treeLength = size(oneTree,2);
    for loop = 2 : treeLength
        binTree(loop - 1).FeaID = oneTree(loop).FeaID;
        binTree(loop - 1).Value = oneTree(loop).Value;
        if oneTree(loop).LeftLink ~= 0
            binTree(loop - 1).LLink = oneTree(loop).LeftLink - 1;
        else
            binTree(loop - 1).LLink = -oneTree(loop).LeftLabel(1);
        end
        if oneTree(loop).RightLink ~= 0
            binTree(loop - 1).RLink = oneTree(loop).RightLink - 1;
        else
            binTree(loop - 1).RLink = -oneTree(loop).RightLabel(1);
        end
    end
end

% -------------------------------------
% 测试决策树
% -------------------------------------
function succRatio = testTree(TSet,TLabel,binTree)
    % 获得数据集合参数
    Rows = size(TSet,1);
    
    % 初始化参数
    succRatio = 0;
    
    % 测试每个样本且统计结果
    for no = 1 : Rows
        curNodeIndex = 1;
        endFlag = false;
        while ~endFlag
            oneNode = binTree(curNodeIndex);
            feaIndex = oneNode.FeaID;
            feaValue = TSet(no,feaIndex);
            if feaValue <= oneNode.Value
                curNodeIndex = oneNode.LLink;
                if curNodeIndex < 0
                    curLabel = -oneNode.LLink;
                end
            else
                curNodeIndex = oneNode.RLink;
                if curNodeIndex < 0
                    curLabel = -oneNode.RLink;
                end
            end
            if curNodeIndex < 0
                succRatio = succRatio + (curLabel == TLabel(no));
                endFlag = true;
            end
        end
    end
    succRatio = succRatio / Rows;
end

% -------------------------------------
% 建立决策树
% -------------------------------------
function oneTree = setupTree(TrainSet, TrainLabel, MTry, MinNodeSize)
    % 建立一棵决策树
    fprintf('<<<<<<<<<< Setup one tree >>>>>>>>>>\n');
    
    % 定义单棵决策树的结构
    oneTree = struct('FeaID',0,'Value',0,'LeftLink',0,'RightLink',0,...
        'LeftSet',[],'LeftLabel',[],'RightSet',[],'RightLabel',[]);

    % 定义决策树构造变量
    ConsFlag = true;
    curNodeIndex = 1;
    oneTree(curNodeIndex).LeftSet = TrainSet;
    oneTree(curNodeIndex).LeftLabel = TrainLabel;

    % 构造二叉决策树
    while ConsFlag
        % 获得当前节点
        oneNode = oneTree(curNodeIndex);
        fprintf('Current node index = %d\n',curNodeIndex);
    
        % 判断如何处理该节点
        if oneNode.LeftLink == 0
            % 左子节点待处理，检查是否需要处理
            % 查看是否有待处理的数据集
            if size(oneNode.LeftSet,1) ~= 0   %~isempty(oneNode.LeftSet)
                % 查看样本数目是否过小
                if length(oneNode.LeftLabel) <= MinNodeSize
                    % 样本数目过小，则直接定义为叶节点
                    % 寻找不同的标签
                    diffLabel = unique(oneNode.LeftLabel);
                    diffNum = size(diffLabel,1);
                    if diffNum == 1
                        curLabel = diffLabel;
                    else
                        staLabel = hist(oneNode.LeftLabel,diffLabel);
                        [~,index] = max(staLabel);
                        curLabel = diffLabel(index);
                    end
                
                    % 标记该左节点
                    oneNode.LeftSet = [];
                    oneNode.LeftLabel = curLabel;
                    
                    oneTree(curNodeIndex).LeftSet = [];
                    oneTree(curNodeIndex).LeftLabel = curLabel;
                else
                    % 检测是否同样类别样本
                    diffLabel = unique(oneNode.LeftLabel);
                    diffNum = size(diffLabel,1);
                    if diffNum == 1
                        % 标记该左节点
                        oneNode.LeftSet = [];
                        oneNode.LeftLabel = diffLabel;
                    
                        oneTree(curNodeIndex).LeftSet = [];
                        oneTree(curNodeIndex).LeftLabel = diffLabel;
                    else
                        % 非叶节点，则添加新节点
                        TSet = oneNode.LeftSet;
                        TLabel = oneNode.LeftLabel;
                        newNode = addNode(TSet,TLabel,MTry);
                
                        % 获得新节点索引
                        index = size(oneTree,2) + 1;
                        oneNode.LeftLink = index;
                        oneTree(curNodeIndex).LeftLink = index;
                        oneTree(curNodeIndex).LeftSet = [];
                        oneTree(curNodeIndex).LeftLabel = [];
                    
                        oneTree(index) = newNode;
                    end
                end
            end
        end
        
        % 判断右节点是否需要处理
        if oneNode.RightLink == 0
            % 查看是否有待处理的数据集
            if size(oneNode.RightSet,1) ~= 0    %~isempty(oneNode.RightSet)
                % 查看样本数目是否过小
                if length(oneNode.RightLabel) <= MinNodeSize
                    % 样本数目过小，则直接定义为叶节点
                    % 寻找不同的标签
                    diffLabel = unique(oneNode.RightLabel);
                    diffNum = size(diffLabel,1);
                    if diffNum == 1
                        curLabel = diffLabel;
                    else
                        staLabel = hist(oneNode.RightLabel,diffLabel);
                        [~,index] = max(staLabel);
                        curLabel = diffLabel(index);
                    end
                
                    % 标记该右节点
                    oneNode.RightSet = [];
                    oneNode.RightLabel = curLabel;
                    
                    oneTree(curNodeIndex).RightSet = [];
                    oneTree(curNodeIndex).RightLabel = curLabel;
                else
                    % 检测是否同样类别样本
                    diffLabel = unique(oneNode.RightLabel);
                    diffNum = size(diffLabel,1);
                    if diffNum == 1
                        % 标记该右节点
                        oneNode.RightSet = [];
                        oneNode.RightLabel = diffLabel;
                    
                        oneTree(curNodeIndex).RightSet = [];
                        oneTree(curNodeIndex).RightLabel = diffLabel;
                    else
                        % 非叶节点，则添加新节点
                        TSet = oneNode.RightSet;
                        TLabel = oneNode.RightLabel;
                        newNode = addNode(TSet,TLabel,MTry);
                
                        % 获得新节点索引
                        index = size(oneTree,2) + 1;
                        oneNode.RightLink = index;
                        oneTree(curNodeIndex).RightLink = index;
                        oneTree(curNodeIndex).RightSet = [];
                        oneTree(curNodeIndex).RightLabel = [];
                    
                        oneTree(index) = newNode;
                    end
                end
            end
        end
    
        % 继续处理剩余树节点
        treeLength = size(oneTree,2);
        curNodeIndex = curNodeIndex + 1;
        if curNodeIndex > treeLength
            ConsFlag = false;
        end
    end
end

% -------------------------------------
% 添加一个新节点
% -------------------------------------
function newNode = addNode(TSet,TLabel,MTry)
    % 添加一个新节点
%     fprintf('----> Add new node\n');
    
    %定义新节点数据结构
    newNode = struct('FeaID',0,'Value',0,'LeftLink',0,'RightLink',0,...
        'LeftSet',[],'LeftLabel',[],'RightSet',[],'RightLabel',[]);

    % 随机选择MTry个决策特征
    [Rows,Cols] = size(TSet);
    fprintf('----> Add new node (%d)\n',Rows);
    
    % 测试所选特征是否具有不同的取值
    OKFlag = true;
    while OKFlag
%         selFeat = randFeat(Cols,MTry);
        selFeat = randperm(Cols,MTry);
        for feaIndex = 1 : MTry
            feaNo = selFeat(feaIndex);
            feaValue = TSet(:,feaNo);
            feaThreshold = unique(feaValue);
            numThreshold = size(feaThreshold,1) - 1;
            if numThreshold ~= 0
                OKFlag = false;
                break
            else
                fprintf('Failed selected features.\n');
            end
        end
    end

    % 初始化最佳结果变量
    optEpy = -1;
    optTh = 0;
    optFea = 0;
    
    % 计算样本集合的类别分布
    dLabel = unique(TLabel);
    maxLabel = max(dLabel);
    labelHist = hist(TLabel,1 : maxLabel);
    sampleNum = length(TLabel);
    labelH = labelHist(labelHist > 0) ./ sampleNum; 
    preEntropy = -sum(labelH .* log2(labelH));

    % 对每个特征循环
    for feaIndex = 1 : MTry
        % 计算训练样本集合中该特征的所有取值
        feaNo = selFeat(feaIndex);
        feaValue = TSet(:,feaNo);
        feaThreshold = unique(feaValue);
        numThreshold = size(feaThreshold,1) - 1;
        
        % 计算各区间的样本数目
        for threIndex = 1 : numThreshold
            thre = feaThreshold(threIndex);
            leftLabel = TLabel(feaValue <= thre);
            leftHist = hist(leftLabel,1 : maxLabel);
            rightHist = labelHist - leftHist;
            
            LNum = sum(leftHist);
            RNum = sampleNum - LNum;
            
            % 选取大于零的有效项
            leftH = leftHist(leftHist > 0) ./ LNum;
            rightH = rightHist(rightHist > 0) ./ RNum;
            
            leftEpy = -sum(leftH .* log2(leftH));
            rightEpy = -sum(rightH .* log2(rightH));
            
            epyReduce = preEntropy - LNum / sampleNum * leftEpy -...
                RNum / sampleNum * rightEpy;
            
            if epyReduce > optEpy
                optEpy = epyReduce;
                optTh = thre;
                optFea = feaNo;
            end
        end
    end

    % 获得节点数据
    feaValue = TSet(:,optFea);
    leftB = find(feaValue <= optTh);
    rightB = find(feaValue > optTh);
        
    if optEpy == -1
        fprintf('Error.\n');
    end
    
    % 保存计算结果，返回。
    newNode.FeaID = optFea;
    newNode.Value = optTh;
    newNode.LeftLink = 0;
    newNode.RightLink = 0;
    newNode.LeftSet = TSet(leftB,:);   %LSet;
    newNode.RightSet = TSet(rightB,:); %RSet;
    newNode.LeftLabel = TLabel(leftB); %LLabel;
    newNode.RightLabel = TLabel(rightB);   %RLabel;
end

% -------------------------------------
% 获得自举样本集合和袋外样本集合
% -------------------------------------
function [TrainSet,TrainLabel,OOB,OOBLabel] = boostSample(DataSet,LabelSet)
    % 获得均匀分布的随机数序列
    Rows = size(DataSet,1);
    selIndex = random('unid',Rows,[Rows,1]);
    
    % 抽取自举训练样本集合
    TrainSet = DataSet(selIndex,:);
    TrainLabel = LabelSet(selIndex);

    % 获得袋外样本索引
    dS = unique(selIndex);
    Ob = setdiff(1:Rows,dS);
    OOB = DataSet(Ob,:);
    OOBLabel = LabelSet(Ob);
end
