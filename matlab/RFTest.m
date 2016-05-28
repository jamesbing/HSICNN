function [SuccRatioTr,SuccRatioTe] = RFTest(ResName,NumTree)
    %
    clc;
    fprintf('Resource Filename: %s, Number of Trees: %d\n', ResName, NumTree);
    
    % 读入数据文件
    DataFile = [ResName '.mat'];
    RFFile = [ResName '_RF.mat'];
    TrainResult = [RFFile 'TrainResult.mat']
    TestResult = [RFFile 'TestResult.mat']

    load(DataFile);
    load(RFFile);
    
    % 测试训练样本集合
    SampNum = size(TrD,1);  % 获得样本数目
    SuccNum = 0;
    recgMat = zeros(NumClass);
    for loop1 = 1 : SampNum
        % 获得测试数据
        Sample = TrD(loop1, :);
        CId = CIdTr(loop1);
        %这一块是新加入的：start
        %原数组a[]，假设是一维的。取n个元素。
        rand=randperm(length(binTree));
        index=rand(1:200);
        index=sort(index);
        binTree=binTree(index);
        %这一块是新加入的：end
        % 随机森林给出分类结果
        preLabel = testRF(Sample, binTree, NumClass, NumTree);
        
        if size(preLabel, 2) > 1
            ind = find(preLabel == CId);
            if isscalar(ind)
                recgMat(CId,ind) = recgMat(CId,ind) + 1;
                SuccNum = SuccNum + 1;
                preLabel = ind;
            else
                preLabel = 1000;
            end
        else
            recgMat(CId,preLabel) = recgMat(CId,preLabel) + 1;
            if preLabel == CId
                SuccNum = SuccNum + 1;
            end
        end
        %下面是新加的 start
        trainresult(loop1,1) = CId;
        trainresult(loop1,2) = preLabel;
        save(TrainResult, 'trainresult');
        %上面是新加的 end


        if mod(loop1,100) == 0
            fprintf('Sample No. = %d，Succ. number = %d(%f)\n', loop1, SuccNum, SuccNum/loop1*100);
            trainresult(loop1,3) = SuccNum / loop1 * 100;
            save(TrainResult,'trainresult');
        end
    end
    SuccRatioTr = SuccNum / SampNum;
    fprintf('Succ. ratio of training samples : %f\n', SuccRatioTr);
%     disp(recgMat);
    
    localSum = sum(recgMat,2);
    localRatio = zeros(size(localSum));
    for loop1 = 1 : size(localSum)
        localRatio(loop1) = recgMat(loop1,loop1) / localSum(loop1);
    end
    fprintf('Succ. ratio of training samples per each class:\n');
%     disp(localRatio);
    fprintf('Avg succ. ratio of training samples: %f\n\n', mean(localRatio));
    
    % 测试测试样本集合
    SampNum = size(TeD,1);  % 获得样本数目
    SuccNum = 0;
    recgMat = zeros(NumClass);
    for loop1 = 1 : SampNum
        % 获得测试数据
        Sample = TeD(loop1, :);
        CId = CIdTe(loop1);
        
        % 随机森林给出分类结果
         %这一块是新加入的：start
        %原数组a[]，假设是一维的。取n个元素。
        rand=randperm(length(binTree));
        index=rand(1:200);
        index=sort(index);
        binTree=binTree(index);
        %这一块是新加入的：end
        preLabel = testRF(Sample, binTree, NumClass, NumTree);
 
        if size(preLabel, 2) > 1
            ind = find(preLabel == CId);
            if isscalar(ind)
                recgMat(CId,ind) = recgMat(CId,ind) + 1;
                SuccNum = SuccNum + 1;
                preLabel = ind;
            end
        else
            recgMat(CId,preLabel) = recgMat(CId,preLabel) + 1;
            if preLabel == CId
                SuccNum = SuccNum + 1;
            end
        end
        %下面是新加的  start
        if size(preLabel, 2) > 1 
            preLabel = 1000;
        end
        testingResult(loop1,1) = CId;
        testingResult(loop1,2) = preLabel;
        save(TestResult, 'testingResult');
        %上面是新加的  end
        if mod(loop1, 100) == 0
            fprintf('Sample No. = %d，Succ. number = %d(%f)\n', loop1, SuccNum, SuccNum/loop1*100);
            testingResult(loop1, 3) = SuccNum / 100 * loop1;
            save(TestResult, 'testingResult');
        end
    end
    SuccRatioTe = SuccNum / SampNum;
    fprintf('Succ. ratio of test samples : %f\n', SuccRatioTe);
%     disp(recgMat);
    
    localSum = sum(recgMat,2);
    for loop1 = 1 : size(localSum)
        localRatio(loop1) = recgMat(loop1,loop1) / localSum(loop1);
    end
    fprintf('Succ. ratio of test samples per each class:\n');
%     disp(localRatio);
    fprintf('Avg succ. ratio of test samples: %f\n', mean(localRatio));
end
        
% -------------------------------------
% 测试随机森林
% -------------------------------------
function preLabel = testRF(sample,binTreeSet,ClassNum,TreeNum)
    % 初始化参数
    ResVec = zeros(1, ClassNum);
    
    % 收集每棵决策树对样本的分类结果
    if TreeNum > size(binTreeSet,2)
        TreeNum = size(binTreeSet,2);
    end
%     TreeNum = size(binTreeSet,2);
    for loop1 = 1 : TreeNum
        % 获得决策树
        binTree = binTreeSet{loop1};
    
        % 计算分类结果
        curNodeIndex = 1;
        endFlag = false;
        while ~endFlag
            oneNode = binTree(curNodeIndex);
            feaIndex = oneNode.FeaID;
            feaValue = sample(feaIndex);
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
                % 获得决策结果
                ResVec(curLabel) = ResVec(curLabel) + 1;
                endFlag = true;
            end
        end
    end
    C = max(ResVec);
    preLabel = find(ResVec == C);

%     succFlag = (maxLabel == classID);
end
