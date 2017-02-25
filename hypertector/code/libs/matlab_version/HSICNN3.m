function perfMat = HSICNN3(ResName,DataName,ParamStep,ParamN3)
% 训练多光谱CNN网络
% ResName：输出结果文件
% DataName：输入数据文件
% ParamStep：卷积窗口跳步
% ParamN3：第三层节点数目/10

% 装载训练数据
load(DataName);

% 获得输入节点数目和输出节点数目
n1 = size(DataTr,2);
n5 = max(CIdTr);    % 样本类别数目

% 初始化运行参数
alpha = 0.01;   % 学习速率
IterMax = 100000;   % 最大迭代次数
ErrMin = 0.00001;   % 最小误差
Batches = 1000;     % 训练批次
BatchSize = 10;    % 批次大小

% 初始化CNN网络
[cnnModel, perfMat, k2] = InitCNNModel(n1,n5,ResName,ParamStep,ParamN3);
LenPerf = size(perfMat, 1);
if (LenPerf == 1) && (perfMat(1,1) == 0)
    StBatch = 1;
else
    StBatch = LenPerf + 1;
end

% 设置训练参数
iter = 0;   % 迭代次数
err = realmax;  % 误差

% 迭代训练
while (err > ErrMin) && (iter < IterMax)
    err = 0;
    
    for batch = StBatch : Batches
        fprintf('Batch = %d :',batch);
        [cnnModel, J] = trainCNNModel(cnnModel, DataTr, CIdTr, k2, n5, alpha, ParamStep);

        if isnan(J)
            fprintf('Error at batch = %d.\n',batch);
            break;
        end
        
        err = err + mean(J);
        perfMat(batch, 1) = J;
        
        if mod(batch, 10) == 0
            fprintf('Testing train data ');
            ratioTr = testCNNModel(cnnModel,DataTr,CIdTr,k2, ParamStep);
            fprintf('OK. Correct ratio = %f\n',ratioTr);

            fprintf('Testing test data ');
            ratioTe = testCNNModel(cnnModel,DataVa,CIdVa,k2, ParamStep);
            fprintf('OK. Correct ratio = %f\n\n',ratioTe);

            perfMat(batch,2) = ratioTr;
            perfMat(batch,3) = ratioTe;
        else
            perfMat(batch,2) = 0;
            perfMat(batch,3) = 0;
        end
        save(ResName,'cnnModel','perfMat');
    end
    
    break
    err = err / Batches;
    iter = iter + 1;
end
fprintf('OK.\n');
% save(ResName,'cnnModel','J','ratioTr','ratioV','ratioTe');
end

function ratio = testCNNModel(cnnModel,Data,Label,k2, ParamStep)
% 测试分类结果
sampleNum = size(Data,1);
correct = 0;
step = ceil(sampleNum / 60);
for loop1 = 1 : sampleNum
    if mod(loop1,step) == 0
        fprintf('.');
    end
    signal = Data(loop1,:);
    cnnModel = forwardCNNModel(cnnModel,signal,k2, ParamStep);
    pred = cnnModel(4).Neurons;
    [~,predLabel] = max(pred);
    labelIdea = Label(loop1);
    if predLabel == labelIdea
        correct = correct + 1;
    end
end
ratio = correct / sampleNum;
end


function [cnnModel, J] = trainCNNModel(cnnModel, TrainD, TrainL, k2, n5, alpha, ParamStep)
% 传输训练样本
J = 0;
sampleNum = size(TrainD,1);
step = ceil(sampleNum / 60);
for loop1 = 1 : sampleNum
    if mod(loop1,step) == 0
        fprintf('.');
    end
    signal = TrainD(loop1,:);
    % 传播网络信号
    cnnModel = forwardCNNModel(cnnModel, signal, k2, ParamStep);
    classIdea = cnnModel(4).Neurons;

    label = TrainL(loop1);
    J = J + log(classIdea(label));
    
    labelVec = zeros(1, n5);
    labelVec(label) = 1;
    % 调整网络参数
    cnnModel = backwardCNNModel(cnnModel, signal, labelVec, alpha, ParamStep);
end
J = -J / sampleNum;
    
fprintf('OK. J = %f\n',J);
end

function cnnModel = backwardCNNModel(cnnModel, sample, label, alpha, ParamStep)
% 调整连接权值
cnnModelBak = cnnModel;

% 第四层传播
labelPret = cnnModelBak(4).Neurons;
deltaY = label - labelPret;
delta4 = -deltaY .* labelPret .* (1 - labelPret);
x4 = cnnModelBak(3).Neurons';
cnnModel(4).Params = cnnModelBak(4).Params - [x4;1] * delta4 .* alpha;

% 第三层传播
W = cnnModelBak(4).Params(1:end-1,:);
delta3 = (W * delta4') .* (1 - x4) .* (1 + x4);
x3 = [reshape(cnnModelBak(2).Neurons,1,[]) 1];
cnnModel(3).Params = cnnModelBak(3).Params - x3' * delta3' .* alpha;

% 第二层传播
W = cnnModelBak(3).Params(1:end-1,:);
delta2 = reshape(W * delta3,20,[]);

% 第一层传播
n2 = cnnModel(2).InputSize;
delta = zeros(20,n2);
% 填充最大值对应误差项
for loop1 = 1 : 20
    pos = find(cnnModel(2).MaxPos(loop1,:) ~= 0);
    idx = cnnModel(2).MaxPos(loop1,pos);
    delta(loop1,idx) = delta2(loop1,pos);
end
x1 = cnnModelBak(1).Neurons;
delta1 = delta .* (1 - x1) .* (1 + x1);


k1 = size(cnnModelBak(1).Params,2) - 1;
for loop1 = 1 : 20
    delta = zeros(1,k1+1);
    pos = find(delta1(loop1,:) ~= 0);
    count = size(pos,2);
    for loop2 = 1 : count
        curPos = pos(loop2);
        stPos = (curPos - 1) * ParamStep + 1;
        x = [sample(stPos : stPos + k1 - 1) 1];
        delta = delta + delta1(loop1,curPos) .* x;
    end
    if count ~= 0
        delta = delta ./ count;
    end
    cnnModel(1).Params(loop1,:) = cnnModelBak(1).Params(loop1,:) - alpha .* delta;
end

end

function cnnModel = forwardCNNModel(cnnModel, sample, k2, ParamStep)
% 信号在ＣＮＮ模型中向前传播

% 第一层传输
kernel = cnnModel(1).Params;
kernelSize = size(kernel,2);
signal = ones(kernelSize,cnnModel(2).InputSize);
for loop1 = 1 : cnnModel(2).InputSize
    stPos = (loop1 - 1) * ParamStep + 1;
    signal(1 : kernelSize - 1, loop1) = sample(stPos : stPos + kernelSize - 2);
end
partS = kernel * signal;
cnnModel(1).Neurons = tansig(partS);    % 计算sigmoid函数

% 第二层传输
n2 = cnnModel(2).InputSize;
stepWin = 1 : k2 : n2;
stepNum = size(stepWin,2);
for loop1 = 1 : stepNum
    stCol = stepWin(loop1);
    enCol = stCol + k2 - 1;
    if enCol > n2
        enCol = n2;
    end
    signal = cnnModel(1).Neurons(:,stCol:enCol);
    [C,I] = max(signal,[],2);
    cnnModel(2).Neurons(:,loop1) = C;
    cnnModel(2).MaxPos(:,loop1) = I + stCol - 1;
end

% 第三层传输
signal = [reshape(cnnModel(2).Neurons,1,[]) 1];
partS = signal * cnnModel(3).Params;
cnnModel(3).Neurons = tansig(partS);

% 第四层传输
signal = [cnnModel(3).Neurons 1];
partS = signal * cnnModel(4).Params;
cnnModel(4).Neurons = exp(partS);
factor = sum(cnnModel(4).Neurons);
cnnModel(4).Neurons = cnnModel(4).Neurons ./ factor;
%classOut = cnnModel(4).Neurons;
end

function [cnnModel, perfMat, k2] = InitCNNModel(n1, n5, ResName, ParamStep, ParamN3)
% 根据n1和n5计算其它参数
k1 = ceil(n1 / ParamStep / 9) * ParamStep;
n2 = (n1 - k1) / ParamStep + 1;
n3 = 10 * ParamN3;    % 30<= n3 <= 40
k2 = ceil(n2 / n3);
n4 = 100;

% 如果存在中间结果，则读入中间结果。
fid = fopen(ResName,'r');
if fid == -1
    % 没有中间结果文件
    % 定义网络模型
    cnnModel = struct('Type',{'convolution','max-pooling','fully-connected','fully-connected'},...
        'Activation',{'tanh','max','tanh','softmax'},...
        'InputSize',{n1,n2,n3,n4},...
        'Neurons',{zeros(20,n2),zeros(20,n3),zeros(1,n4),zeros(1,n5)},...
        'Params',{zeros(20,k1 + 1),0,zeros(20*n3 + 1,n4), zeros(n4 + 1,n5)},...
        'MaxPos',{[],zeros(20,n3),[],[]});


    % 初始化随机数种子
    rng('shuffle');

    % 初始化权值
    cnnModel(1).Params = (rand(20,k1 + 1) - 0.5) / 10;
    cnnModel(3).Params = (rand(20*n3 + 1, n4) - 0.5) / 10;
    cnnModel(4).Params = (rand(n4 + 1, n5) - 0.5) / 10;
    
    perfMat = zeros(1, 3);
else
    fclose(fid);
    load(ResName);
end

end