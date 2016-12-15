%用于从原始文件中提取高光谱数据集
function sampDS(resName, SetNum)
% 将数据集合按照indexName文件要求分成训练、验证和测试三个数据集合
% 数据集合抽取模式按照：
%   （１）单像素样本
%   （２）4邻域像素样本
%   （３）8邻域像素样本

fprintf('Name: %s, Num: %s\n', resName, SetNum);
% 输入高光谱图像数据：DataSet
DataName = [resName 'Data.mat'];
load(DataName);

% 输入类别标签数据：ClsID
ClsFilename = [resName 'Gt.mat'];
load(ClsFilename);

% 输入分裂数据集合模式：ClsIndex
IndexFilename = [resName 'Index.mat'];
load(IndexFilename);

% 获得三个数据集合的样本分裂索引号
[IdxTr,IdxVa,IdxTe] = splitIdx(ClsIndex);

% 获得类别数目和频带数目
clsNum = size(ClsIndex,1);
[ROW,COL,bandNum] = size(DataSet);

%--------------------------------------
% 从高光谱图像中提取每种类别的像素样本
%--------------------------------------
tmpData = zeros(1,bandNum); % 光谱数据
tmpCId = zeros(1,1);    % 类别标签
tmpPos = zeros(1,2);    % 像素位置
sampleIdx = 0;
fprintf('One pixel sample set\n');
for clsNo = 1 : clsNum
    fprintf('No. of Class: %d\n',clsNo);
    
    % 找出类别标号相同的像素位置
    [row,col,v] = find(ClsID == clsNo);
    sampleNum = size(v,1);
    
    for sampleNo = 1 : sampleNum
        sampleIdx = sampleIdx + 1;
        tmpData(sampleIdx,:) = DataSet(row(sampleNo),col(sampleNo),1:bandNum);
        tmpCId(sampleIdx) = clsNo;
        tmpPos(sampleIdx,:) = [row(sampleNo),col(sampleNo)];
    end
end

% 规格化像素值为零均值单位方差
norV = mapstd(tmpData');
tmpData = norV';

% 获得三个数据集合
tmpD = tmpData(IdxTr,:);
tmpC = tmpCId(IdxTr);
tmpP = tmpPos(IdxTr,:);
row = size(tmpD,1);
r = randperm(row);  % 将训练样本的顺序打乱，便于训练。
DataTr = tmpD(r,:);
CIdTr = tmpC(r);
PosTr = tmpP(r,:);

DataVa = tmpData(IdxVa,:);
CIdVa = tmpCId(IdxVa);
PosVa = tmpPos(IdxVa,:);
DataTe = tmpData(IdxTe,:);
CIdTe = tmpCId(IdxTe);
PosTe = tmpPos(IdxTe,:);

filename = ['new' resName SetNum 'N.mat'];
save(filename,'DataTr','CIdTr','PosTr','DataVa','CIdVa','PosVa','DataTe','CIdTe','PosTe');

clear tmpD tmpC tmpP r tmpData tmpCId tmpPos DataTr CIdTr PosTr DataVa CIdVa PosVa DataTe CIdTe PosTe

%--------------------------------------
% 从高光谱图像中提取每种类别的4邻域像素样本
%--------------------------------------
tmpData = zeros(1,bandNum * 5);
tmpCId = zeros(1,1);
tmpPos = zeros(1,2);
sampleIdx = 0;
fprintf('\n4-neighbor pixel sample set\n');
for clsNo = 1 : clsNum
    fprintf('No. of Class: %d\n',clsNo);
    
    % 找出类别标号相同的像素位置
    [row,col,v] = find(ClsID == clsNo);
    sampleNum = size(v,1);
    
    tmp = zeros(5,bandNum);
    for sampleNo = 1 : sampleNum
        sampleIdx = sampleIdx + 1;

        if row(sampleNo) == 1
            tmp(1,:) = DataSet(1,col(sampleNo),:);
        else
            tmp(1,:) = DataSet(row(sampleNo) - 1,col(sampleNo),:);
        end
        
        if col(sampleNo) == 1
            tmp(2,:) = DataSet(row(sampleNo),1,:);
        else
            tmp(2,:) = DataSet(row(sampleNo),col(sampleNo) - 1,:);
        end
        
        tmp(3,:) = DataSet(row(sampleNo),col(sampleNo),:);
        
        if col(sampleNo) == COL
            tmp(4,:) = DataSet(row(sampleNo),col(sampleNo),:);
        else
            tmp(4,:) = DataSet(row(sampleNo),col(sampleNo) + 1,:);
        end
        
        if row(sampleNo) == ROW
            tmp(5,:) = DataSet(row(sampleNo),col(sampleNo),:);
        else
            tmp(5,:) = DataSet(row(sampleNo) + 1,col(sampleNo),:);
        end
        
        tmpData(sampleIdx,:) = reshape(tmp',1,[]);
        tmpCId(sampleIdx) = clsNo;
        tmpPos(sampleIdx,:) = [row(sampleNo), col(sampleNo)];
    end
end

% 规格化像素值为零均值单位方差
norV = mapstd(tmpData');
tmpData = norV';

% 获得三个数据集合
tmpD = tmpData(IdxTr,:);
tmpC = tmpCId(IdxTr);
tmpP = tmpPos(IdxTr,:);
row = size(tmpD,1);
r = randperm(row);
DataTr = tmpD(r,:);
CIdTr = tmpC(r);
PosTr = tmpP(r,:);

DataVa = tmpData(IdxVa,:);
CIdVa = tmpCId(IdxVa);
PosVa = tmpPos(IdxVa,:);
DataTe = tmpData(IdxTe,:);
CIdTe = tmpCId(IdxTe);
PosTe = tmpPos(IdxTe,:);

filename = ['new' resName SetNum 'N4.mat'];
save(filename,'DataTr','CIdTr','PosTr','DataVa','CIdVa','PosVa','DataTe','CIdTe','PosTe');

clear tmpD tmpC tmpP r tmpData tmpCId temPos DataTr CIdTr PosTr DataVa CIdVa PosVa DataTe CIdTe PosTe


%--------------------------------------
% 从高光谱图像中提取每种类别的8邻域像素样本
%--------------------------------------
tmpData = zeros(1,bandNum * 9);
tmpCId = zeros(1,1);
tmpPos = zeros(1,2);
sampleIdx = 0;
fprintf('\n8-neighbor pixel sample set\n');
for clsNo = 1 : clsNum
    fprintf('No. of Class: %d\n',clsNo);
    
    % 找出类别标号相同的像素位置
    [row,col,v] = find(ClsID == clsNo);
    sampleNum = size(v,1);
    
    tmp = zeros(9,bandNum);
    for sampleNo = 1 : sampleNum
        sampleIdx = sampleIdx + 1;
        
        r = row(sampleNo);
        c = col(sampleNo);
        
        if r == 1
            if c == 1
                tmp(1,:) = DataSet(r,c,:);
                tmp(2,:) = DataSet(r,c,:);
                tmp(3,:) = DataSet(r,c + 1,:);
                tmp(4,:) = DataSet(r,c,:);
                tmp(5,:) = DataSet(r,c,:);
                tmp(6,:) = DataSet(r,c + 1,:);
                tmp(7,:) = DataSet(r + 1,c,:);
                tmp(8,:) = DataSet(r + 1,c,:);
                tmp(9,:) = DataSet(r + 1,c + 1,:);
            elseif c == COL
                tmp(1,:) = DataSet(r,c - 1,:);
                tmp(2,:) = DataSet(r,c,:);
                tmp(3,:) = DataSet(r,c,:);
                tmp(4,:) = DataSet(r,c - 1,:);
                tmp(5,:) = DataSet(r,c,:);
                tmp(6,:) = DataSet(r,c,:);
                tmp(7,:) = DataSet(r + 1,c - 1,:);
                tmp(8,:) = DataSet(r + 1,c,:);
                tmp(9,:) = DataSet(r + 1,c,:);
            else
                tmp(1,:) = DataSet(r,c - 1,:);
                tmp(2,:) = DataSet(r,c,:);
                tmp(3,:) = DataSet(r,c + 1,:);
                tmp(4,:) = DataSet(r,c - 1,:);
                tmp(5,:) = DataSet(r,c,:);
                tmp(6,:) = DataSet(r,c + 1,:);
                tmp(7,:) = DataSet(r + 1,c - 1,:);
                tmp(8,:) = DataSet(r + 1,c,:);
                tmp(9,:) = DataSet(r + 1,c + 1,:);
            end
        elseif r == ROW
            if c == 1
                tmp(1,:) = DataSet(r - 1,c,:);
                tmp(2,:) = DataSet(r - 1,c,:);
                tmp(3,:) = DataSet(r - 1,c + 1,:);
                tmp(4,:) = DataSet(r,c,:);
                tmp(5,:) = DataSet(r,c,:);
                tmp(6,:) = DataSet(r,c + 1,:);
                tmp(7,:) = DataSet(r,c,:);
                tmp(8,:) = DataSet(r,c,:);
                tmp(9,:) = DataSet(r,c + 1,:);
            elseif c == COL
                tmp(1,:) = DataSet(r - 1,c - 1,:);
                tmp(2,:) = DataSet(r - 1,c,:);
                tmp(3,:) = DataSet(r - 1,c,:);
                tmp(4,:) = DataSet(r,c - 1,:);
                tmp(5,:) = DataSet(r,c,:);
                tmp(6,:) = DataSet(r,c,:);
                tmp(7,:) = DataSet(r,c - 1,:);
                tmp(8,:) = DataSet(r,c,:);
                tmp(9,:) = DataSet(r,c,:);
            else
                tmp(1,:) = DataSet(r - 1,c - 1,:);
                tmp(2,:) = DataSet(r - 1,c,:);
                tmp(3,:) = DataSet(r - 1,c + 1,:);
                tmp(4,:) = DataSet(r,c - 1,:);
                tmp(5,:) = DataSet(r,c,:);
                tmp(6,:) = DataSet(r,c + 1,:);
                tmp(7,:) = DataSet(r,c - 1,:);
                tmp(8,:) = DataSet(r,c,:);
                tmp(9,:) = DataSet(r,c + 1,:);
            end
        else
            if c == 1
                tmp(1,:) = DataSet(r - 1,c,:);
                tmp(2,:) = DataSet(r - 1,c,:);
                tmp(3,:) = DataSet(r - 1,c + 1,:);
                tmp(4,:) = DataSet(r,c,:);
                tmp(5,:) = DataSet(r,c,:);
                tmp(6,:) = DataSet(r,c + 1,:);
                tmp(7,:) = DataSet(r + 1,c,:);
                tmp(8,:) = DataSet(r + 1,c,:);
                tmp(9,:) = DataSet(r + 1,c + 1,:);
            elseif c == COL
                tmp(1,:) = DataSet(r - 1,c - 1,:);
                tmp(2,:) = DataSet(r - 1,c,:);
                tmp(3,:) = DataSet(r - 1,c,:);
                tmp(4,:) = DataSet(r,c - 1,:);
                tmp(5,:) = DataSet(r,c,:);
                tmp(6,:) = DataSet(r,c,:);
                tmp(7,:) = DataSet(r + 1,c - 1,:);
                tmp(8,:) = DataSet(r + 1,c,:);
                tmp(9,:) = DataSet(r + 1,c,:);
            else
                tmp(1,:) = DataSet(r - 1,c - 1,:);
                tmp(2,:) = DataSet(r - 1,c,:);
                tmp(3,:) = DataSet(r - 1,c + 1,:);
                tmp(4,:) = DataSet(r,c - 1,:);
                tmp(5,:) = DataSet(r,c,:);
                tmp(6,:) = DataSet(r,c + 1,:);
                tmp(7,:) = DataSet(r + 1,c - 1,:);
                tmp(8,:) = DataSet(r + 1,c,:);
                tmp(9,:) = DataSet(r + 1,c + 1,:);
            end
        end
        tmpData(sampleIdx,:) = reshape(tmp',1,[]);
        tmpCId(sampleIdx) = clsNo;
        tmpPos(sampleIdx,:) = [row(sampleNo), col(sampleNo)];
    end
end

% 规格化像素值为零均值单位方差
norV = mapstd(tmpData');
tmpData = norV';

% 获得三个数据集合
tmpD = tmpData(IdxTr,:);
tmpC = tmpCId(IdxTr);
tmpP = tmpPos(IdxTr,:);
row = size(tmpD,1);
r = randperm(row);
DataTr = tmpD(r,:);
CIdTr = tmpC(r);
PosTr = tmpP(r,:);

DataVa = tmpData(IdxVa,:);
CIdVa = tmpCId(IdxVa);
PosVa = tmpPos(IdxVa,:);
DataTe = tmpData(IdxTe,:);
CIdTe = tmpCId(IdxTe);
PosTe = tmpPos(IdxTe,:);

filename = ['new' resName SetNum 'N8.mat'];
save(filename,'DataTr','CIdTr','PosTr','DataVa','CIdVa','PosVa','DataTe','CIdTe','PosTe');

end

function [IdxTr,IdxVa,IdxTe] = splitIdx(ClsIndex)
% 样本总数
sampleClsNum = sum(ClsIndex,2);
clsNum = size(ClsIndex,1);

% 初始化随机数种子
rng('shuffle');

totalTr = 0;
totalVa = 0;
totalTe = 0;
IdxTotal = 0;
% 按照ClsIndex要求，生成随机分裂样本索引
for loop1 = 1 : clsNum
    numTr = ClsIndex(loop1,1);
    numVa = ClsIndex(loop1,2);
    numTe = ClsIndex(loop1,3);
    
    % 计算样本类内索引
    idx = 1 : sampleClsNum(loop1);
    num = size(idx,2);

    % 获得训练样本随机索引
    s = randperm(num, numTr);
    IdxTr(totalTr + 1 : totalTr + numTr) = IdxTotal + idx(s);
    totalTr = totalTr + numTr;
    
    % 获得验证样本随机索引
    idx = setdiff(idx,idx(s));
    num = size(idx,2);
    s = randperm(num, numVa);
    IdxVa(totalVa + 1 : totalVa + numVa) = IdxTotal + idx(s);
    totalVa = totalVa + numVa;
    
    % 获得测试样本随机索引
    idx = setdiff(idx,idx(s));
    IdxTe(totalTe + 1 : totalTe + numTe) = IdxTotal + idx;
    totalTe = totalTe + numTe;
    
    % 计算样本总数
    IdxTotal = IdxTotal + sampleClsNum(loop1);
end
end
