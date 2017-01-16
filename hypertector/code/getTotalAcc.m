function [totalAccVector, totalAccMatrix] = getTotalAcc(dataset, pixelStrategy, modelName)
    if pixelStrategy == 1
        path = strcat('new', dataset, '1N', modelName ,'Result.mat');
        last = strcat(modelName ,'Result.mat')
    else
        path = strcat('new', dataset, '1N',num2str(pixelStrategy), modelName ,'Result.mat');
        last = strcat(num2str(pixelStrategy), modelName ,'Result.mat')
    end
    load(path);
    class_number = max(actual);
    totalAccVector = zeros(class_number);
    totalAccMatrix = zeros(class_number, class_number);

    count = 0;
    for mark = [1:30]
        if (mark == 2 || mark == 3 || mark ==14 || mark == 15 || mark ==18 || mark ==19 || mark == 21 || mark == 22 || mark == 23 || mark == 25) == false
            path = strcat('new',dataset ,num2str(mark) ,'N',last);
            temp_a = zeros(class_number);
            temp_c = zeros(class_number, class_number);
            [temp_a, temp_c] = calculateDetailed(path);
            totalAccVector = totalAccVector + temp_a;
            totalAccMatrix = totalAccMatrix + temp_c;
        end
    end
    totalAccVector = (totalAccVector ./ 20);
    totalAccMatrix = (totalAccMatrix ./ 20);
end

%为了画分类准确率柱状图
	
%下面开始是正确的
function [classifyVector, classifyMatrix] = calculateDetailed(path)
load(path);
predictLabel = prediction;
class_number = max(actual);
predictLabel_vector = zeros(class_number);
actualVector = zeros(class_number);
predictLabel_matrix = zeros(class_number,class_number);
for mark = [1:length(predictLabel)]
	if predictLabel(mark) == actual(mark)
		predictLabel_vector(predictLabel(mark)) = predictLabel_vector(predictLabel(mark)) + 1;
		actualVector(actual(mark)) = actualVector(actual(mark)) + 1;
	else
		actualVector(actual(mark)) = actualVector(actual(mark)) + 1;
	end
	predictLabel_matrix(predictLabel(mark),actual(mark)) = predictLabel_matrix(predictLabel(mark),actual(mark)) + 1;
end
classifyVector = zeros(class_number);
for mark = [1:class_number]
    classifyVector(mark) = predictLabel_vector(mark) / actualVector(mark);
end
classifyVector = classifyVector * 100;
for mark = [1:class_number]
    for insideMark = [1:class_number]
        predictLabel_matrix(mark,insideMark) = roundn((predictLabel_matrix(mark,insideMark) ./ actualVector(mark)) * 100,-2);
    end
end
classifyMatrix = predictLabel_matrix'; 
end
