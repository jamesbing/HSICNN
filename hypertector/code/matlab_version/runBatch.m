
matlabpool('open','local',4)
parfor i = 11:15
	str1='newKSC0';
	str1Train='newKSC';
	str2='N.mat';
	str3='NResult';
	str=[str1,num2str(i)];
	strTrain=[str1Train,num2str(i)];
	strResult=[str,str3];
	strTrain=[strTrain,str2];
	HSICNN3(strResult,strTrain,1,4);
end
