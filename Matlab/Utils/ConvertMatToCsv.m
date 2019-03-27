workingFolder = './20181227_Calibrate_Surface_Data/DLP1/';
for i = 1:5
    fileName = sprintf('HP%d', i)
    FileData = load(strcat(workingFolder, fileName, '.mat'));
    resultFile = strcat(workingFolder, fileName, '.csv');
    csvwrite(resultFile, eval(strcat('FileData.HP', int2str(i))));
end