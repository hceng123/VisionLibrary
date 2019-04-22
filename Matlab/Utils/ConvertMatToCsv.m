workingFolder = './';
for i = 0:5
    fileName = sprintf('HP%d', i)
    FileData = load(strcat(workingFolder, fileName, '.mat'));
    resultFile = strcat(workingFolder, fileName, '.csv');
    csvwrite(resultFile, eval(strcat('FileData.HP', int2str(i))));
end

for i = 1:5
    fileName = sprintf('HN%d', i)
    FileData = load(strcat(workingFolder, fileName, '.mat'));
    resultFile = strcat(workingFolder, fileName, '.csv');
    csvwrite(resultFile, eval(strcat('FileData.HN', int2str(i))));
end