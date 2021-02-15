%% Za³adowanie i wizualizacja danych
clear all
clc

data = importdata('A_train_14.txt');
[dataTrain(1,:), mX, sX] = zscore([data.data(1:end/2,1);data.data(1:end/2,2)]);
[dataTrain(2,:), mY, sY] = zscore([data.data(end/2+1:end,1);data.data(end/2+1:end,2)]);
dataTrain = sortrows(dataTrain');
dataTrainX = dataTrain(:,1);
dataTrainY = dataTrain(:,2);

figure
plot(dataTrainX,dataTrainY,'r.')
xlabel('X')
ylabel('Y')
title('dane trenuj¹ce')

data = importdata('A_test_14.txt');
dataTest(1,:) = ([data.data(1:end/2,1);data.data(1:end/2,2)]-mX)/sX;
dataTest(2,:) = ([data.data(end/2+1:end,1);data.data(end/2+1:end,2)]-mY)/sY;
dataTest = sortrows(dataTest');
dataTestX = dataTest(:,1);
dataTestY = dataTest(:,2);

figure
plot(dataTestX,dataTestY,'b.')
xlabel('X')
ylabel('Y')
title('dane testowe')

%% Utworzenie i trenowanie sieci, liczenie b³êdu œredniokwadratowego oraz leave one out
mseTest = zeros(1,15);
mseTrain = zeros(1,15);
stdHkk = zeros(1,15);
for k = 1:50
    rng('default')
    rng(k);
    for i = 1:15
        % utworzenie i trenowanie sieci
        net = NewNet(dataTrainX,dataTrainY,i);
        yTest = net(dataTestX');
        yTrain = net(dataTrainX');
        
        % MSE
        mseTest(i) = mseTest(i) + sum((yTest-dataTestY').^2)/numel(yTest);
        mseTrain(i) = mseTrain(i) +  sum((yTrain-dataTrainY').^2)/numel(yTrain);
        
        Z = fpderiv('de_dwb',net,dataTrainX',dataTrainY');
        Z=Z';
        H = Z*inv(Z'*Z)*Z';
        if(ceil(rank(Z))-rank(Z)==0)
            for j = 1:200
                h(j) = H(j,j);
            end
        end
        figure(i)
        plot(dataTrainX,yTrain,dataTrainX,dataTrainY,'.');
        title(strcat('Liczba ukrytych neuronów: ',num2str(i)))
        % leave one out
        
        stdHkk(i) = stdHkk(i) + std(h);
        
%         figure(15+i)
%         plot(dataTrainX,h,dataTrainX, ones(numel(dataTrainX),1)*size(Z,2)/numel(dataTrainX))
%         title(strcat('Liczba ukrytych neuronów: ',num2str(i)))
%         xlabel('X')
%         ylabel('hkk')
        
        H=[];
        Z=[];
    end    
end
rng('default')
figure
plot(1:15,mseTest/50,1:15,mseTrain/50)
xlabel('liczba neuronów ukrytych')
ylabel('MSE')
title('B³¹d œredniokwadratowy')
hold on
[mT,iT] = min(mseTest);
[mTr,iTr] = min(mseTrain);
plot(iT,mT/50,'b.',iTr,mTr/50,'r.')
legend('test','train','min test','min train')
figure
plot(stdHkk/50)
xlabel('liczba neuronów ukrytych')
ylabel('std(hkk)')


%% Ep i u
net =[];
wybraneHN = 7;
yTest = [];
yTrain = [];
for i = 1:50
    net{i} = NewNet(dataTrainX,dataTrainY,wybraneHN);
    yTest(:,i) = net{i}(dataTestX');
    yTrain(:,i) = net{i}(dataTrainX');
    Z = fpderiv('de_dwb',net{i},dataTrainX',dataTrainY');
    Z=Z';
    H = Z*inv(Z'*Z)*Z';
    if(ceil(rank(Z))-rank(Z)==0)
        for j = 1:200
            h(j) = H(j,j);
        end
    end
    r = dataTrainY-yTrain(:,i);
    r_k = r'./(1-h);
    
    Ep(i) = sqrt(1/200*sum((r_k).^2));
    u(i) = 1/200*sum(sqrt(200/size(Z,2)*h));
end

figure
plot(Ep,u,'.')
xlabel('Ep')
ylabel('u')
title(strcat('Liczba ukrytych neuronów: ',num2str(wybraneHN)))
%% przedzia³ ufnoœci

q = size(Z,2);
[m,ii] = min(Ep)
yTest = net{ii}(dataTestX');
yTrain = net{ii}(dataTrainX');
Z = fpderiv('de_dwb',net{ii},dataTrainX',dataTrainY');
Z=Z';
H = Z*inv(Z'*Z)*Z';
if(ceil(rank(Z))-rank(Z)==0)
    for j = 1:200
        h(j) = H(j,j);
    end
end
u = 1/200*sum(sqrt(200/size(Z,2)*h));

s = sqrt(1/(200-q)*sum((dataTrainY-yTrain').^2));
t = (mean(dataTrainX)-u)/std(dataTrainX)*sqrt(199);
T = t.*s./sqrt(199);
figure
plot(dataTrainX,yTrain,dataTrainX,yTrain+T,dataTrainX,yTrain-T,dataTrainX,dataTrainY,'.');
xlabel('X')
ylabel('Y')
%%
yTest = net{ii}(dataTestX');
figure
plot(dataTestX,dataTestY,'.',dataTestX,yTest,'Linewidth',2);
xlabel('X')
ylabel('Y')
MSE = sum((yTest-dataTestY').^2)/numel(yTest)
%%

for i=1:15
p = polyfit(dataTrainX,dataTrainY,i);
yTest = polyval(p,dataTestX);
yTrain = polyval(p,dataTrainX);
mseTestP(i) = sum((yTest-dataTestY).^2)/numel(yTest);
mseTrainP(i) =sum((yTrain-dataTrainY).^2)/numel(yTrain);
% figure
% plot(dataTestX,dataTestY,'.',dataTestX,yTest,'Linewidth',2)
end

figure
plot([1:15],mseTestP,[1:15],mseTrainP)
xlabel('stopieñ wielomianu')
ylabel('MSE')