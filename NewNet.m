function net = NewNet(dataTrainX,dataTrainY,hiddenSizes)

net = feedforwardnet(hiddenSizes,'traingd');
rand('state',sum(100*clock)); %inicjalizacja generatora liczb
%pseudolosowych
net=init(net); %inicjalizacja wag sieci
net.trainParam.goal = 0.001; %warunek stopu � poziom b��du
net.trainParam.epochs = 100; %maksymalna liczba epok
net.trainParam.showWindow = false;
net.layers{1}.transferFcn = 'tansig'; %ustawienie funkcji aktywacji neuron�w ukrytych na tansig
net.layers{2}.transferFcn = 'purelin'; %ustawienie funkjci aktywacji neuronu wyj�ciowego na liniow�

net=train(net,dataTrainX',dataTrainY'); %uczenie sieci
%zmiana funkcji ucz�cej na: Levenberg-Marquardt backpropagation
net.trainFcn = 'trainlm';
net.trainParam.goal = 0.001; %warunek stopu � poziom b��du
net.trainParam.epochs = 200; %maksymalna liczba epok
net.trainParam.showWindow = false; %nie pokazywa� okna z wykresami
%w trakcie uczenia
net=train(net,dataTrainX',dataTrainY'); %uczenie sieci

end