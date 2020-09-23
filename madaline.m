% Mode: -*- Text -*-
%
% The MADALINE program from Prof Daniel Graupes book
% 2nd edtion.
%
% This one for the GNU Octave, in Graupes book for the MATLAB
%
% Dr Antti Juhani Ylikoski 2020-09-16
%
% Use:
% sudo apt-get install octave
% sudo apt-get install octave-doc
% octave train_pattern.m test_pattern.m madaline.m 
%
% with the Linux
%
% NOTA BENE: With the current GNU Octave the author did not 
% get this one fully working.
% Needs debugging, and, perhaps some porting and adaptation work.
%


% Training Patterns
X = train_pattern;
nu = 0.04;

% Displaying the 15 training patterns
figure(1)
for i = 1:15,
    subplot(5,3,i)
    display_image(X(:,i),6,6,1);
end

% Testing Patterns
Y = test_pattern;
nu = 0.04;

% Displaying the 15 testing patterns
figure(2)
for i = 1:15,
    subplot(5,3,i)
    display_image(Y(:,i),6,6,1);
end

% Initializations
index = zeros(2,6);
counter1 = 0;
counter2 = 0;

% Assign random weights initially at the start of training
w_hidden = (rand(6,36)-0.5)*2
w_output = (rand(2,6)-0.5)*2

%load w_hidden.mat
%load w_output.mat

% Function to calculate the parameters (z,y at the hidden and output
% layers given the weights at the two layers)

[z_hidden, w_hidden, y_hidden, z_output, w_output, y_output, counter] = calculation(w_hidden, w_output, X);

disp('Before Any Changes')

w_output
z_output
y_output

save z_output z_output;
save z_hidden z_hidden;
save y_hidden y_hidden;
save y_output y_output;

counter

i = 1;

%min_z_output = min(abs(z_output));
disp('At counter minimum')

if (counter~= 0),
   [w_output_min,z_index] = min_case(z_output,w_output,counter,y_hidden,nu);
   [z_hidden_min, w_hidden_min, y_hidden_min, z_output_min, w_output_min, y_output_min, counter1] = calculation(w_hidden, w_output_min, X);
   counter1
end


w_output_min;
z_output_min;
y_output_min;

if (counter > counter1),
   %load w_output.mat;
   %load z_output.mat;
   %load y_output.mat;
   counter = counter1;
   w_output = w_output_min;
   z_output = z_output_min;
   y_output = y_output_min;
   index(2,z_index) = 1;
end



[w_output_max,z_ind] = max_case(z_output,w_output,counter,y_hidden,nu);

[z_hidden_max, w_hidden_max, y_hidden_max, z_output_max, w_output_max, y_output_max, counter2] = calculation(w_hidden, w_output_max, X);

disp('At Counter minimum')

counter2

w_output_max;
z_output_max;
y_output_max;

if (counter2<counter),
   counter = counter2;
   w_output = w_output_max;
   z_output = z_output_max;
   y_output = y_output_max;
   index(2,z_ind) = 1;
end


% Adjusting the weights of the hidden layer
hidden_ind = zeros(1,6);
z_hid_asc = sort(abs(z_hidden));

for i = 1:6,
    for k = 1:6,
    if z_hid_asc(i) == abs(z_hidden(k)),

    hidden_ind(i) = k;
    end
  end


r1 = hidden_ind(1);
r2 = hidden_ind(2);
r3 = hidden_ind(3);
r4 = hidden_ind(4);
r5 = hidden_ind(5);
r6 = hidden_ind(6);

disp('At the beginning of the hidden layer Weight Changes - Neuron 1')

%load w_hidden.mat;
if ((counter~=0)&(counter>6)),
   [w_hidden_min] = min_hidden_case(z_hidden,w_hidden,counter,X,nu,hidden_ind(1));
   [z_hidden_min, w_hidden_min, y_hidden_min, z_output_min, w_output, y_output_min, counter3] = calculation(w_hidden_min, w_output, X);
   counter3
end

w_hidden;

if (counter3<counter),
   counter=counter3;
   w_hidden = w_hidden_min;
   y_hidden = y_hidden_min;
   z_hidden = z_hidden_min;
   z_output = z_output_min;
   y_output = y_output_min;
   index(1,r1) = 1;
end

disp('Hidden Layer - Neuron 2')

%load w_hidden.mat;
%counter=counter2;

if ((counter~=0)&(counter>6)),
   [w_hidden_min] = min_hidden_case(z_hidden,w_hidden,counter,X,nu,hidden_ind(2));
   [z_hidden_min, w_hidden_min, y_hidden_min, z_output_min, w_output, y_output_min, counter3] = calculation(w_hidden_min, w_output, X);
   counter3
end

w_hidden;
w_hidden_min;

if (counter3<counter),
   counter = counter3;
   w_hidden = w_hidden_min;
   y_hidden = y_hidden_min;
   z_hidden = z_hidden_min;
   z_output = z_output_min;
   y_output = y_output_min;
   index(1,r2)=1;
end

disp('Hidden Layer - Neuron 3')
%load w_hidden.mat;
%counter=counter2;

if ((counter~=0)&(counter>6)),
   [w_hidden_min] = min_hidden_case(z_hidden,w_hidden,counter,X,nu,hidden_ind(3));
   [z_hidden_min, w_hidden_min, y_hidden_min, z_output_min, w_output, y_output_min, counter3] = calculation(w_hidden_min, w_output, X);
   counter3
end

w_hidden;
w_hidden_min;

if (counter3<counter),
   counter = counter3;
   w_hidden = w_hidden_min;
   y_hidden = y_hidden_min;
   z_hidden = z_hidden_min;
   z_output = z_output_min;
   y_output = y_output_min;
   index(1,r3) = 1;
end

disp('Hidden Layer - Neuron 4')
%load w_hidden.mat;
%counter=counter2;


if ((counter~=0)&(counter>6)),
   [w_hidden_min] = min_hidden_case(z_hidden,w_hidden,counter,X,nu,hidden_ind(4));
   [z_hidden_min, w_hidden_min, y_hidden_min, z_output_min, w_output, y_output_min, counter3] = calculation(w_hidden_min, w_output, X);
   counter3
end

w_hidden;
w_hidden_min;

if (counter3<counter),
   counter = counter3;
   w_hidden = w_hidden_min;
   y_hidden = y_hidden_min;
   z_hidden = z_hidden_min;
   z_output = z_output_min;
   y_output = y_output_min;
   index(1,r4)=1;
end

disp('Hidden Layer - Neuron 5')


%load w_hidden.mat;
%counter=counter2;
if (counter~=0),
   [w_hidden_min] = min_hidden_case(z_hidden,w_hidden,counter,X,nu,hidden_ind(5));
   [z_hidden_min, w_hidden_min, y_hidden_min, z_output_min, w_output, y_output_min, counter3] = calculation(w_hidden_min, w_output, X);
counter3
end
end

w_hidden;
w_hidden_min;

if (counter3<counter),
   counter = counter3;
   w_hidden = w_hidden_min;
   y_hidden = y_hidden_min;
   z_hidden = z_hidden_min;
   z_output = z_output_min;
   y_output = y_output_min;
   index(1,r5)=1;
end

disp('Combined Output Layer Neurons weight change');


%load w_hidden.mat;
%counter = counter2;
if ((counter~=0)&(index(2,[1:2])~=1)&(counter>6)),
   [w_output_two] = min_output_double(z_hidden,y_hidden,counter,X,nu,w_output);
   [z_hidden_min, w_hidden_min, y_hidden_min, z_output_min, w_output, y_output_min, counter3] = calculation(w_hidden,w_output_two, X);
   counter3
end


w_output;
%w_output_two;

if (counter3<counter),
   counter = counter3;
   %w_hidden = w_hidden_min;
   y_hidden = y_hidden_min;
   z_hidden = z_hidden_min;
   z_output = z_output_min;
   y_output = y_output_min;
   w_output = w_output_two;
end

disp('Begin 2 neuron changes - First Pair')

%load w_hidden.mat;
%counter = counter2;

if ((counter~=0)&(index(1,r1)~=1)&(index(1,r2)~=1)&(counter>6)),
   [w_hidden_two] = min_hidden_double(z_hidden,w_hidden,counter,X,nu,hidden_ind(1),hidden_ind(2));
   [z_hidden_min, w_hidden_min, y_hidden_min, z_output_min, w_output, y_output_min, counter3] = calculation(w_hidden_two, w_output, X);
   counter3
end

w_hidden;
w_hidden_min;


if (counter3<counter),
   counter = counter3;
   w_hidden = w_hidden_min;
   y_hidden = y_hidden_min;
   z_hidden = z_hidden_min;
   z_output = z_output_min;
   y_output = y_output_min;
end

disp('Begin 2 neuron changes - Second Pair')

%load w_hidden.mat;
%counter = counter2;


if ((counter~=0)&(index(1,r2)~=1)&(index(1,r3)~=1)&(counter>6)),
   [w_hidden_two] = min_hidden_double(z_hidden,w_hidden,counter,X,nu,hidden_ind(2),hidden_ind(3));
   [z_hidden_min, w_hidden_min, y_hidden_min, z_output_min, w_output, y_output_min, counter3] = calculation(w_hidden_two, w_output, X);
   counter3
end

w_hidden;
w_hidden_min;

if (counter3<counter),
   counter = counter3;
   w_hidden = w_hidden_min;
   y_hidden = y_hidden_min;
   z_hidden = z_hidden_min;
   z_output = z_output_min;
   y_output = y_output_min;
end

disp('Begin 2 neuron changes - Third Pair')

%load w_hidden.mat;
%counter = counter2;


if ((counter~=0)&(index(1,r3)~=1)&(index(1,r4)~=1)&(counter>6)),
   [w_hidden_two] = min_hidden_double(z_hidden,w_hidden,counter,X,nu,hidden_ind(3),hidden_ind(4));
   [z_hidden_min, w_hidden_min, y_hidden_min, z_output_min, w_output, y_output_min, counter3] = calculation(w_hidden_two, w_output, X);
   counter3
end

w_hidden;
w_hidden_min;


if (counter3<counter),
   counter = counter3;
   w_hidden = w_hidden_min;
   y_hidden = y_hidden_min;
   z_hidden = z_hidden_min;
   z_output = z_output_min;
   y_output = y_output_min;
end

disp('Begin 2 neuron changes - Fourth Pair')

%load w_hidden.mat;
%counter = counter2;


if ((counter~=0)&(index(1,r4)~=1)&(index(1,r5)~=1)&(counter>6)),
   [w_hidden_two] = min_hidden_double(z_hidden,w_hidden,counter,X,nu,hidden_ind(4),hidden_ind(5));
   [z_hidden_min, w_hidden_min, y_hidden_min, z_output_min, w_output, y_output_min, counter3] = calculation(w_hidden_two, w_output, X);
   counter3
end

w_hidden;
w_hidden_min;

disp('Final Values For Output')

w_output
z_output
y_output

disp('Final Values for Hidden')

w_hidden
z_hidden
y_hidden

disp('Final Error Number')
counter

disp('Efficiency')

eff = 100 - counter/40*100


% *****************Function to calculate the parameters (z,y at the
% hidden and output layers given the weights at the two layers)******************

function [z_hidden, w_hidden, y_hidden, z_output, w_output, y_output, counter] = calculation(w_hidden, w_output, X)

% Outputs:
% z_hidden - hidden layer z value
% w_hidden - hidden layer weight
% y_hidden - hidden layer output
% Respecitvely for the output layers
% Inputs:
% Weights at the hidden and output layers and the training pattern set

counter = 0;
r = 1;
while(r<=15),
	r;
for i = 1:6,
    z_hidden(i) = w_hidden(i,:)*X(:,r);
    if (z_hidden(i)>=0),
       y_hidden(i) = 1;
    else
	y_hidden(i) = -1;
end %%End of If loop
end %% End of for loop

z_hidden;
y_hidden = y_hidden;
for i = 1:2
    z_output(i) = w_output(i,:)*y_hiddent;
    if (z_output(i)>=0),
       y_output(i) = 1;
       else
	y_output(i) = -1;
end %% End of If loop
end%% End of for loop
y_output;

% Desired Output
if (r<=5),
d1 = [1 1]; % For 0
else if (r>10),
d1 = [-1 -1] %For F
else
d1 = [-1 1]; % For C
end
end
for i = 1:2,
    error_val(i) = d1(i)-y_output(i);
if (error_val(i)~=0),
   counter = counter+1;
end
end
r = r+1;
end
end



% ******Function to find weight changes for paired hidden layer**********

function [w_hidden_two] = min_hidden_double(z_hidden,w_hidden,counter,X,nu,k,l)
w_hidden_two = w_hidden;
for j = 1:36,
w_hidden_two(k,j) = w_hidden_two(k,j) + 2*nu*X(j,15)*counter;
w_hidden_two(l,j) = w_hidden_two(l,j) + 2*nu*X(j,15)*counter;
end
end


% *********Function to find weight changes at hidden layer**************
function [w_hidden_min] = min_hidden_case(z_hidden,w_hidden,counter,X,nu,k)
w_hidden_min = w_hidden;
for j = 1:36,
w_hidden_min(k,j) = w_hidden_min(k,j) + 2*nu*X(j,15)*counter;
end
%w_hidden_min
end


% ****Function to change weights for the max of 2z values at Output****
function [w_output_max,z_ind] = max_case(z_output,w_output,counter,y_hidden,nu)
%load w_output;
%load z_output;
w_output_max = w_output;
z_ind = find(abs(z_output) == max(abs(z_output)))
for j = 1:5,
w_output_max(z_ind,j) = w_output(z_ind,j)+2*nu*y_hidden(j)*counter;
%
end
%
z_output(z_index) = w_output(z_index,:)*y_hiddent;
end

% ****************Function to compute weight change at the output for
neuron whose Z value is close to the threshold**********************
function [w_output_min,z_index] = min_case(z_output,w_output,counter,y_hidden,nu)
z_index = find(abs(z_output) == min(abs(z_output)))
w_output_min = w_output
for j = 1:5,
w_output_min(z_index,j) = w_output(z_index,j) + 2*nu*y_hidden(j)*counter;
end
w_output_min
end



% *******Function to find weight changes with paired output neurons******
function [w_output_two] = min_output_double(z_hidden,y_hidden,counter,X,nu,w_output)
w_output_two = w_output;
for j = 1:6,
w_output_two([1:2],j) = w_output([1:2],j)+2*nu*y_hidden(j)*counter;
end
y_hidden;
counter;
2*nu*y_hidden*counter;
end


