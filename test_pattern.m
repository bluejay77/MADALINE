% -*- Mode: octave -*-



function [X_test] = test_pattern
    
X1 = [1 1 1 -1 1 1 ; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 1; 1 1 1 -1 1 1];
X2 = [1 1 1 1 1 -1 ; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 1; -1 1 1 1 1 1];
X3 = [-1 1 1 1 1 1 ; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 1; -1 1 1 1 1 1];
X4 = [1 1 1 1 1 1 ; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 1; -1 1 -1 -1 -1 1; -1 -1 1 1 1 -1];
X5 = [-1 1 1 1 -1 -1 ; 1 -1 -1 -1 1 1; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 1; -1 1 1 1 1 1];
X6 = [-1 -1 1 1 1 1 ; -1 1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; -1 1 -1 -1 -1 -1; -1 -1 1 1 1 1];
X7 = [1 1 1 1 -1 -1 ; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 1 1 1 1 1];
X8 = [1 1 1 1 1 1 ; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 1 1 1 -1 -1];
X9 = [1 1 1 1 1 -1 ; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; -1 1 -1 -1 -1 -1;-1 -1 1 1 1 -1];
X10 = [-1 1 1 1 1 1 ; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; -1 -1 1 1 1 -1];
X11 = [-1 1 1 1 1 1 ; 1 -1 -1 -1 -1 -1; 1 1 1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1];
X12 = [1 1 1 1 1 1 ; -1 -1 -1 -1 -1 -1; 1 1 1 1 1 1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1];
X13 = [1 1 1 -1 -1 -1 ; 1 -1 -1 -1 -1 -1; 1 1 1 1 1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1];
X14 = [1 1 -1 1 1 -1 ; 1 -1 -1 -1 -1 -1; -1 -1 1 1 1 1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; -1 -1 -1 -1 -1 -1];
X15 = [-1 -1 1 1 1 1 ; -1 1 -1 -1 -1 -1; -1 1 1 1 1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1];


xr1 = reshape(X1',1,36);
xr2 = reshape(X2',1,36);
xr3 = reshape(X3',1,36);
xr4 = reshape(X4',1,36);

xr5 = reshape(X5',1,36);
xr6 = reshape(X6',1,36);
xr7 = reshape(X7',1,36);
xr8 = reshape(X8',1,36);
xr9 = reshape(X9',1,36);
xr10 = reshape(X10',1,36);
xr11 = reshape(X11',1,36);
xr12 = reshape(X12',1,36);
xr13 = reshape(X13',1,36);
xr14 = reshape(X14',1,36);
xr15 = reshape(X15',1,36);


X_test = [xr1' xr2' xr3' xr4' xr5' xr6' xr7' xr8' xr9' xr10' xr11' xr12' xr13' xr14' xr15'];

end

