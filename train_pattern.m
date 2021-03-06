% -*- Mode: octave -*-


function X = train_pattern

x1 = [1 1 1 1 1 1 ; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 1; 1 1 1 1 1 1];
x2 = [-1 1 1 1 1 1 ; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 1; 1 1 1 1 1 1];
x3 = [1 1 1 1 1 -1 ; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 1; 1 1 1 1 1 1];
x4 = [1 1 1 1 1 1 ; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 1; -1 1 1 1 1 1];
x5 = [-1 1 1 1 1 -1 ; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 1; 1 -1 -1 -1 -1 1; 1 1 1 1 1 1];
x6 = [1 1 1 1 1 1 ; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 1 1 1 1 1];
x7 = [-1 1 1 1 1 1 ; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 1 1 1 1 1];
x8 = [1 1 1 1 1 1 ; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; -1 1 1 1 1 1];
x9 = [1 1 1 1 1 -1 ; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1;1 1 1 1 1 -1];
x10 = [-1 1 1 1 1 1 ; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 1 1 1 1 -1];
x11 = [1 1 1 1 1 1 ; 1 -1 -1 -1 -1 -1; 1 1 1 1 1 1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1];
x12 = [-1 1 1 1 1 1 ; 1 -1 -1 -1 -1 -1; 1 1 1 1 1 1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1];
x13 = [1 1 1 1 1 -1 ; 1 -1 -1 -1 -1 -1; 1 1 1 1 1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1];
x14 = [1 1 1 1 1 1 ; 1 -1 -1 -1 -1 -1; -1 1 1 1 1 1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1];
x15 = [1 1 1 1 1 1 ; 1 -1 -1 -1 -1 -1; 1 1 1 1 1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1; 1 -1 -1 -1 -1 -1];

xr1 = reshape(x1',1,36);
xr2 = reshape(x2',1,36);
xr3 = reshape(x3',1,36);
xr4 = reshape(x4',1,36);
xr5 = reshape(x5',1,36);
xr6 = reshape(x6',1,36);
xr7 = reshape(x7',1,36);
xr8 = reshape(x8',1,36);
xr9 = reshape(x9',1,36);
xr10 = reshape(x10',1,36);
xr11 = reshape(x11',1,36);
xr12 = reshape(x12',1,36);
xr13 = reshape(x13',1,36);
xr14 = reshape(x14',1,36);
xr15 = reshape(x15',1,36);

X = [xr1' xr2' xr3' xr4' xr5' xr6' xr7' xr8' xr9' xr10' xr11' xr12' xr13' xr14' xr15'];

end
