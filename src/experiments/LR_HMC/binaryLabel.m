function bY = binaryLabel(Y)

Y = uint8(Y);
[N,D] = size(Y);
if D ~= 1
    error('Dim must be 1');
end

cls = unique(Y);
nC = length(cls);

bY = zeros(N,nC);

for i=1:N
    bY(i,Y(i)+1) = 1;
end
