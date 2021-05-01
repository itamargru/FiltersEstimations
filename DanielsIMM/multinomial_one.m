function [x] = multinomial_one(p)
pp = cumsum(p);
temp = rand;
x = find(temp < pp, 1, 'first');