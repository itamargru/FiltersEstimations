function [states,probs] = generate_MC(N, TPM, init_probs)

states = zeros(1,N);
probs = zeros(1,N);
states(1) = multinomial_one(init_probs);
probs(1) = init_probs(states(1));
for i=2:N   
    states(i) = multinomial_one(TPM(states(i-1),:));
    probs(i) = TPM(states(i-1),states(i));
end