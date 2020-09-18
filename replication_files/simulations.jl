using Distributed
using JSON
addprocs(8)
@everywhere using IterTools;
@everywhere using Dates;
@everywhere using Distances;
@everywhere using Random;
@everywhere using Distributions;
@everywhere using LinearAlgebra;


"""
iota(n::Int64, N::Int64)
ι : N → R 2 associates with each index n a point in
        the unit circle, evenly spaced, with ι(n) = (cos(n/N) π, sin(n/N) * π)
# Arguments
- `n::Int64`: point on circle  
- `N::Int64`: # of goods so length of circle 
# Returns
- the cos and sin components as Array{Float64,1}
"""

@everywhere function iota(n::Int64,N::Int64)::Array{Float64,1}
    return [cos(n/N) * pi, sin(n/N) * pi]
end


""" the minimum # of hops from node i to node j
"""

@everywhere function hop_distance(
        i::Int64,
        j::Int64,
        N::Int64
    )::Int64
    return min(abs(i-j), abs(j-i), abs(j-i-N), abs(i-j-N))
end

"""
	cov_mat_fun(sigma::Float64, rho::Float64, N::Int64)
Create covariance matrix
# Arguments
- `sigma::Float64`: 
- `rho::Float64`: Covariance coefficient
- `N::Int`: Number of goods
# Returns
- Covariance matrix 
"""

@everywhere function cov_mat_fun(sigma::Float64, rho::Float64, N::Int64)::Array{Float64,2}
    cov_mat = zeros(Float64, N, N)
    for i in 1:N 
        for j in 1:N
            dist = hop_distance( i, j , N)
            #dist = Distances.euclidean( iota(i,N), iota(j,N) )
            cov_mat[i,j] = sigma * rho^dist
        end
    end
    return cov_mat
end

""" 
CARA
# Arguments:
- alpha (int) :
- mu (numpy.ndarray): shape (N,)
- sigma (numpy.ndarray) : shape (N,N)
# Returns:
    (numpy.ndarray) of shape (N,)
# Notes:
    alpha: the coefficient of absolute risk aversion
    μ and σ2 are the mean and the variance of the distribution F
    https://ocw.mit.edu/courses/economics/14-123-microeconomic-theory-iii-spring-2015/lecture-notes-and-slides/MIT14_123S15_Chap3.pdf 
    pg 21
"""

@everywhere function certainty_equivalent(
        alpha::Float64, 
        mu::Array{Float64,1},
        sig::Array{Float64,2}
    )
    new_mu = mu - (.5 * alpha * diag(sig)) 
    return new_mu
end

### Welfare Functions - Statistic Calculation Functions

@everywhere function w_fun(
        CiT::Array{Int64, 1},
        Ui::Array{Float64,1},
        T::Int64
    )
    w_score = 0.0
    for i in 1:length(CiT)
        w_score = w_score + Ui[CiT[i]]
    end
    return w_score*(T^(-1))
end



"""
    init_sigma
init for bayesian update

"""

@everywhere function init_sigma(Cit::Array{Int64,1},
            Nit::Array{Int64,1},
            Sigma_Ui::Array{Float64,2})

    Sigma11::Array{Float64,2} = ones(Float64, length(Cit), length(Cit))
    Sigma12::Array{Float64,2} = ones(Float64, length(Cit), length(Nit))
    Sigma21::Array{Float64,2} = ones(Float64, length(Nit), length(Cit))
    Sigma22::Array{Float64,2} = ones(Float64, length(Nit), length(Nit))

    for i in 1:length(Cit)
        for j in 1:length(Cit)
            Sigma11[i,j] = Sigma_Ui[Cit[i],Cit[j]] 
        end
        for j in 1:length(Nit)
            Sigma21[j,i] = Sigma_Ui[Cit[i], Nit[j]] 
        end
    end

    for i in 1:length(Nit)
        for j in 1:length(Cit)
            Sigma12[j,i] = Sigma_Ui[Nit[i],Cit[j]] 
        end
        for j in 1:length(Nit)
            Sigma22[i,j] = Sigma_Ui[Nit[i], Nit[j]]
        end
    end
    return Sigma11, Sigma12, Sigma21, Sigma22
end

@everywhere function get_mubar_sigmamu(
        Sigma_Ui::Array{Float64,2}, 
        Ui::Array{Float64,1}, 
        x1::Array{Int64,1}, 
        Sigma11::Array{Float64,2}, 
        Sigma12::Array{Float64,2}, 
        Sigma21::Array{Float64,2}, 
        Sigma22::Array{Float64,2}, 
        mu1::Array{Float64,1}, 
        mu2::Array{Float64,1}
    )
   
    a = [Ui[n] for n in x1]

    inv_mat = inv(Sigma11)

    inner = Sigma21 * inv_mat

    mubar = mu2 + inner * (a-mu1)
    
    sigmabar = Sigma22 - (inner * Sigma12)

    return mubar, sigmabar
end

"""
    update_Ui()

Bayesian Update

# Arguments

"""

@everywhere function update_Ui(
            Cit::Array{Int64,1}, 
            Ui::Array{Float64,1}, 
            mu_Ui::Array{Float64,1}, 
            Sigma_Ui::Array{Float64,2}, 
            Nset::Array{Int64,1}
    )
    
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
    # μ_bar = μ_1 + Ε12 Ε22^-1 ( a - μ_2 )  

    x1 = Cit
    x2 = [n for n in Nset if n ∉ Cit]

    mu1 = [mu_Ui[n] for n in x1]
    mu2 = [mu_Ui[n] for n in x2]
   
    Sigma11, Sigma12, Sigma21, Sigma22 = init_sigma(x1,x2, Sigma_Ui)

    mubar, sigmabar = get_mubar_sigmamu(Sigma_Ui, Ui, x1, Sigma11, Sigma12, Sigma21, Sigma22, mu1, mu2)
    return mubar, sigmabar
end

@everywhere function choice_helper(
        ce::Array{Float64,1}, 
        choice_set::Array{Int64, 1}
    )
    cit = choice_set[argmax(ce)]
    return cit
end 

@everywhere function thompson_sampling(
    mu_V::Array{Float64,1},
    Sigma_V::Array{Float64,2}
)

    
    draws = [ 
            rand(
                Normal(
                    mu_V[ii],
                    Sigma_V[ii,ii]
                )
            )  for ii in 1:length(mu_V) ]
    c_it = argmax(draws)

    return c_it
end

@everywhere function choice_part(
    V_i::Array{Float64,1}, 
    mu_V_i::Array{Float64,1},
    Sigma_V_i::Array{Float64,2},
    V::Array{Float64,1}, 
    T::Int64,
    N::Int64,
    Nset::Array{Int64,1},
    alpha::Float64,
    beta::Float64
)
    C_iT::Array{Int64,1} = []
    R_iT::Array{Int64,1} = []
    cur_V = copy(V)

    for t=1:T
        choice_set = [n for n in Nset if n  ∉ C_iT]
        mu_Vit = copy(mu_V_i)
        Sigma_Vit = copy(Sigma_V_i)
        if length(C_iT) > 0
            # update beliefs
            mu_Vit, Sigma_Vit = update_Ui(C_iT, copy(V_i), copy(mu_Vit), copy(Sigma_Vit), Nset)
            cur_V = [V[i] for i in Nset if i ∉ C_iT]
        end
        mu_Uit = mu_Vit + beta * cur_V
        # make choice
        ce_Uit = certainty_equivalent(alpha, mu_Uit, Sigma_Vit) # γ: uncertainty aversion

        c_it = choice_helper(ce_Uit, choice_set)
        r_it = choice_set[argmax([V[i] for i in choice_set])]
        append!(R_iT,r_it)
        append!(C_iT,c_it)
    end

    return C_iT, R_iT
end

@everywhere function choice_omni(U_i::Array{Float64,1},T::Int64,N::Int64, Nset::Array{Int64, 1})
    C_iT::Array{Int64,1} = []
    for t=1:T
        choice_set = [n for n in Nset if n ∉ C_iT]
        sub_U_i = [U_i[n] for n in choice_set]
        c_it = choice_helper(sub_U_i, choice_set)
        append!(C_iT, c_it)
    end
    return C_iT
end

@everywhere function choice_ind(U_i::Array{Float64,1}, 
			mu_U_i::Array{Float64,1}, 
			Sigma_U_i::Array{Float64,2}, 
			T::Int64, 
			N::Int64, 
			Nset::Array{Int64,1}, 
			alpha::Float64
)

    C_iT::Array{Int64,1} = []
    for t=1:T
        if length(C_iT) > 0
            mu_Uit, Sigma_Uit = update_Ui(C_iT, copy(U_i), copy(mu_U_i), copy(Sigma_U_i), Nset)
        else
            mu_Uit = copy(mu_U_i)
            Sigma_Uit = copy(Sigma_U_i)
        end
        
        # make choice
        ce_Uit = certainty_equivalent(alpha, mu_Uit, Sigma_Uit)
        choice_set = [n for n in Nset if n ∉ C_iT]
        c_it = choice_helper(ce_Uit, choice_set)
        append!(C_iT, c_it)
    end
		
    return C_iT
end


"""
    simulate(
        N::Int64
        T::Int64, 
        sigma::Float64,
        sigma_i::Float64, 
        sigma_ibar::Float64,
        beta::Float64, 
        nr_ind::Int64,
        Sigma_V_i::Array{Float64,2},
        Sigma_V::Array{Float64,2},
        Sigma_V_ibar::Array{Float64,2},
        alpha::Float64,
        seed::Float64
    )

# Arguments
# Returns
"""

@everywhere function simulate(N::Int64,
    T::Int64, 
    sigma::Float64,
    sigma_i::Float64, 
    sigma_ibar::Float64,
    beta::Float64, 
    nr_ind::Int64,
    Sigma_V_i::Array{Float64,2},
    Sigma_V::Array{Float64,2},
    Sigma_V_ibar::Array{Float64,2},
    alpha::Float64,
    seed::Int64
    )

    
    println("iteration: $seed ")
    Random.seed!(seed);

    Nset = [ n for n=1:N]   # set of N items I = {1, ..., N}

    C_pop = Dict( "no_rec"  => zeros(Int64, nr_ind,T), "omni"  => zeros(Int64, nr_ind,T), "partial" => zeros(Int64, nr_ind,T))
    W_pop = Dict( "no_rec"  => zeros(nr_ind,T), "omni"  => zeros(nr_ind,T), "partial" => zeros(nr_ind,T))
    R_pop = Dict( "no_rec"  => zeros(nr_ind,T), "omni"  => zeros(nr_ind,T), "partial" => zeros(nr_ind,T))

    # V = (v_n) n in I aka: common value component v_n in vector form

    # MvNormal(mu, sig) 
    #Construct a multivariate normal distribution with mean mu and covariance represented by sig.
    # https://juliastats.github.io/Distributions.jl/stable/multivariate/#Distributions.MvNormal
    mu_V = zeros(Float64, N)
    V = rand(MvNormal(mu_V, Sigma_V))

    for it_ind=1:nr_ind
        # V_i = (v_in) n in I aka: consumer i’s idiosyncratic taste for good n in vector form

        mu_V_i = mu_V_ibar = rand(MvNormal(zeros(Float64, N), Sigma_V_ibar))
        V_i = rand(MvNormal(mu_V_i, Sigma_V_i))

        # Utility in vector form
        U_i = V_i + (beta * V)
        mu_U_i = mu_V_i + beta * mu_V
        mu_U_i = convert(Array{Float64,1}, mu_U_i)
        mu_V_i = convert(Array{Float64,1}, mu_V_i)

        ## NO RECOMMENDATION CASE
        Sigma_U_i = Sigma_V_i + beta^2 * (Sigma_V)
        C_iT_no_rec = choice_ind(copy(U_i), copy(mu_U_i), copy(Sigma_U_i),T,N, Nset, alpha)
        C_pop["no_rec"][it_ind,:] = C_iT_no_rec
        W_pop["no_rec"][it_ind,:] = U_i[C_iT_no_rec]

        
        ## OMNISCIENT CASE
        C_iT = choice_omni(copy(U_i),T,N, Nset)
        C_pop["omni"][it_ind,:] = C_iT
        W_pop["omni"][it_ind,:] = U_i[C_iT]

 
        ## PARTIAL REC Case
        C_iT_partial, R_iT = choice_part(copy(V_i), copy(mu_V_i), copy(Sigma_V_i), copy(V), T, N, Nset, alpha, beta)
        C_pop["partial"][it_ind,:] = C_iT_partial
        W_pop["partial"][it_ind,:] = copy(U_i[C_iT_partial])
        R_pop["partial"][it_ind,:] = R_iT
    end

    return Dict( "Consumption" => C_pop, "Welfare" => W_pop, "Rec" => R_pop )
end
#
nr_pop = 100
#
nr_ind = 100
#
sigma_ibar = .1
#
rho_ibar = 0.0

N_vals = [200]

T_vals = [20]

# Covariance structure
rho_vals = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
# utility idiosyncratic degree 
beta_vals = [0.0, 0.4, 0.8, 1., 2., 5.]
# absolute risk aversion
alpha_vals = [0.0, 0.3, 0.6, 1., 5.]


sigma_vals = [0.25, 0.5, 1.0, 2.0, 4.0]

params = Iterators.product(N_vals, T_vals, rho_vals, beta_vals, sigma_vals, alpha_vals)

println(length(collect(params)))

WORKING_DIR = "/Users/guyaridor/Desktop/ExAnteFilterBubble/data/sim_results/"
#WORKING_DIR = "/media/IntentMedia/rec_data/sim_results/"

global NUM_SIMS_TO_WRITE = 25
global file_idx = 1
global sim_results = Dict()
global total_num = 0

for (N, T, rho, beta, sigma, alpha) in params
    println("STARTING")
    println("N: $N, T: $T, ρ: $rho β: $beta σ: $sigma α: $alpha")
    println(Dates.now())
    flush(stdout) # so that nohup shows progress
    sigma_i = sigma

    Sigma_V_i = cov_mat_fun(sigma, rho, N)
    Sigma_V = cov_mat_fun(sigma,rho,N)

    Sigma_V_ibar = cov_mat_fun(sigma_ibar,rho_ibar,N)
    global sim_results[(N, T, rho, beta, sigma, alpha, nr_pop, nr_ind)] = @sync @distributed vcat for i= 1:nr_pop
        simulate(N, T,sigma, sigma_i, sigma_ibar, beta, nr_ind, Sigma_V_i,  Sigma_V,  Sigma_V_ibar,  alpha, i)
    end
    total_num = total_num
    if total_num > NUM_SIMS_TO_WRITE
        file_name = string("new_sim_",file_idx,".json")
        open(string(WORKING_DIR, file_name),"w") do f
            JSON.print(f, sim_results)
        end
        global file_idx = file_idx + 1
        global total_num = 0
        global sim_results = Dict()
    else
        #print(NUM_SIMS_TO_WRITE)
        #print(total_num)
        global total_num  = total_num + 1
    end
end

file_name = string("new_sim_",file_idx,".json")
open(string(WORKING_DIR, file_name),"w") do f
    JSON.print(f, sim_results)
end
print(total_num)
print(NUM_SIMS_TO_WRITE)


