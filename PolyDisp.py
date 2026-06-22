# PolyDis: A deterministic open-source code for atomic
# displacement calculations in arbitrary polyatomic materials
#
# PolyDis is writen in and runs with Python3
#
# MIT License
# Copyright (c) 2026 Shengli Chen
# Sun Yat-sen University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
###############################################################################

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.integrate import quad
from scipy.optimize import root
import matplotlib.pyplot as plt

# =============================================
# =============================================
# 1. Constants and Basic Functions
# =============================================
# =============================================

LAMBDA_WSS = 1.309 
CONST_34 = 34.8552
CONST_U = 2.646e-4

# =============================================
# 1.1 Function for getting atomic data
# =============================================

_ATOMIC_DATA = {}
try:
    with open('list_elements.txt', 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                symbol = parts[0]
                A = float(parts[2])      
                Z = int(parts[3])        
                _ATOMIC_DATA[symbol] = (Z, A)
except FileNotFoundError:
    # Fallback dictionary if list_elements.txt is not found
    _ATOMIC_DATA = {
        'C': (6, 12.011),
        'O': (8, 15.999),
        'Al': (13, 26.982),
        'Si': (14, 28.086),
        'Cr': (24, 51.996),  
        'Fe': (26, 55.845),
        'Ni': (28, 58.693),
        'Y': (39, 88.906),
        'U': (92, 238.029),
        'virtual': (26.222, 56.222)
    }

def get_atomic_data(name):
    return _ATOMIC_DATA.get(name)


# =============================================
# 1.2 Function for calculating the constants
# =============================================
# elements can be extracted from the Function in 1.1
# or user-defined ones with 'arbitrary' atomic
# numbers and masses for specific investigation,
# such as to perform the averaged-target NRT comparison
#
def calc_parameters(elements, stoichiometry):
    n = len(elements)
    Z = np.array([e[0] for e in elements])
    A = np.array([e[1] for e in elements])
    N_frac = np.array(stoichiometry) / sum(stoichiometry)

    ## G_ij, U_ij, and \Lambda_ij
    #  See Sec. 2.1 for definitions
    G = np.zeros((n, n))
    U = np.zeros((n, n))
    M = np.zeros((n, n))
    
    for i in range(n):
        denom_sum = 0.0
        for k in range(n):
            term = (N_frac[k] * Z[k]) / ((Z[i]**(2/3) + Z[k]**(2/3))**1.5)
            denom_sum += term
            
        for j in range(n):
            num = 34.8552 * (A[i] / (Z[i]**(1/6))) * N_frac[j] * (Z[j] / (A[j]**0.5))
            num /= (Z[i]**(2/3) + Z[j]**(2/3))**0.5
            G[i, j] = num / denom_sum

    for i in range(n):
        for j in range(n):
            term = 2.646e-4 * (A[j] / A[i]) * (1.0 / (Z[i]**2 * Z[j]**2 * (Z[i]**(2/3) + Z[j]**(2/3))))
            U[i, j] = term
            M[i, j] = (4 * A[i] * A[j]) / ((A[i] + A[j])**2)
            
    return G, U, M


# =============================================
# 1.3 Winterbon's analytic form
#       of the Lindhard univariable function
#       for differential cross section
# =============================================

def f_winterbon(xi):
    """
    The screening function f(xi) using Winterbon approximation.
    Note: f(t^1/2). Let xi = t^1/2 = (U*E*T)^1/2 in Coulter's paper.
    Then t = xi^2.
    Formula: lambda * t^(1/6) * [1 + (2*lambda*t^(2/3))^(2/3)]^(-3/2)
           = lambda * xi^(1/3) * [1 + (2*lambda*xi^(4/3))^(2/3)]^(-3/2)
    """
    lam = LAMBDA_WSS
    term1 = lam * xi**(1.0/3.0)
    term2 = (1.0 + (2 * lam * xi**(4.0/3.0))**(2.0/3.0))**(-1.5)
    return term1 * term2


# =============================================
# 1.4 Define the integrand kernel
#       for each {il} and {jl}
# =============================================

#'''
def integrand_kernel_for_quad(T, E_curr, nu_func_j, nu_func_i, nu_i_E_curr, U_val):
    """
    Computes the term inside the integral: 
    (f(xi) / 2T^1.5) * [nu_jl(T) + n_il(E-T) - n_il(E)]
    E_curr here is the current energy point E_next.
    nu_i_E_curr is nu_il(E_next), passed as a value to avoid re-evaluation.
    """
    xi = np.sqrt(U_val * E_curr * T)
    f_val = f_winterbon(xi)
    
    val_j_T = nu_func_j(T)
    E_minus_T = E_curr - T
    val_i_E_minus_T = nu_func_i(E_minus_T) if E_minus_T > 0 else 0.0
        
    bracket = val_j_T + val_i_E_minus_T - nu_i_E_curr
    # bracket = max(0, bracket)
    
    T_safe = np.maximum(T, 1e-10) 
    kernel = f_val / (2 * T_safe**1.5)
    
    return kernel * bracket

'''
# =============================================
# 1.4 bis. Define the integrand kernel
#       for each {il} and {jl}
#    + is l==i? for ...
# =============================================

def integrand_kernel_for_quad(T, E_curr, nu_func_j, nu_func_i, nu_i_E_curr, U_val, Edl, isPKA):
    xi = np.sqrt(U_val * E_curr * T)
    f_val = f_winterbon(xi)
    
    val_j_T = nu_func_j(T)
    E_minus_T = E_curr - T
    if isPKA and T > Edl:
        val_j_T = val_j_T + 1
        
    val_i_E_minus_T = nu_func_i(E_minus_T) if E_minus_T > 0 else 0.0
        
    bracket = val_j_T + val_i_E_minus_T - nu_i_E_curr
    # bracket = max(0, bracket)
    
    T_safe = np.maximum(T, 1e-10) 
    kernel = f_val / (2 * T_safe**1.5)
    
    return kernel * bracket
'''

# =============================================
# =============================================
# 2. Main Solver for Number of Displacements
# =============================================
#   For accelerating the calculations,
#       one may copy this function for
#       ion projectiles.
#   Many 'if's can be removed if it is
#       for only PKAs
# =============================================
#
# The first parameter material_name can be arbitrary
# It is used only for saving the results in an
# ASCII file named {material_name}_disp.dat
#

def solve_displacements(material_name, elements, stoichiometry, Eds, 
                                        E_max=1e7, steps_decade=10):
    
    n_types = len(elements)
    G, U, M = calc_parameters(elements, stoichiometry)
    
    # 1. Create a unified energy grid that covers all Eds
    E_min_global = min(Eds)
    if E_min_global < 1:
        E_min_global = min(e for e in Eds if e > 0) ### if include a projectile
    
    num_decades = np.log10(E_max) - np.log10(E_min_global)
    n_steps = int(num_decades * steps_decade) + 1
    E_grid_base = np.logspace(np.log10(E_min_global), np.log10(E_max), n_steps)

    # Insert Eds into the grid to perfectly capture thresholds
    # Add E_min_global-0.01 to explicitly show 0 for E < Ed
    E_grid_0 = np.unique(np.sort(np.concatenate([[E_min_global-0.01], E_grid_base, Eds])))

    # Insert all break points (not used for convergent calculations)
    # thrs = [ Eds[i] + Eds[l]/M[i,l] for i in range(n_types) for l in range(n_types)]
    # E_grid_0 = np.unique(np.sort(np.concatenate([[E_min_global-0.01], E_grid_base, Eds, thrs])))

    ### Avoid mathematically equal but slightly different
    ### due to floating-point precision, which can cause
    ### failure due to the cubic Hermit spline interpolation
    E_grid = np.unique(np.round(E_grid_0, decimals=5))
    n_steps = len(E_grid)

    # N_vals shape: (PKA_type_i, Target_type_l, Energy_index_k)
    # 3D array store the results n_il(E)
    N_vals = np.zeros((n_types, n_types, n_steps))
    
    # Solve n independent systems (one for each target displaced atom 'l')
    for l in range(n_types):
        Ed_l = Eds[l]
        start_idx = np.searchsorted(E_grid, Ed_l)
        
        y_vals = np.zeros((n_types, n_steps))
        y_primes = np.zeros((n_types, n_steps))
        
        # Initialization at threshold E = Ed_l
        for i in range(n_types):
            y_vals[i, start_idx] = 1.0 if i == l else 0.0
            # y_primes[i, start_idx] = 0.0  # Assumed flat plateau near threshold
            
        print(f"Solving Displacements for Target Atom [{l}] (Ed = {Ed_l} eV)...")

        # Integration Loop for E > Ed_l
        for k in range(start_idx, n_steps - 1):
            E_curr = E_grid[k]
            E_next = E_grid[k+1]
            h = E_next - E_curr
            # print (f'Starting E = {E_next} eV')
            
            def residual_func(x_primes):
                """
                x_primes contains [nu'_0l, nu'_1l, ... nu'_{n-1}l] at E_next.
                We calculate n_next using Hermite interpolation formula.
                Then calculate residuals.
                """
                curr_primes = x_primes
                curr_vals = np.zeros(n_types)
                for i in range(n_types):
                    curr_vals[i] = y_vals[i, k] + (h/2.0) * (y_primes[i, k] + curr_primes[i])
                
                temp_splines = []
                for i in range(n_types):
                    e_hist = np.append(E_grid[start_idx : k+1], E_next)
                    v_hist = np.append(y_vals[i, start_idx : k+1], curr_vals[i])
                    p_hist = np.append(y_primes[i, start_idx : k+1], curr_primes[i])
                    
                    base_spline = CubicHermiteSpline(e_hist, v_hist, p_hist)
                    
                    # Wrapper ensuring strict 0 value for E < Ed_l
                    class StepWrapper:
                        def __init__(self, spl, threshold):
                            self.spl = spl
                            self.threshold = threshold
                        def __call__(self, E):
                            if isinstance(E, np.ndarray):
                                res = np.zeros_like(E)
                                mask = E >= self.threshold - 1e-9
                                res[mask] = self.spl(E[mask])
                                return res
                            else:
                                return float(self.spl(E)) if E >= self.threshold - 1e-9 else 0.0
                                
                    temp_splines.append(StepWrapper(base_spline, Ed_l))

                ## Calculate residual R
                residuals = np.zeros(n_types)
                for i in range(n_types):
                    rhs_sum = 0.0
                    nu_i_E_next = curr_vals[i]
                    
                    for j in range(n_types):
                        limit = M[i, j] * E_next
                        if limit > 0:
                            # Explicitly mark breakpoints to help `quad` handle the jump at Ed
                            break_points = []
                            if 0 < Ed_l < limit: 
                                break_points.append(Ed_l)
                            if 0 < E_next - Ed_l < limit: 
                                break_points.append(E_next - Ed_l)

                            ### For a projectile, G[i, j] == 0, no need to calculate the integral
                            if G[i, j] < 1e-10:
                                integral = 0

                            else:
                                #'''
                                integral, err = quad(
                                    integrand_kernel_for_quad, 
                                    0., limit,  # Lower limit = 0
                                    args=(E_next, temp_splines[j], temp_splines[i], nu_i_E_next, U[i, j]),
                                    points=break_points,
                                    limit = 200 ### maximum points for integration
                                )

                                '''
                                ### using 1.4 bis kernel
                                integral, err = quad(
                                    integrand_kernel_for_quad, 
                                    0., limit,  # Lower limit = 0
                                    args=(E_next, temp_splines[j], temp_splines[i], nu_i_E_next, U[i, j], Eds[l], j==l),
                                    points=break_points
                                )
                                '''

                            if not np.isfinite(integral):
                                return np.full(n_types, np.nan) 
                            
                            rhs_sum += G[i, j] * integral

                    # Eq (7): E * nu' = Sum(...)
                    residuals[i] = E_next * curr_primes[i] - rhs_sum
                    
                return residuals

            ## predictor
            guess_slope = y_primes[:, k]
            
            ## corrector: using Powell’s hybrid root-finding algorithm
            sol = root(residual_func, guess_slope, method='hybr', tol=1e-3)
            
            if not sol.success:
                print(f"  Warning: Target [{l}], convergence failed at E={E_next:.2e} eV")
                
            final_primes = sol.x
            
            ### direct correction on negative derivaties
            # final_primes = [max(0, val) for val in sol.x]
            # y_primes[:, k+1] = final_primes
            
            ### add constraints Eq. (14)
            for i in range(n_types):
                # if E_next < Eds[l]/M[i,l]:
                if E_next < Eds[l]/M[i,l] + Eds[i]:
                    final_primes[i] = 0
                y_vals[i, k+1] = y_vals[i, k] + (h/2.0) * (y_primes[i, k] + final_primes[i])

            '''
            ### test and print derivatives at thresholds
            for i in range(n_types):
                # if E_next < Eds[l]/M[i,l]:
                if E_next < Eds[l]/M[i,l] + Eds[i]:
                    final_primes[i] = 0
                    y_vals[i, k+1] = i==l
                    print (E_next, i, l)
                elif E_next == Eds[l]/M[i,l] + Eds[i]:
                    y_vals[i, k+1] = i == l
                    print (final_primes[i], E_next, i, l)
                    # final_primes[i] = max(final_primes[i], 0)
                else:
                    y_vals[i, k+1] = y_vals[i, k] + (h/2.0) * (y_primes[i, k] + final_primes[i])
            ### end of the test
            '''
            y_primes[:, k+1] = final_primes
            
        # Store solutions for target l
        N_vals[:, l, :] = y_vals

    # Output routine
    output_cols = [E_grid]
    header = "E_PKA(eV) "
    for i in range(n_types):
        for l in range(n_types):
            output_cols.append(N_vals[i, l, :])
            header += f"n_PKA{i}_Target{l} "
            
    # Save results in a 2D array
    data_to_save = np.column_stack(output_cols)
    np.savetxt(f'{material_name}_disp.dat', data_to_save, fmt='%.4e', header=header)

    return E_grid, N_vals

'''
# ==========================================
# 3. Example Usage: Al2O3 Displacements
# ==========================================

if __name__ == "__main__":
    elements = [get_atomic_data('Al'), get_atomic_data('O')]
    stoichiometry = [2, 3] # Al2O3
    labels = ["Al", "O"]
    
    # Set representative Threshold Displacement Energies (Eds) for Al and O
    Eds = [18, 75]  
    
    print("Solving Number of Displacements for Al2O3...")
    E, n_vals = solve_displacements("Al2O3", elements, stoichiometry, Eds, E_max=1e6)
    
    # Plotting results
    plt.figure(figsize=(9, 6))
    
    # Plot PKA = Al
    plt.loglog(E, n_vals[0, 0, :], label=rf"{labels[0]} $\rightarrow$ {labels[0]}", color='blue', linestyle='-')
    plt.loglog(E, n_vals[0, 1, :], label=rf"{labels[0]} $\rightarrow$ {labels[1]}", color='blue', linestyle='-.')
        
    # Plot PKA = O
    plt.loglog(E, n_vals[1, 0, :], label=rf"{labels[1]} $\rightarrow$ {labels[0]}", color='red', linestyle='--')
    plt.loglog(E, n_vals[1, 1, :], label=rf"{labels[1]} $\rightarrow$ {labels[1]}", color='red', linestyle=':')

    plt.xlim(1e1, 1e6)
    plt.ylim(1e-1, 1e4)
    plt.xlabel("PKA energy (eV)")
    plt.ylabel("Number of Displacements $n_{il}$")
    #plt.title(r"Number of Displacements in Al$_2$O$_3$")
    #plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ex_Al2O3.png', dpi=200)
    plt.show()
'''
