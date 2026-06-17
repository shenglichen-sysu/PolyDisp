
import PolyDisp

#####-------------------- Mandatory parameter --------------------#####

labels = ["Fe", "Cr", "Al"] # FeCrAl
Eds = [40, 40, 40]         # TDE for all elements
stoichiometry = [0.702, 0.201, 0.097]

#####-------------------- PolyDisp calculation --------------------#####

elements = [PolyDisp.get_atomic_data(el) for el in labels]

E, n_vals = PolyDisp.solve_polyatomic_displacement("tmp", \
                                    elements, stoichiometry, Eds)

'''

#####-------------------- Plot n_i(E) versus E --------------------#####

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','no-latex'])


print(f"Total runtime for FeCrAl is {end_t - start_t:.2f} seconds")

plt.loglog(E, n_vals[0, 0, :] + n_vals[0, 1, :] + n_vals[0, 2, :], \
           label=rf"{labels[0]} PKA", linestyle='-')
plt.loglog(E, n_vals[1, 0, :] + n_vals[1, 1, :] + n_vals[1, 2, :], \
           label=rf"{labels[1]} PKA", linestyle='--')
plt.loglog(E, n_vals[2, 0, :] + n_vals[2, 1, :] + n_vals[2, 2, :], \
           label=rf"{labels[2]} PKA", linestyle=':')


plt.text(x=.7, y=.1, s=r'FeCrAl', transform=plt.gca().transAxes)
plt.xlim(1e1, 1e7)
plt.ylim(1e-1, 1e4)
plt.xlabel("PKA energy (eV)")
plt.ylabel("Number of displacements")
plt.legend()
plt.tight_layout()
plt.savefig('FeCrAl.png', dpi=200)
plt.show()
'''

