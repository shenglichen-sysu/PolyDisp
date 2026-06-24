# run_disp.py - Entry point for PolyDisp with key-value input file

import sys
import PolyDisp

def parse_input_file(input_file="inputDisp.txt"):
    """
    Parse key-value formatted input file.
    Expected keys: name, elements, stoichiometry, Eds, E_max (optional)
    Returns: (material_name, elements_list, stoichiometry_list, Eds_list, E_max)
    """
    data = {}
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            data[key.strip()] = value.strip()

    required = ['name', 'elements', 'stoichiometry', 'Eds']
    for k in required:
        if k not in data:
            raise ValueError(f"Missing required key '{k}' in input file")

    material_name = data['name']
    elem_symbols = data['elements'].split()
    stoich_vals = [float(x) for x in data['stoichiometry'].split()]
    eds_vals = [float(x) for x in data['Eds'].split()]

    if len(elem_symbols) != len(stoich_vals) or len(elem_symbols) != len(eds_vals):
        raise ValueError("Number of elements, stoichiometry, and Eds must match")

    elements = []
    for sym in elem_symbols:
        atom = PolyDisp.get_atomic_data(sym)
        if atom is None:
            raise ValueError(f"Unknown element symbol: {sym}")
        elements.append(atom)

    E_max = float(data.get('E_max', '1e7'))
    return material_name, elements, stoich_vals, eds_vals, E_max

if __name__ == "__main__":
    try:
        # Allow user to specify input file path as command-line argument
        input_file = sys.argv[1] if len(sys.argv) > 1 else "inputDisp.txt"
        name, elems, stoich, eds, Emax = parse_input_file(input_file)

        print(f"Material: {name}")
        print(f"Elements: {[e[0] for e in elems]}")
        print(f"Stoichiometry: {stoich}")
        print(f"Eds: {eds} eV")
        print(f"E_max = {Emax:.3e} eV")
        print("Running displacement calculation...")

        E_grid, n_vals = PolyDisp.solve_displacements(
            name, elems, stoich, eds, E_max=Emax, steps_decade=10
        )

        print(f"Calculation finished! Results saved to {name}_disp.dat")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
