""" Additional functions"""
#
#DATE = 19082021
#
def return_energy(element):  # given in eV,  (old value)
    """Give the stable reference energy of a pure element
    calculated using VASP 5.4 GGA-PBE 400eV of Cutoff"""
    #
    if element == "Al":
        return -3.74608353    # fcc (old: -2.827352) 
    if element == "Co":
        return -7.035938715   # ferro (-7.1077975)
    if element == "Cr":
        return -18.99331970/2 # bcc af-mag (-2.50)
    if element == "Fe":
        return -8.23696426    # bcc ferro (-8.309665)
    if element == "Mn":
        return -9.026950897 # mag
    if element == "Mo":
        return -10.94833837   # bcc
    if element == "Nb":
        return -10.09141055   # bcc
    if element == "Ni":
        return -5.46814167    # bcc ferro (-5.57014)
    if element == "Os":
        return -22.49693993/2 # hcp (-11.2256)
    if element == "Pt":
        return -6.09686670    # fcc (6.056198)
    if element == "Pu":
        return -14.265855
    if element == "Re":
        return -24.84904818/2 # hcp (-12.422491)
    if element == "Ru":
        return -18.50610393/2 # hcp (-9.2032835)
    if element == "Ta":
        return -11.86187715   # bcc (-11.862407)
    if element == "Tc":
        return -20.74909376/2 # hcp (-10.305959)
    if element == "Ti":
        return -15.52472177/2 # hcp
    if element == "V":
        return -8.94118656    # bcc
    if element == "U":
        return -11.2913
    if element == "W":
        return -13.01799131   # bcc
    if element == "Zr":
        return -17.03944468/2 # hcp (-8.4778285)
#    
def return_heat(row):
    """Calculate the heat of formation of a sigma configuration
    by total energies difference with pure element"""
    #
    E_ref = 2*return_energy(row['X1'])+4*return_energy(row['X2'])+8*return_energy(row['X3'])+8*return_energy(row['X4'])+8*return_energy(row['X5'])
    heat = 96.486*(row['F']-E_ref)/30 # in kJ/mol-at
    return heat
#
def return_heat_meV(row):
    """Calculate the heat of formation of a sigma configuration
    by total energies difference with pure element"""
    #
    E_ref = 2*return_energy(row['X1'])+4*return_energy(row['X2'])+8*return_energy(row['X3'])+8*return_energy(row['X4'])+8*return_energy(row['X5'])
    heat = 1000*(row['F']-E_ref)/30 # in meV/at
    return heat
 #
# fonction
def return_radius(element):
    """Give the atomic radius of element"""
    if element == "Al":
        return 143 
    if element == "Co":
        return 125
    if element == "Cr":
        return 128
    if element == "Fe":
        return 126
    if element == "Mn":
        return 127
    if element == "Mo":
        return 139
    if element == "Nb":
        return 146
    if element == "Ni":
        return 124
    if element == "Pt":
        return 139
    if element == "Re":
        return 137
    if element == "Ru":
        return 134
    if element == "V":
        return 134
    if element == "W":
        return 139
    if element == "Zr":
        return 160
#
# fonction
def return_valen_el(element):
    """Give the number of valence electron"""
    if element == "Al":
        return 3 
    if element == "Co":
        return 9
    if element == "Cr":
        return 6
    if element == "Fe":
        return 8
    if element == "Mn":
        return 7
    if element == "Mo":
        return 6
    if element == "Nb":
        return 5
    if element == "Ni":
        return 10
    if element == "Pt":
        return 10
    if element == "Re":
        return 7
    if element == "Ru":
        return 8
    if element == "V":
        return 5
    if element == "W":
        return 6
    if element == "Zr":
        return 4
#       
