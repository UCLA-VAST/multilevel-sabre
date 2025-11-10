from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import SabreLayout
from qiskit.converters import circuit_to_dag, dag_to_circuit

def read_qasm(file_name):
    qc = QuantumCircuit.from_qasm_file(file_name)
    num_of_qubits=qc.num_qubits
    num_of_clbits=qc.num_clbits
    result_qc=QuantumCircuit(num_of_qubits,num_of_clbits)
    two_qubit_qc=QuantumCircuit(num_of_qubits,num_of_clbits)

    print("Number of gates",len(qc.data))

    qubits_correspondence_dict = {q: i for i, q in enumerate(qc.qubits)}
    clbits_correspondence_dict = {c: i for i, c in enumerate(qc.clbits)}

    for instr, qargs, cargs in qc.data:
        new_qargs = [result_qc.qubits[qubits_correspondence_dict[q]] for q in qargs]
        new_cargs = [result_qc.clbits[clbits_correspondence_dict[c]] for c in cargs]
        if instr.num_qubits<=2 and instr.name!="barrier":
            result_qc.append(instr, new_qargs, new_cargs)

            if instr.num_qubits==2:  # Ensure it's a two-qubit gate
                two_qubit_qc.append(instr, new_qargs)

    return result_qc, two_qubit_qc


def count_swaps(circuit):
    """Count the number of SWAP gates in a circuit.
    
    Args:
        circuit (QuantumCircuit): The circuit to count SWAPs in
        
    Returns:
        int: Number of SWAP gates in the circuit
    """
    return sum(1 for instruction in circuit.data if instruction.operation.name == 'swap')

def sabre(circuit, coupling, number_of_trial, random_seed):
    """Run SABRE algorithm on a circuit.
    
    Args:
        circuit (QuantumCircuit): The circuit to optimize
        coupling (list): List of coupling pairs for the hardware
        number_of_trial (int): Number of trials to run
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (number of SWAPs, optimized circuit)
    """
    qc = circuit
    device = CouplingMap(couplinglist=coupling, description="sabre_test")
    num_program_qubit = qc.num_qubits
    num_classical_bits = qc.num_clbits
    num_physical_qubit = max(max(i) for i in coupling) + 1
    if num_physical_qubit > num_program_qubit:
        temp_qc = QuantumCircuit(num_physical_qubit, num_classical_bits)
        temp_qc.compose(qc, inplace=True)
        qc = temp_qc

    sabre_layout = SabreLayout(coupling_map=device, seed=random_seed, layout_trials=number_of_trial, skip_routing=False)
    out_dag = sabre_layout.run(circuit_to_dag(qc))
    sabre_cir = dag_to_circuit(out_dag)

    return count_swaps(sabre_cir), sabre_cir

EAGLE_COUPLING = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)
    (9, 10), (10, 11), (11, 12), (12, 13),
    (0, 14), (14, 18), (4, 15), (15, 22), (8, 16), (16, 26), (12, 17), (17, 30),
    (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26),
    (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32),
    (20, 33), (33, 39), (24, 34), (34, 43), (28, 35), (35, 47), (32, 36), (36, 51),
    (37, 38), (38, 39), (39, 40), (40, 41), (41, 42), (42, 43), (43, 44), (44, 45),
    (45, 46), (46, 47), (47, 48), (48, 49), (49, 50), (50, 51),
    (37, 52), (52, 56), (41, 53), (53, 60), (45, 54), (54, 64), (49, 55), (55, 68),
    (56, 57), (57, 58), (58, 59), (59, 60), (60, 61), (61, 62), (62, 63), (63, 64),
    (64, 65), (65, 66), (66, 67), (67, 68), (68, 69), (69, 70),
    (58, 71), (71, 77), (62, 72), (72, 81), (66, 73), (73, 85), (70, 74), (74, 89),
    (75, 76), (76, 77), (77, 78), (78, 79), (79, 80), (80, 81), (81, 82), (82, 83),
    (83, 84), (84, 85), (85, 86), (86, 87), (87, 88), (88, 89),
    (75, 90), (90, 94), (79, 91), (91, 98), (83, 92), (92, 102), (87, 93), (93, 106),
    (94, 95), (95, 96), (96, 97), (97, 98), (98, 99), (99, 100), (100, 101), (101, 102),
    (102, 103), (103, 104), (104, 105), (105, 106), (106, 107), (107, 108),
    (96, 109), (109, 114), (100, 110), (110, 118), (104, 111), (111, 112), (108, 112), (112, 126),
    (113, 114), (114, 115), (115, 116), (116, 117), (117, 118), (118, 119), (119, 120),
    (120, 121), (121, 122), (122, 123), (123, 124), (124, 125), (125, 126)
]

WILLOW_COUPLING = [
    (0,1), (1,2),
    (3,4), (4,5), (5,6), (0,5), (1,4), (2,3),
    (7,8), (8,9), (9,10), (10,11), (11,12), (12,13), (6,8), (5,9), (4,10), (3,11),
    (14,15), (15,16), (16,17), (17,18), (18,19), (19,20), (20,21), (13,14), (15,12),
    (16,11), (17,10), (18,9), (19,8), (20,7),
    (22,23), (23,24), (24,25), (25,26), (26,27), (27,28), (28,29), (29,30), (30,31),
    (31,32), (23,21), (24,20), (25,19), (26,18), (27,17), (28,16), (29,15), (30,14),
    (33,34), (34,35), (35,36), (36,37), (37,38), (38,39), (39,40), (40,41), (41,42),
    (42,43), (43,44), (33,32), (34,31), (35,30), (36,29), (37,28), (38,27), (39,26),
    (40,25), (41,24), (42,23), (43,22),
    (45,46), (46,47), (47,48), (48,49), (49,50), (50,51), (51,52), (52,53), (53,54),
    (54,55), (55,56), (56,57), (57,58), (58,59), (46,44), (47,43), (48,42), (49,41),
    (50,40), (51,39), (52,38), (53,37), (54,36), (55,35), (56,34), (57,33),
    (60,61), (61,62), (62,63), (63,64), (64,65), (65,66), (66,67), (67,68), (68,69),
    (69,70), (70,71), (60,58), (61,57), (62,56), (63,55), (64,54), (65,53), (66,52),
    (67,51), (68,50), (69,49), (70,48), (71,47),
    (72,73), (73,74), (74,75), (75,76), (76,77), (77,78), (78,79), (79,80), (80,81),
    (81,82), (72,71), (73,70), (74,69), (75,68), (76,67), (77,66), (78,65), (79,64),
    (80,63), (81,62), (82,61),
    (83,84), (84,85), (85,86), (86,87), (87,88), (88,89), (89,90), (83,81), (84,80),
    (85,79), (86,78), (87,77), (88,76), (89,75), (90,74),
    (91,92), (92,93), (93,94), (94,95), (95,96), (96,97), (91,90), (92,89), (93,88),
    (94,87), (95,86), (96,85), (97,84),
    (98,99), (99,100), (100,101), (98,96), (99,95), (100,94), (101,93),
    (102,103), (103,104), (102,101), (103,100), (104,99)
]
