from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, state_fidelity, SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
import numpy as np
import matplotlib.pyplot as plt


# --- 6 cardinal input states ---
cardinal_states = [
    Statevector.from_label('0'),                 # |0>
    Statevector.from_label('1'),                 # |1>
    Statevector([1/np.sqrt(2), 1/np.sqrt(2)]),  # |+> = (|0> + |1>)/√2
    Statevector([1/np.sqrt(2), -1/np.sqrt(2)]), # |-> = (|0> - |1>)/√2
    Statevector([1/np.sqrt(2), 1j/np.sqrt(2)]), # |i> = (|0> + i|1>)/√2
    Statevector([1/np.sqrt(2), -1j/np.sqrt(2)]) # |-i> = (|0> - i|1>)/√2
]

def calculate_gate_fidelity(ideal_gate, custom_gate, cardinal_states, error_list):
    '''
    Calculate the average gate fidelity between an ideal gate and a custom gate
    over a set of cardinal input states.
    '''
    
    avg_fidelities_dif_errors = []

    for e in error_list:
        fidelities = []
        for state in cardinal_states:
            # Apply ideal gate
            ideal_qc = QuantumCircuit(1)
            ideal_gate(ideal_qc, 0, params=None)
            ideal_sv = state.evolve(ideal_qc)

            # Apply custom gate
            custom_qc = QuantumCircuit(1)
            custom_gate(custom_qc, 0, params=None, error=e)
            custom_sv = state.evolve(custom_qc)

            # Compute fidelity for this input state
            F = state_fidelity(ideal_sv, custom_sv)

            fidelities.append(F)

        # Average fidelity over the 6 cardinal states
        avg_fidelities_dif_errors.append(np.mean(fidelities))
    
    return avg_fidelities_dif_errors
    

def plot(error_list, gate_fidelities, gate_names, target_fidelity=0.99):
    # Plot results
    plt.figure()
    for i, fidelities in enumerate(gate_fidelities):
        plt.plot(error_list, fidelities, linestyle= '-', label=gate_names[i])
    plt.axhline(y=target_fidelity, color='r', linestyle='--', label='Target Fidelity')
    plt.xlabel('Error Parameter')
    plt.ylabel('Gate Fidelity')
    plt.title('Gate Fidelity vs Error Parameter')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":

    #error_list = [1e-4, 1e-3, 1e-2, 1e-1]
    error_list = np.linspace(0.001, 0.4, 100)
    target_fidelity = 0.99

    ########################################################################################
    '''
    ADD MORE CUSTOM GATES AND IDEAL GATES AS NEEDED (copy paste and modify). 
    AND ADD THEM TO THE LISTS BELOW.
    '''  

    # --- Define your custom gates ---
    def X(qc, qubit=0, params=None, error=None):
        # Example: Slightly imperfect X rotation
        qc.rx(np.pi + error, qubit) # Implement the three rotation gates adding a small error

    # --- Ideal gate (for comparison) ---
    def ideal_X(qc, qubit=0, params=None):
        qc.rx(np.pi, qubit)  # perfect pi rotation

    def Y(qc, qubit=0, params=None, error=None):
        # Example: Slightly imperfect X rotation
        qc.ry(np.pi + error, qubit) # Implement the three rotation gates adding a small error

    # --- Ideal gate (for comparison) ---
    def ideal_Y(qc, qubit=0, params=None):
        qc.ry(np.pi, qubit)  # perfect pi rotation

    def Z(qc, qubit=0, params=None, error=None):
        # Example: Slightly imperfect X rotation
        qc.rz(np.pi + error, qubit) # Implement the three rotation gates adding a small error

    # --- Ideal gate (for comparison) ---
    def ideal_Z(qc, qubit=0, params=None):
        qc.rz(np.pi, qubit)  # perfect pi rotation

    def H(qc, qubit=0, params=None, error=None):
        # Example: Slightly imperfect X rotation
        qc.ry(np.pi/2 + error, qubit) # Implement the three rotation gates adding a small error
        qc.rz(np.pi + error, qubit)

    # --- Ideal gate (for comparison) ---
    def ideal_H(qc, qubit=0, params=None):
        qc.ry(np.pi/2, qubit) # Implement the three rotation gates adding a small error
        qc.rz(np.pi, qubit)


    # Define a list of custom gates to evaluate
    custom_gates = [X, Y, Z, H]  # Add more custom gates as needed
    ideal_gates = [ideal_X,ideal_Y,ideal_Z,ideal_H]  # Corresponding ideal gates
    gate_names = ['X', 'Y', 'Z', 'H']  # Names for plotting

    ########################################################################################
    
    # Calculate gate fidelities
    gate_fidelities = []
    for i in range(len(custom_gates)):
        avg_fidelities_dif_errors = calculate_gate_fidelity(ideal_gates[i], custom_gates[i], cardinal_states, error_list)
        gate_fidelities.append(avg_fidelities_dif_errors)

    # Plot results
    plot(error_list, gate_fidelities, gate_names, target_fidelity)
    


    

    