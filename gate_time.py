import csv
import numpy as np
# Import the data structure and the dictionary from your exact file
from gate_library import GATE_LIBRARY, GateAngles

def generate_gate_durations_csv(filename, j_offset, alpha, V_control):
    """
    Calculates the required pulse duration for each rotation stage based on 
    the exchange interaction J and exports the structured data to a CSV file,
    including a final column for the total composite gate time.
    """
    # Calculate the active exchange interaction: J = J_offset * e^(2 * alpha * V)
    J = 200e6
    
    # Define the CSV table headers
    header = [
        "Gate Name", 
        "t_pulse_1 (ns)", 
        "t_pulse_2 (ns)", 
        "t_pulse_3 (ns)", 
        "t_pulse_4 (ns)", 
        "Total Gate Time (ns)"
    ]
    
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        
        # Iterate sequentially through your imported gate library
        for gate_name, angles in GATE_LIBRARY.items():
            # Physical mapping equation: t = theta / (2 * pi * J)
            # Scaled by 1e12 to directly convert from seconds to picoseconds (ps)
            t1 = (angles.theta1 / (2 * np.pi * J)) * 1e9 if angles.theta1 > 0 else 0.0
            t2 = (angles.theta2 / (2 * np.pi * J)) * 1e9 if angles.theta2 > 0 else 0.0
            t3 = (angles.theta3 / (2 * np.pi * J)) * 1e9 if angles.theta3 > 0 else 0.0
            t4 = (angles.theta4 / (2 * np.pi * J)) * 1e9 if angles.theta4 > 0 else 0.0
            
            # Sum up the non-zero active slices to evaluate total sequential time
            total_gate_time = t1 + t2 + t3 + t4
            
            # Write row with float rounding to 3 decimal places for presentation clarity
            writer.writerow([
                gate_name,
                round(t1, 3),
                round(t2, 3),
                round(t3, 3),
                round(t4, 3),
                round(total_gate_time, 3)
            ])

# =========================================================================
# Execution & Hardware Setup Parameters (Matching your Appendix Data)
# =========================================================================
if __name__ == "__main__":
    # Standard hardware configuration from your specification tables
    J_OFFSET = 10e3      # 10 kHz baseline offset
    ALPHA = 25.0         # 25.0 V^-1 sensitivity coefficient
    V_CONTROL = 0.185    # Example control voltage level across the CDAC matrix (185 mV)

    output_file = "dtc_gate_pulse_durations.csv"
    
    try:
        generate_gate_durations_csv(output_file, J_OFFSET, ALPHA, V_CONTROL)
        print(f"============== Execution Successful ==============")
        print(f"✅ Extracted pulse durations written to: '{output_file}'")
        print(f"Calculated using J = {J_OFFSET/1e3} kHz * exp(2 * {ALPHA} * {V_CONTROL} V)")
    except ImportError:
        print("❌ Error: Could not find 'gate_library.py'. Make sure to save your library data structure in the same directory.")