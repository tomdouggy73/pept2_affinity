import csv

def convert_dat_to_csv(exp_affinities, export_affinities, delimiter=None):
    with open(exp_affinities, 'r') as dat_f:
        with open(export_affinities, 'w', newline='') as csv_f:
            writer = csv.writer(csv_f)
            
            # Loop through each line in the .dat file
            for line in dat_f:
                row = line.strip().split(delimiter) if delimiter else line.strip().split()
                writer.writerow(row)

    print(f"Conversion complete: '{exp_affinities}' -> '{export_affinities}'")

# Example usage
convert_dat_to_csv('exp_affinities.dat', 'export_affinities.csv')

