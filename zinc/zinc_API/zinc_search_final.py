import pandas as pd
import requests
import time

# Load your SMILES strings from a CSV file
input_file = 'final_top_1000.csv'  # Replace with your actual file path
smiles_data = pd.read_csv(input_file, header=None)  # Assuming no header
smiles_list = smiles_data[0].values.tolist()  # Convert to a list

# Initialize an empty list to store compound data
compound_list = []

# Loop through each SMILES string and search in ZINC database
for smile in smiles_list:
    # Search ZINC using the SMILES string
    url = f"https://zinc15.docking.org/substances/search/?q={smile}"  # Adjust the URL as needed
    response = requests.get(url)
    
    # Introduce a small delay to prevent hitting rate limits
    time.sleep(2)

    if response.status_code == 200:
        try:
            # Parse the response JSON
            data = response.json()
            compounds = data.get('substances', [])
            print(response.text) 
            
            for compound in compounds:
                # Extract desired information
                compound_info = {
                    'ZINC_ID': compound.get('zinc_id'),
                    'SMILES': compound.get('smiles'),
                    'Name': compound.get('name'),
                    'Molecular_Weight': compound.get('molecular_weight'),
                    # Add more fields as necessary
                }
                compound_list.append(compound_info)
        except ValueError:
            print(f"Error parsing JSON for SMILES {smile}: {response.text}")
    else:
        print(f"Error fetching data for SMILES {smile}: {response.status_code} - {response.text}")

# Create a DataFrame from the compound list
compound_df = pd.DataFrame(compound_list)

# Save the results to a CSV file
output_file = 'zinc_fetch.csv'  # Replace with your desired output path
compound_df.to_csv(output_file, index=False)

print(f"Compound data saved to '{output_file}'.")

