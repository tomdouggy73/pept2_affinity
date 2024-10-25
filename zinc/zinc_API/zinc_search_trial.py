import pandas as pd
import requests
import time

# Load your SMILES strings from a CSV file
input_file = 'final_top_1000.csv'  # Replace with your actual file path
smiles_data = pd.read_csv(input_file, header=None)  # Assuming no header
smiles_list = smiles_data[0].values.tolist()  # Convert to a list

# Test with just one SMILES (use the first SMILES in the list)
test_smile = smiles_list[0]  # Replace 0 with any other index if you want to test a different SMILES

# Initialize an empty list to store compound data
compound_list = []

# Search ZINC using the SMILES string
url = f"https://zinc15.docking.org/substances/search/?structure.smiles={test_smile}"  # Adjust the URL as needed
response = requests.get(url)

# Introduce a small delay to prevent hitting rate limits
time.sleep(5)  # Increase delay if necessary

# Check the response status
if response.status_code == 200:
    # Check the Content-Type header
    content_type = response.headers.get('Content-Type', '')
    
    # Print the content type for debugging
    print(f"Content-Type: {content_type}")

    if 'application/json' in content_type:
        # Parse the response as JSON
        try:
            data = response.json()  # This will work if the response is indeed JSON
            compounds = data.get('substances', [])
            if not compounds:
                print(f"No compounds found for SMILES {test_smile}")
            else:
                for compound in compounds:
                    # Extract desired information
                    compound_info = {
                        'ZINC_ID': compound.get('zinc_id'),
                        'SMILES': compound.get('smiles'),
                        'Name': compound.get('name'),
                        'Molecular_Weight': compound.get('molecular_weight'),
                    }
                    compound_list.append(compound_info)

        except ValueError:
            print(f"Error parsing JSON for SMILES {test_smile}: {response.text}")

    elif 'text/html' in content_type:
        # Handle HTML response
        print(f"Received HTML response for SMILES {test_smile}. Here is the content:")
        print(response.text)  # Print the HTML for debugging

    elif 'text/plain' in content_type:
        # Handle plain text response
        print(f"Received plain text response for SMILES {test_smile}. Here is the content:")
        print(response.text)  # Print the text for debugging

    else:
        # Handle unexpected content types
        print(f"Received unexpected content type for SMILES {test_smile}: {content_type}")
        print(response.text)  # Print the raw content for debugging

else:
    print(f"Error fetching data for SMILES {test_smile}: {response.status_code} - {response.text}")

# Optionally, save to CSV
if compound_list:
    compound_df = pd.DataFrame(compound_list)
    output_file = 'zinc_fetch_test.csv'
    compound_df.to_csv(output_file, index=False)
    print(f"Compound data saved to '{output_file}'.")