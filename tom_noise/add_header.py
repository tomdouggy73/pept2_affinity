{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "189e3d56-9ffc-48c9-b382-fc2f419929cb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Header '['affinity']' inserted into 'export_affinities.csv'.\n"
     ]
    }
   ],
   "source": [
    "import csv\n",
    "\n",
    "def insert_header_to_csv(export_affinities, header_row):\n",
    "    # Read the existing data from the CSV file\n",
    "    with open(export_affinities, 'r') as file:\n",
    "        existing_data = file.readlines()\n",
    "\n",
    "    # Open the CSV file for writing (this will overwrite the existing file)\n",
    "    with open(export_affinities, 'w', newline='') as file:\n",
    "        writer = csv.writer(file)\n",
    "\n",
    "        # Write the new header row\n",
    "        writer.writerow(header_row)\n",
    "\n",
    "        # Write the existing data back to the file\n",
    "        for line in existing_data:\n",
    "            # Split the line into a list of values and write to CSV\n",
    "            writer.writerow(line.strip().split(','))  # Adjust delimiter if necessary\n",
    "\n",
    "    print(f\"Header '{header_row}' inserted into '{export_affinities}'.\")\n",
    "\n",
    "# Example usage\n",
    "insert_header_to_csv('export_affinities.csv', ['affinity'])\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
