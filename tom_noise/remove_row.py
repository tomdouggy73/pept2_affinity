{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "d97d6e9c-0300-4bdf-b022-eeb220a8321c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Removed the top row from 'export_affinities.csv' and saved the result to 'output_affinities.csv'.\n"
     ]
    }
   ],
   "source": [
    "import csv\n",
    "\n",
    "def remove_top_row(csv_file, output_file=None):\n",
    "    # Set output_file to the same as csv_file if not provided\n",
    "    if output_file is None:\n",
    "        output_file = csv_file\n",
    "\n",
    "    # Open the input CSV file and read the contents\n",
    "    with open(csv_file, 'r', newline='') as infile:\n",
    "        reader = csv.reader(infile)\n",
    "        \n",
    "        # Skip the first row (header or unwanted row)\n",
    "        next(reader)  # This line effectively removes the top row\n",
    "\n",
    "        # Open the output CSV file for writing\n",
    "        with open(output_file, 'w', newline='') as outfile:\n",
    "            writer = csv.writer(outfile)\n",
    "\n",
    "            # Write the remaining rows to the output file\n",
    "            for row in reader:\n",
    "                writer.writerow(row)\n",
    "\n",
    "    print(f\"Removed the top row from '{csv_file}' and saved the result to '{output_file}'.\")\n",
    "\n",
    "# Example usage\n",
    "remove_top_row('export_affinities.csv', 'output_affinities.csv')  # Replace with your actual file names"
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
