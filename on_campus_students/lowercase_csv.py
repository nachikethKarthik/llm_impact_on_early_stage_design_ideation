import csv

# Open the CSV file
with open('exp_3_unprocessed_data.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    # remove the empty rows
    rows = [row for row in reader if row]


# Convert all text to lowercase
for i in range(len(rows)):
    for j in range(len(rows[i])):
        rows[i][j] = rows[i][j].lower()

# Write the processed text back to the CSV file
with open('exp_3_processed.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, lineterminator='\n')
    writer.writerows(rows)

with open('exp_3_processed.csv', 'r') as file_in, open('exp_3_processed_idea_only.csv', 'w', newline='') as file_out:
    reader = csv.reader(file_in)
    writer = csv.writer(file_out)
    
    # Iterate through each line in the input file
    for line in reader:
        print(f'now processing f{line}')
        # Extract the text part by splitting on the ':' character and getting the second element
        text = line[0].split(':')[1].strip()
        
        # Write the extracted text to the output file
        writer.writerow([text])

