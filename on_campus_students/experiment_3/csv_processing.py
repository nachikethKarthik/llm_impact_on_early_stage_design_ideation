import csv

# Assuming the CSV file is named 'data.csv'
filename = 'D:/research/llm_impact_on_early_stage_design_ideation/on_campus_students/experiment_3/prompt_data.csv'

# Create an empty dictionary to store the data
user_data = {}

# Read the CSV file
with open(filename, 'r') as file:
    # Create a CSV reader
    csv_reader = csv.reader(file)

    # Initialize the user ID counter
    user_id = 1

    # Initialize the variable to store prompts for each user
    prompts = []

    # Iterate over each row in the CSV file
    for row in csv_reader:
        # Check if the row is empty (indicates a new set of prompts)
        if len(row) == 0:
            # Assign the prompts to the user ID and reset the prompts list
            user_data[f'U{user_id:02}'] = prompts
            prompts = []

            # Increment the user ID counter
            user_id += 1
        else:
            # Add the prompt to the list
            prompts.append(row[0])

# Assign the last set of prompts (if any)
if len(prompts) > 0:
    user_data[f'U{user_id:02}'] = prompts

# # Print the user data
# for user_id, prompts in user_data.items():
#     print(f'User ID: {user_id}')
#     print('Prompts:')
#     for prompt in prompts:
#         print(prompt)
#     print()
