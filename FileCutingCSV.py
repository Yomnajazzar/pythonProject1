import pandas as pd
import os
df = pd.read_csv('collection.csv')
for i, row in df.iterrows():
    text = row['text_right']
    filename = f"{row['id_right']}.txt"
    # with open(f"{filename}", "w") as f:
    #     f.write(text)
    # Create a folder to store the documents
if not os.path.exists('documents'):
    os.makedirs('documents')

# Loop through the rows of the DataFrame and save each document as a separate file
for i, row in df.iterrows():
    # Extract the relevant information for each document
    text = row['text_right']
    id = row['id_right']

    # Save the document as a file with the name of its id
    filename = os.path.join('documents', f"{id}.txt")
    with open(filename, "w") as f:
        f.write(text)