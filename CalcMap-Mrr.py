# Read the evaluation file
evaluation_file_path = r"C:/Users/User 2004/Desktop/New folder (2)/relevant_newle.txt"
evaluation_data = []

with open(evaluation_file_path, 'r') as evaluation_file:
    for line in evaluation_file:
        line = line.strip()
        if line:
            evaluation_data.append(line)

# Process the evaluation data
query_id = None
num_relevant = 0
num_returned = 0
total = 0
average_precisions = []
reciprocal_ranks = []

output_file_path = r"C:/Users/User 2004/Desktop/New folder (2)/results33.txt"

with open(output_file_path, 'w') as output_file:
    for line in evaluation_data:
        if line.startswith("Query ID:"):
            # Check if there are previous query results to process
            if query_id is not None:
                # Calculate precision and recall for the previous query
                precision = num_relevant / num_returned if num_returned > 0 else 0.0
                recall = num_relevant / total if total > 0 else 0.0

                # Calculate average precision for the previous query
                average_precision = precision if num_relevant > 0 else 0.0
                average_precisions.append(average_precision)

                # Write precision, recall, average precision, and reciprocal rank to the output file
                output_file.write("Query ID: {}\n".format(query_id))
                output_file.write("Precision: {:.10f}\n".format(precision))
                output_file.write("Recall: {:.10f}\n".format(recall))

                output_file.write("---------------\n")

                # Reset the values for the next query
                num_relevant = 0
                num_returned = 0

            # Extract the query ID for the current query
            query_id = int(line.split(":")[1].strip())

        elif line.startswith("Relevant Documents:"):
            # Extract the number of relevant documents for the current query
            num_relevant = int(line.split(":")[1].strip())

        elif line.startswith("Total Returned Documents:"):
            # Extract the number of returned documents for the current query
            num_returned = int(line.split(":")[1].strip())

        elif line.startswith("Total:"):
            # Extract the total number of documents for the current query
            total = int(line.split(":")[1].strip())

    # Calculate precision and recall for the last query in the file
    precision = num_relevant / num_returned if num_returned > 0 else 0.0
    recall = num_relevant / total if total > 0 else 0.0

    # Calculate average precision for the last query
    average_precision = precision if num_relevant > 0 else 0.0
    average_precisions.append(average_precision)

    # Write precision, recall, average precision, and reciprocal rank for the last query to the output file
    output_file.write("Query ID: {}\n".format(query_id))
    output_file.write("Precision: {:.10f}\n".format(precision))
    output_file.write("Recall: {:.10f}\n".format(recall))

    output_file.write("---------------\n")

    map_score = sum(average_precisions) / len(average_precisions)

    # Write MAP and MRR to the output file
    output_file.write("MAP: {:.10f}\n".format(map_score))

print("MAP:", map_score)

# Print success message
print("Evaluation results saved to:", output_file_path)
